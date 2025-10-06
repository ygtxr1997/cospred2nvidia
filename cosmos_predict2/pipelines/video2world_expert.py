# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, Callable, Tuple, Union

import numpy as np
import torch
from einops import rearrange
from megatron.core import parallel_state

from cosmos_predict2.auxiliary.cosmos_reason1 import CosmosReason1
from cosmos_predict2.auxiliary.text_encoder import CosmosT5TextEncoder
# from cosmos_predict2.configs.base.config_video2world import Video2WorldPipelineConfig
from cosmos_predict2.configs.expert.config import Video2WorldExpertPipelineConfig
from cosmos_predict2.models.utils import load_state_dict
from cosmos_predict2.module.denoiser_scaling import RectifiedFlowScaling
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from cosmos_predict2.pipelines.video2world import TextCondition, DataType
from cosmos_predict2.pipelines.video2world import ConditioningStrategy
from cosmos_predict2.module.denoise_prediction import DenoisePredictionWithAction
from cosmos_predict2.conditioner import ActionCondition, ActionConditioner
from cosmos_predict2.schedulers.rectified_flow_scheduler import RectifiedFlowAB2Scheduler
from cosmos_predict2.utils.context_parallel import cat_outputs_cp, split_inputs_cp
from cosmos_predict2.utils.vis_helpers import save_action_as_image
from imaginaire.utils.io import save_image_or_video
from imaginaire.lazy_config import instantiate
from imaginaire.utils import log, misc
from imaginaire.utils.ema import FastEmaModelUpdater

IS_PREPROCESSED_KEY = "is_preprocessed"
_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", "webp"]
_VIDEO_EXTENSIONS = [".mp4"]
NUM_CONDITIONAL_FRAMES_KEY: str = "num_conditional_frames"
NUM_CONDITIONAL_ACTIONS_KEY: str = "num_conditional_actions"


# Modified: pipelines/video2world.py Video2WorldActionConditionedPipeline
class Video2WorldExpertPipeline(Video2WorldPipeline):
    def __init__(self, device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype)
        # action training related
        self.input_action_key: str = "action"
        self.input_agent_pos_key: str = "agent_pos"
        self.max_obs: int = 5
        self.max_act_out: int = 12
        self.p_all_actions_as_condition: float = 0.5

    @staticmethod
    def from_config(
        config: Video2WorldExpertPipelineConfig,
        dit_path: str = "",
        text_encoder_path: str = "",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        load_ema_to_reg: bool = False,
        load_prompt_refiner: bool = False,
    ) -> Any:
        # Create a pipe
        pipe = Video2WorldExpertPipeline(device=device, torch_dtype=torch_dtype)
        pipe.config = config
        pipe.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.precision]
        pipe.tensor_kwargs = {"device": "cuda", "dtype": pipe.precision}
        log.warning(f"precision {pipe.precision}")

        # 1. set data keys and data information
        pipe.sigma_data = config.sigma_data
        pipe.setup_data_key()

        # 2. setup up diffusion processing and scaling~(pre-condition)
        pipe.scheduler = RectifiedFlowAB2Scheduler(
            sigma_min=config.timestamps.t_min,
            sigma_max=config.timestamps.t_max,
            order=config.timestamps.order,
            t_scaling_factor=config.rectified_flow_t_scaling_factor,
        )

        pipe.scaling = RectifiedFlowScaling(
            pipe.sigma_data, config.rectified_flow_t_scaling_factor, config.rectified_flow_loss_weight_uniform
        )

        # 3. Set up tokenizer
        pipe.tokenizer = instantiate(config.tokenizer)
        assert (
            pipe.tokenizer.latent_ch == pipe.config.state_ch
        ), f"latent_ch {pipe.tokenizer.latent_ch} != state_shape {pipe.config.state_ch}"

        # 4. Load text encoder
        if text_encoder_path:
            # inference
            pipe.text_encoder = CosmosT5TextEncoder(device=device, cache_dir=text_encoder_path)
            pipe.text_encoder.to(device)
        else:
            # training
            pipe.text_encoder = None

        # 5. Initialize conditioner
        pipe.conditioner = instantiate(config.conditioner)
        assert (
            sum(p.numel() for p in pipe.conditioner.parameters() if p.requires_grad) == 0
        ), "conditioner should not have learnable parameters"

        if load_prompt_refiner:
            pipe.prompt_refiner = CosmosReason1(
                checkpoint_dir=config.prompt_refiner_config.checkpoint_dir,
                offload_model_to_cpu=config.prompt_refiner_config.offload_model_to_cpu,
                enabled=config.prompt_refiner_config.enabled,
            )

        if config.guardrail_config.enabled:
            from cosmos_predict2.auxiliary.guardrail.common import presets as guardrail_presets

            pipe.text_guardrail_runner = guardrail_presets.create_text_guardrail_runner(
                config.guardrail_config.checkpoint_dir, config.guardrail_config.offload_model_to_cpu
            )
            pipe.video_guardrail_runner = guardrail_presets.create_video_guardrail_runner(
                config.guardrail_config.checkpoint_dir, config.guardrail_config.offload_model_to_cpu
            )
        else:
            pipe.text_guardrail_runner = None
            pipe.video_guardrail_runner = None

        # 6. Set up DiT
        if dit_path:
            log.info(f"Loading DiT from {dit_path}")
        else:
            log.warning("dit_path not provided, initializing DiT with random weights")
        # with init_weights_on_device():
        # NOTE: we don't load checkpoint on meta device since we have additional action encoder
        dit_config = config.net
        pipe.dit = instantiate(dit_config).eval()  # inference

        # # log.success("[DEBUG][Warning] disable the pretrain loading for faster debugging")
        # if dit_path:
        #     state_dict = load_state_dict(dit_path)
        # # drop net. prefix
        # state_dict_dit_compatible = dict()
        # for k, v in state_dict.items():
        #     if k.startswith("net."):
        #         state_dict_dit_compatible[k[4:]] = v
        #     else:
        #         state_dict_dit_compatible[k] = v
        # pipe.dit.load_state_dict(state_dict_dit_compatible, strict=False, assign=True)
        # del state_dict, state_dict_dit_compatible
        # log.success(f"Successfully loaded DiT from {dit_path}")
        if dit_path:
            # load weights - already materializes tensors
            state_dict = load_state_dict(dit_path)
            prefix_to_load = "net_ema." if load_ema_to_reg else "net."
            # drop net. prefix
            state_dict_dit_compatible = dict()
            for k, v in state_dict.items():
                if k.startswith(prefix_to_load):
                    state_dict_dit_compatible[k[len(prefix_to_load):]] = v
                else:
                    state_dict_dit_compatible[k] = v

            # 获取模型参数名称集合
            model_param_names = set(pipe.dit.state_dict().keys())
            checkpoint_param_names = set(state_dict_dit_compatible.keys())

            # 统计加载情况
            loaded_params = model_param_names & checkpoint_param_names  # 交集：能够加载的参数
            missing_params = model_param_names - checkpoint_param_names  # 差集：模型有但检查点没有的参数
            unused_params = checkpoint_param_names - model_param_names  # 差集：检查点有但模型不需要的参数

            log.info(f"参数加载统计:")
            log.info(f"  - 成功加载的参数: {len(loaded_params)}")
            log.info(f"  - 模型缺失的参数: {len(missing_params)}")
            log.info(f"  - 检查点多余的参数: {len(unused_params)}")

            if missing_params:
                log.warning(
                    f"以下参数在检查点中缺失: {list(missing_params)[:10]}{'...' if len(missing_params) > 10 else ''}")

            if unused_params:
                log.warning(
                    f"以下参数在检查点中多余: {list(unused_params)[:10]}{'...' if len(unused_params) > 10 else ''}")

            pipe.dit.load_state_dict(state_dict_dit_compatible, strict=False, assign=True)
            del state_dict, state_dict_dit_compatible
            log.success(f"Successfully loaded DiT from {dit_path}")
        else:
            pipe.dit = pipe.dit.to(device=device, dtype=torch_dtype)

        # 6-2. Handle EMA
        if config.ema.enabled:
            pipe.dit_ema = instantiate(dit_config).eval()
            pipe.dit_ema.requires_grad_(False)

            pipe.dit_ema_worker = FastEmaModelUpdater()  # default when not using FSDP

            s = config.ema.rate
            pipe.ema_exp_coefficient = np.roots([1, 7, 16 - s**-2, 12 - s**-2]).real.max()
            # copying is only necessary when starting the training at iteration 0.
            # Actual state_dict should be loaded after the pipe is created.
            pipe.dit_ema_worker.copy_to(src_model=pipe.dit, tgt_model=pipe.dit_ema)

        pipe.dit = pipe.dit.to(device=device, dtype=torch_dtype)
        torch.cuda.empty_cache()

        # 7. training states
        if parallel_state.is_initialized():
            pipe.data_parallel_size = parallel_state.get_data_parallel_world_size()
        else:
            pipe.data_parallel_size = 1

        # 8. action related
        pipe.max_obs = config.max_obs
        pipe.max_act_out = config.max_act_out
        pipe.p_all_actions_as_condition = config.p_all_actions_as_condition
        pipe.input_action_key = config.input_action_key
        pipe.input_agent_pos_key = config.input_agent_pos_key

        return pipe

    @torch.no_grad()
    def vis_latent_state(self, latent_B_C_T_H_W, batch_dim: int = 0, save_name: str = "output/tmp_x0.mp4", fps=5):
        vis_raw = self.decode(latent_B_C_T_H_W[batch_dim: batch_dim + 2])  # shape: (B, C, T, H, W), possibly out of [-1, 1]
        save_image_or_video(vis_raw[0, :3].detach().cpu(), save_name, fps=fps)

    @torch.no_grad()
    def vis_action(self, action_B_T_D, batch_dim: int = 0, save_name: str = "output/tmp_action.mp4"):
        save_action_as_image(action_B_T_D[batch_dim].detach().float().detach().cpu().numpy(),
                             save_name)

    @torch.no_grad()
    def vis_sigma(self, sigma_B_T, batch_dim: int = 0, save_name: str = "output/tmp_sigma.png"):
        import matplotlib.pyplot as plt
        import numpy as np
        print("[DEBUG] vis_sigma:", sigma_B_T.shape, sigma_B_T.dtype, sigma_B_T.min(), sigma_B_T.max())
        if sigma_B_T.ndim == 3:
            sigma_B_T = sigma_B_T.squeeze([2])
        elif sigma_B_T.ndim == 5:
            sigma_B_T = sigma_B_T.squeeze([1, 3, 4])
        else:
            assert sigma_B_T.ndim == 2, f"not supported sigma shape {sigma_B_T.shape}"
        B, T = sigma_B_T.shape
        sigma_np = sigma_B_T[batch_dim].detach().float().detach().cpu().numpy()

        plt.figure(figsize=(12, 6))
        bars = plt.bar(np.arange(T), sigma_np, alpha=0.7, color='steelblue')

        # 在每个柱子上方添加数值标注
        for i, (bar, value) in enumerate(zip(bars, sigma_np)):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * sigma_np.max(),
                     f'{value:.4f}', ha='center', va='bottom', fontsize=8, rotation=45)

        plt.xlabel("Timestep")
        plt.ylabel("Sigma")
        plt.ylim(0, 1)
        plt.title(f"Sigma Values - {save_name}, shape={sigma_B_T.shape}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_name, dpi=150, bbox_inches='tight')
        plt.close()
        print("[DEBUG] vis_sigma: saved to", save_name)

    def denoise(
            self,
            xt_B_C_T_H_W: torch.Tensor,
            sigma: torch.Tensor,
            condition: ActionCondition,
            use_cuda_graphs: bool = False,
            # Action related
            at_B_T_D: torch.Tensor = None,
            a_sigma: torch.Tensor = None,
    ) -> DenoisePredictionWithAction:
        """
        Performs denoising on the input noise data, noise level, and condition
        Called when doing training (model.compute_loss_with_epsilon_and_sigma)
            or inference (self.get_x0_fn_from_batch).

        Args:
            xt (torch.Tensor): The input noise data.
            sigma (torch.Tensor): The noise level.
            condition (TextCondition): conditional information, generated from self.conditioner
            use_cuda_graphs (bool, optional): Whether to use CUDA Graphs for inference. Defaults to False.
            at_B_T_D (torch.Tensor, optional): Action tensor of B-T values. Defaults to None.
            a_sigma (torch.Tensor, optional): Action noise level. Defaults to None.

        Returns:
            DenoisePredictionWithAction: The denoised prediction, it includes clean data predicton (x0), \
                noise prediction (eps_pred), and clean action prediction (action0).
        """

        if sigma.ndim == 1:
            sigma_B_T = rearrange(sigma, "b -> b 1")
        elif sigma.ndim == 2:
            sigma_B_T = sigma
        else:
            raise ValueError(f"sigma shape {sigma.shape} is not supported")

        if a_sigma.ndim == 1:
            a_sigma_B_T = rearrange(a_sigma, "b -> b 1")
        elif a_sigma.ndim == 2:
            a_sigma_B_T = a_sigma
        else:
            raise ValueError(f"a_sigma shape {a_sigma.shape} is not supported")

        # print("[DEBUG] denoise: xt_B_C_T_H_W.shape", xt_B_C_T_H_W.shape, "sigma_B_T.shape", sigma_B_T.shape,
        #       "at_B_T_D.shape", at_B_T_D.shape if at_B_T_D is not None else None,
        #       "a_sigma_B_T.shape", a_sigma_B_T.shape)
        '''
        training:
        xt_B_C_T_H_W.shape torch.Size([12, 16, 2, 32, 32]) 
        sigma_B_T.shape torch.Size([12, 1]) 
        at_B_T_D.shape torch.Size([12, 12, 2]) 
        a_sigma_B_T.shape torch.Size([12, 1])
        inference:
        
        '''

        sigma_B_1_T_1_1 = rearrange(sigma_B_T, "b t -> b 1 t 1 1")  # for video
        sigma_B_T_1= rearrange(a_sigma_B_T, "b t -> b t 1")  # for action
        # get precondition for the network. self.scaling: RectifiedFlowScaling.
        c_skip_B_1_T_1_1, c_out_B_1_T_1_1, c_in_B_1_T_1_1, c_noise_B_1_T_1_1 = self.scaling(sigma=sigma_B_1_T_1_1)
        c_skip_B_T_1, c_out_B_T_1, c_in_B_T_1, c_noise_B_T_1 = self.scaling(sigma=sigma_B_T_1)

        net_state_in_B_C_T_H_W = xt_B_C_T_H_W * c_in_B_1_T_1_1
        a_net_state_in_B_T_D = at_B_T_D * c_in_B_T_1

        ''' Process condition: replace '''
        if condition.is_video:
            # Get the conditional frames and actions from condition.gt_frames (latent) and condition.gt_actions (raw)
            condition_state_in_B_C_T_H_W = condition.gt_frames.type_as(net_state_in_B_C_T_H_W) / self.config.sigma_data
            a_condition_state_in_B_T_D = condition.gt_actions.type_as(a_net_state_in_B_T_D) / self.config.sigma_data
            if not condition.use_video_condition:
                # When using random dropout, we zero out the ground truth frames
                condition_state_in_B_C_T_H_W = condition_state_in_B_C_T_H_W * 0
                a_condition_state_in_B_T_D = a_condition_state_in_B_T_D * 0  # currently, action cond flag is consistent with video

            _, C, _, _, _ = xt_B_C_T_H_W.shape
            _, _, Ca = at_B_T_D.shape
            condition_video_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(1, C, 1, 1, 1).type_as(
                net_state_in_B_C_T_H_W
            )
            condition_action_mask = condition.condition_action_input_mask_B_T_D.repeat(1, 1, Ca).type_as(
                a_net_state_in_B_T_D
            )

            # print("[DEBUG] denoise.mask. condition_video_input_mask_B_C_T_H_W:", condition.condition_video_input_mask_B_C_T_H_W,
            #       "condition_action_input_mask_B_T_D:", condition.condition_action_input_mask_B_T_D)

            if self.config.conditioning_strategy == str(ConditioningStrategy.FRAME_REPLACE):
                ## NOTE: go here
                # In case of frame replacement strategy, replace the first few frames of the video with the conditional frames
                # [1] Make the first few frames of x_t be the ground truth frames
                # self.vis_latent_state(net_state_in_B_C_T_H_W, save_name="output/tmp_net_state_in.mp4", fps=1)
                # self.vis_action(a_net_state_in_B_T_D, save_name="output/tmp_a_net_state_in.png")
                net_state_in_B_C_T_H_W = (
                    condition_state_in_B_C_T_H_W * condition_video_mask
                    + net_state_in_B_C_T_H_W * (1 - condition_video_mask)
                )
                # [2] Make the firs few actions of a_t be the ground truth actions (actions will be stepped in env)
                a_net_state_in_B_T_D = (
                    a_condition_state_in_B_T_D * condition_action_mask
                    + a_net_state_in_B_T_D * (1 - condition_action_mask)
                )
                # self.vis_latent_state(condition_state_in_B_C_T_H_W, save_name="output/tmp_cond_state_in.mp4", fps=1)
                # self.vis_latent_state(net_state_in_B_C_T_H_W, save_name="output/tmp_net_state_in_blended.mp4", fps=1)
                # self.vis_action(a_condition_state_in_B_T_D, save_name="output/tmp_a_cond_state_in.png")
                # self.vis_action(a_net_state_in_B_T_D, save_name="output/tmp_a_net_state_in_blended.png")

                # Update the c_noise as the conditional frames are clean and have very low noise
                # Adjust c_noise for the conditional frames
                # [1] Video
                sigma_cond_B_1_T_1_1 = torch.ones_like(sigma_B_1_T_1_1) * self.config.sigma_conditional  # very weak
                _, _, _, c_noise_cond_B_1_T_1_1 = self.scaling(sigma=sigma_cond_B_1_T_1_1)
                condition_video_mask_B_1_T_1_1 = condition_video_mask.mean(dim=[1, 3, 4], keepdim=True)
                # self.vis_sigma(c_noise_B_1_T_1_1, save_name="output/tmp_video_sigma.png")
                c_noise_B_1_T_1_1 = c_noise_cond_B_1_T_1_1 * condition_video_mask_B_1_T_1_1 + c_noise_B_1_T_1_1 * (
                        1 - condition_video_mask_B_1_T_1_1
                )
                # self.vis_sigma(c_noise_cond_B_1_T_1_1, save_name="output/tmp_video_sigma_cond.png")
                # self.vis_sigma(c_noise_B_1_T_1_1, save_name="output/tmp_video_sigma_blended.png")
                # [2] Action. Share the same condition sigma as video, a low constant value (i.e. 0.0001)
                sigma_cond_B_T_1 = torch.ones_like(sigma_B_T_1) * self.config.sigma_conditional  # same as video
                _, _, _, c_noise_cond_B_T_1 = self.scaling(sigma=sigma_cond_B_T_1)  # T != sigma_cond_B_1_T_1_1.T
                condition_action_mask_B_T_1 = condition_action_mask.mean(dim=[2], keepdim=True)
                # self.vis_sigma(c_noise_B_T_1, save_name="output/tmp_action_sigma.png")
                c_noise_B_T_1 = c_noise_cond_B_T_1 * condition_action_mask_B_T_1 + c_noise_B_T_1 * (
                        1 - condition_action_mask_B_T_1
                )
                # self.vis_sigma(c_noise_cond_B_T_1, save_name="output/tmp_action_sigma_cond.png")
                # self.vis_sigma(c_noise_B_T_1, save_name="output/tmp_action_sigma_blended.png")
                # print("[DEBUG] denoise.mask:", "video", condition_video_mask_B_1_T_1_1.shape, condition_video_mask_B_1_T_1_1[0].sum(),
                #       "action", condition_action_mask_B_T_1.shape, condition_action_mask_B_T_1[0].sum())
                '''
                video torch.Size([12, 1, 2, 1, 1]) tensor(2., device='cuda:0') 
                action torch.Size([12, 12, 1]) tensor(0., device='cuda:0')
                '''

            elif self.config.conditioning_strategy == str(ConditioningStrategy.CHANNEL_CONCAT):
                raise NotImplementedError("Action cond + channel concat not implemented yet.")
                # In case of channel concatenation strategy, concatenate the conditional frames in the channel dimension
                condition_state_in_masked_B_C_T_H_W = condition_state_in_B_C_T_H_W * condition_video_mask
                net_state_in_B_C_T_H_W = torch.cat([net_state_in_B_C_T_H_W, condition_state_in_masked_B_C_T_H_W], dim=1)

        else:
            # In case of image batch, simply concatenate the 0 frames when channel concat strategy is used
            raise NotImplementedError("Conditioning strategy not implemented for no-video samples.")
            if self.config.conditioning_strategy == str(ConditioningStrategy.CHANNEL_CONCAT):
                net_state_in_B_C_T_H_W = torch.cat(
                    [net_state_in_B_C_T_H_W, torch.zeros_like(net_state_in_B_C_T_H_W)], dim=1
                )

        # forward pass through the network
        net_output_B_C_T_H_W, action_B_Horizon_Dof = self.dit(
            x_B_C_T_H_W=net_state_in_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=c_noise_B_1_T_1_1.squeeze(dim=[1, 3, 4]).to(**self.tensor_kwargs),
            action_B_T_D=a_net_state_in_B_T_D.to(**self.tensor_kwargs) if a_net_state_in_B_T_D is not None else None,
            action_timesteps_B_T=c_noise_B_T_1.squeeze(dim=2).to(**self.tensor_kwargs) if a_net_state_in_B_T_D is not None else None,
            **condition.to_dict(),  # Not used: gt_frames, gt_actions
            use_cuda_graphs=use_cuda_graphs,
        )
        net_output_B_C_T_H_W = net_output_B_C_T_H_W.float()
        action_B_Horizon_Dof = action_B_Horizon_Dof.float()

        x0_pred_B_C_T_H_W = c_skip_B_1_T_1_1 * xt_B_C_T_H_W + c_out_B_1_T_1_1 * net_output_B_C_T_H_W
        a0_pred_B_T_D = c_skip_B_T_1 * at_B_T_D + c_out_B_T_1 * action_B_Horizon_Dof
        if condition.is_video:
            # Set the first few frames to the ground truth frames. This will ensure that the loss is not computed for the first few frames.
            # self.vis_latent_state(x0_pred_B_C_T_H_W, save_name="output/tmp_v0_pred.mp4", fps=1)
            # self.vis_action(a0_pred_B_T_D, save_name="output/tmp_a0_pred.png")
            x0_pred_B_C_T_H_W = condition.gt_frames.type_as(
                x0_pred_B_C_T_H_W
            ) * condition_video_mask + x0_pred_B_C_T_H_W * (1 - condition_video_mask)
            a0_pred_B_T_D = condition.gt_actions.type_as(
                a0_pred_B_T_D
            ) * condition_action_mask + a0_pred_B_T_D * (1 - condition_action_mask)
            # self.vis_latent_state(condition.gt_frames, save_name="output/tmp_v_cond.mp4", fps=1)
            # self.vis_latent_state(x0_pred_B_C_T_H_W, save_name="output/tmp_v0_pred_blended.mp4", fps=1)
            # self.vis_action(condition.gt_actions, save_name="output/tmp_a_cond.png")
            # self.vis_action(a0_pred_B_T_D, save_name="output/tmp_a0_pred_blended.png")

        # get noise prediction
        eps_pred_B_C_T_H_W = (xt_B_C_T_H_W - x0_pred_B_C_T_H_W) / sigma_B_1_T_1_1
        action_eps_pred_B_T_D = (at_B_T_D - a0_pred_B_T_D) / sigma_B_T_1

        return DenoisePredictionWithAction(
            x0_pred_B_C_T_H_W, eps_pred_B_C_T_H_W,
            None,
            action0=a0_pred_B_T_D, action_eps=action_eps_pred_B_T_D)

    def _randomly_sample_input_output_inplace(self, data_batch: dict) -> dict:
        """ Randomly sample the input and output frames for training
        Inputs:
        video:      (B, 3,  V*(max_obs+max_act_out),    H,  W)
        action:     (B, max_obs+max_act_out,    D)
        agent_pos:  (B, max_obs+max_act_out,    7+1)
        Outputs:
        video:      (B, 3,  V*(v_cond+v_out),   H,      W)
        action:     (B, a_out,  D)
        agent_pos:  (B, v_cond, 7+1)
        """
        # Randomly set num_cond_frames
        if np.random.uniform(0., 1.) <= self.p_all_actions_as_condition:
            # Input: v_cond, a_cond
            # Output: v_out (next frames)
            n_v_cond = self.max_obs
            n_a_cond = self.max_act_out  # use all frames as condition
            n_v_out = n_a_cond
            n_a_out = 0
        else:
            # Input: v_cond
            # Output: a_out
            n_v_cond = self.max_obs
            n_a_cond = 0  # a1=v1-1
            n_v_out = 0
            n_a_out = self.max_act_out

        # Process multi-view data
        one_view_video_len = self.max_obs + self.max_act_out
        all_view_video_len = data_batch[self.input_video_key].shape[2]
        assert all_view_video_len % one_view_video_len == 0, \
            f"video length {all_view_video_len} is not divisible by one_view_video_len {one_view_video_len}"
        n_views = all_view_video_len // one_view_video_len

        # Cut out the input and output frames
        ret_video = data_batch[self.input_video_key][:, :, :all_view_video_len]  # (B,3,V*(v_cond+v_out),256,256)
        ret_action = data_batch[self.input_action_key][:, n_v_cond - 1: n_v_cond -1 + self.max_act_out]  # (B,a_out,2)
        ret_agent_pos = data_batch[self.input_agent_pos_key][:, :n_v_cond]  # (B,v_cond,2)

        n_latent_v_cond = (n_v_cond - 1) // 4 + 1  # 4 is the default latent frame downsample ratio
        # print("[DEBUG] _randomly_sample_input_output_inplace. n_v_cond:", n_v_cond, "n_a_cond:", n_a_cond,
        #       "n_v_out:", n_v_out, "n_a_out:", n_a_out, "n_latent_v_cond:", n_latent_v_cond,
        #       f"data_batch[self.input_video_key]:", data_batch[self.input_video_key].shape)

        B, _, T, H, W = ret_video.shape
        data_batch[self.input_video_key] = ret_video
        data_batch[self.input_action_key] = ret_action
        data_batch[self.input_agent_pos_key] = ret_agent_pos
        data_batch[NUM_CONDITIONAL_FRAMES_KEY] = torch.ones(B, device=ret_video.device, dtype=torch.int32) * n_latent_v_cond
        data_batch[NUM_CONDITIONAL_ACTIONS_KEY] = torch.ones(B, device=ret_action.device, dtype=torch.int32) * n_a_cond

    def _normalize_video_databatch_inplace(self, data_batch: dict[str, torch.Tensor], input_key: str = None) -> None:
        """ Copied from: pipelines.multiview2video.py MultiView2VideoPipeline """
        input_key = self.input_video_key if input_key is None else input_key
        if input_key in data_batch:
            num_video_frames_per_view = self.tokenizer.get_pixel_num_frames(self.config.state_t)  # 33
            n_views = data_batch[input_key].shape[2] // num_video_frames_per_view  # 2
            data_batch[input_key] = rearrange(data_batch[input_key], "B C (V T) H W -> (B V) C T H W", V=n_views)
            super()._normalize_video_databatch_inplace(data_batch, input_key)
            data_batch[input_key] = rearrange(data_batch[input_key], "(B V) C T H W -> B C (V T) H W", V=n_views)

    def get_data_and_condition(
            self, data_batch: dict[str, torch.Tensor], needs_sampling_input_output: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, TextCondition]:
        """ Called when training (model.training_step) and inference (self.get_x0_fn_from_batch) """
        self._normalize_video_databatch_inplace(data_batch)  # sample a sub-sequence of video
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)
        # from debug.printer import print_batch
        # print_batch("[DEBUG] get_data_and_condition input data_batch", data_batch)

        # Randomly sample the input and output frames for training
        if needs_sampling_input_output:
            self._randomly_sample_input_output_inplace(data_batch)

        # Latent state
        raw_state = data_batch[self.input_image_key if is_image_batch else self.input_video_key]
        # raw_state = raw_state[:, :, :-1, :, :]  # (B,C,T,H,W), remove the last padding frame
        # print("[DEBUG] get_data_and_condition raw", raw_state.shape, raw_state.dtype, raw_state.min(), raw_state.max())
        # save_image_or_video(raw_state[0].float().detach().cpu() * 0.5 + 0.5, "output/tmp_pipeline_raw_state.mp4", fps=5)
        latent_state = self.encode(raw_state).contiguous().float()
        # print("[DEBUG] get_data_and_condition. latent", latent_state.shape, "raw_state", raw_state.shape)

        # Condition
        self.conditioner: ActionConditioner
        condition: ActionCondition = self.conditioner(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)

        num_conditional_frames = data_batch.get(NUM_CONDITIONAL_FRAMES_KEY, None)
        num_conditional_actions = data_batch.get(NUM_CONDITIONAL_ACTIONS_KEY, None)
        assert num_conditional_frames is not None and num_conditional_actions is not None, \
            "Expert model needs num_conditional_frames and num_conditional_actions in data_batch"

        condition = condition.set_video_condition(
            gt_frames=latent_state.to(**self.tensor_kwargs),
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            gt_actions=data_batch['action'],
            num_conditional_actions=num_conditional_actions,
            agent_pos=data_batch['agent_pos'],
            state_t=self.config.state_t,  # required by multi-view
        )
        return raw_state, latent_state, condition

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        n_views = state.shape[2] // self.tokenizer.get_pixel_num_frames(self.config.state_t)
        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if n_views > 4 and cp_size > 1 and n_views <= cp_size:
            return self.encode_cp(state)
        state = rearrange(state, "B C (V T) H W -> (B V) C T H W", V=n_views)
        encoded_state = super().encode(state)
        encoded_state = rearrange(encoded_state, "(B V) C T H W -> B C (V T) H W", V=n_views)
        return encoded_state

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        n_views = latent.shape[2] // self.config.state_t
        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if n_views > 4 and cp_size > 1 and n_views <= cp_size:
            return self.decode_cp(latent)
        latent = rearrange(latent, "B C (V T) H W -> (B V) C T H W", V=n_views)
        decoded_state = super().decode(latent)
        decoded_state = rearrange(decoded_state, "(B V) C T H W -> B C (V T) H W", V=n_views)
        return decoded_state

    @torch.no_grad()
    def encode_cp(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("encode_cp not tested yet")
        cp_size = len(get_process_group_ranks(parallel_state.get_context_parallel_group()))
        cp_group = parallel_state.get_context_parallel_group()
        n_views = state.shape[2] // self.tokenizer.get_pixel_num_frames(self.config.state_t)
        assert n_views < cp_size, f"n_views must be less than cp_size, got n_views={n_views} and cp_size={cp_size}"
        state_V_B_C_T_H_W = rearrange(state, "B C (V T) H W -> V B C T H W", V=n_views)
        state_input = torch.zeros((cp_size, *state_V_B_C_T_H_W.shape[1:]), **self.tensor_kwargs)
        state_input[0:n_views] = state_V_B_C_T_H_W
        local_state_V_B_C_T_H_W = broadcast_split_tensor(state_input, seq_dim=0, process_group=cp_group)
        local_state = rearrange(local_state_V_B_C_T_H_W, "V B C T H W -> (B V) C T H W")
        encoded_state = super().encode(local_state)
        encoded_state_list = [torch.empty_like(encoded_state) for _ in range(cp_size)]
        dist.all_gather(encoded_state_list, encoded_state, group=cp_group)
        encoded_state = torch.cat(encoded_state_list[0:n_views], dim=2)  # [B, C, V * T, H, W]
        return encoded_state

    @torch.no_grad()
    def decode_cp(self, latent: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("decode_cp not tested yet")
        cp_size = len(get_process_group_ranks(parallel_state.get_context_parallel_group()))
        cp_group = parallel_state.get_context_parallel_group()
        log.info(f"latent.shape: {latent.shape}")
        log.info(f"self.config.state_t: {self.config.state_t}")
        n_views = latent.shape[2] // self.config.state_t
        assert n_views < cp_size, f"n_views must be less than cp_size, got n_views={n_views} and cp_size={cp_size}"
        latent_V_B_C_T_H_W = rearrange(latent, "B C (V T) H W -> V B C T H W", V=n_views)
        latent_input = torch.zeros((cp_size, *latent_V_B_C_T_H_W.shape[1:]), **self.tensor_kwargs)
        latent_input[0:n_views] = latent_V_B_C_T_H_W
        local_latent_V_B_C_T_H_W = broadcast_split_tensor(latent_input, seq_dim=0, process_group=cp_group)
        local_latent = rearrange(local_latent_V_B_C_T_H_W, "V B C T H W -> (B V) C T H W")
        decoded_state = super().decode(local_latent)
        decoded_state_list = [torch.empty_like(decoded_state) for _ in range(cp_size)]
        dist.all_gather(decoded_state_list, decoded_state, group=cp_group)
        decoded_state = torch.cat(decoded_state_list[0:n_views], dim=2)  # [B, C, V * T, H, W]
        return decoded_state

    def broadcast_split_for_model_parallelsim(
        self,
        x0_B_C_T_H_W: torch.Tensor,
        condition: torch.Tensor,
        epsilon_B_C_T_H_W: torch.Tensor,
        sigma_B_T: torch.Tensor,
    ):
        """ Copied from: pipelines.multiview2video.py MultiView2VideoPipeline """
        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        n_views = x0_B_C_T_H_W.shape[2] // self.config.state_t
        if cp_size > 1 and n_views > 1:
            x0_B_C_T_H_W = rearrange(x0_B_C_T_H_W, "B C (V T) H W -> (B V) C T H W", V=n_views).contiguous()
            if epsilon_B_C_T_H_W is not None:
                epsilon_B_C_T_H_W = rearrange(epsilon_B_C_T_H_W, "B C (V T) H W -> (B V) C T H W", V=n_views).contiguous()
            reshape_sigma_B_T = False
            if sigma_B_T is not None:
                assert sigma_B_T.ndim == 2, "sigma_B_T should be 2D tensor"
                if sigma_B_T.shape[-1] != 1:
                    assert (
                        sigma_B_T.shape[-1] % n_views == 0
                    ), f"sigma_B_T temporal dimension T must either be 1 or a multiple of sample_n_views. Got T={sigma_B_T.shape[-1]} and sample_n_views={n_views}"
                    sigma_B_T = rearrange(sigma_B_T, "B (V T) -> (B V) T", V=n_views).contiguous()
                    reshape_sigma_B_T = True
            x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T = super().broadcast_split_for_model_parallelsim(
                x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T
            )
            x0_B_C_T_H_W = rearrange(x0_B_C_T_H_W, "(B V) C T H W -> B C (V T) H W", V=n_views)
            if epsilon_B_C_T_H_W is not None:
                epsilon_B_C_T_H_W = rearrange(epsilon_B_C_T_H_W, "(B V) C T H W -> B C (V T) H W", V=n_views)
            if reshape_sigma_B_T:
                sigma_B_T = rearrange(sigma_B_T, "(B V) T -> B (V T)", V=n_views)
        return x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T

    def get_x0_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
        use_cuda_graphs: bool = False,
    ) -> Callable:
        """
        Called during inference to get the x0 prediction function.
        Generates a callable function `x0_fn` based on the provided data batch and guidance factor.

        This function first processes the input data batch through a conditioning workflow (`conditioner`) to obtain conditioned and unconditioned states. It then defines a nested function `x0_fn` which applies a denoising operation on an input `noise_x` at a given noise level `sigma` using both the conditioned and unconditioned states.

        Args:
        - data_batch (Dict): A batch of data used for conditioning. The format and content of this dictionary should align with the expectations of the `self.conditioner`
        - guidance (float, optional): A scalar value that modulates the influence of the conditioned state relative to the unconditioned state in the output. Defaults to 1.5.
        - is_negative_prompt (bool): use negative prompt t5 in uncondition if true

        Returns:
        - Callable: A function `x0_fn(noise_x, sigma)` that takes two arguments, `noise_x` and `sigma`, and return x0 predictoin

        The returned function is suitable for use in scenarios where a denoised state is required based on both conditioned and unconditioned inputs, with an adjustable level of guidance influence.
        """
        if NUM_CONDITIONAL_FRAMES_KEY in data_batch:
            num_conditional_frames = data_batch[NUM_CONDITIONAL_FRAMES_KEY]
        else:
            raise "Please provide num_conditional_frames in data_batch during inference"

        if NUM_CONDITIONAL_ACTIONS_KEY in data_batch:
            num_conditional_actions = data_batch[NUM_CONDITIONAL_ACTIONS_KEY]
        else:
            raise "Please provide num_conditional_actions in data_batch during inference"

        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        is_image_batch = self.is_image_batch(data_batch)
        condition: ActionCondition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition: ActionCondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        _, x0, _ = self.get_data_and_condition(data_batch,
                                               needs_sampling_input_output=False)
        # override condition with inference mode; num_conditional_frames used Here!
        condition = condition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            gt_actions=data_batch['action'],
            num_conditional_actions=num_conditional_actions,
            agent_pos=data_batch['agent_pos'],
            state_t=self.config.state_t,  # required by multi-view
        )
        uncondition = uncondition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            gt_actions=data_batch['action'],
            num_conditional_actions=num_conditional_actions,
            agent_pos=data_batch['agent_pos'],
            state_t=self.config.state_t,  # required by multi-view
        )
        # print("[DEUBG] input condition:", condition.gt_frames.shape, condition.gt_actions.shape)
        print("[DEBUG] num_conditional_frames:", num_conditional_frames, "num_conditional_actions:", num_conditional_actions)
        condition = condition.edit_for_inference(
            is_cfg_conditional=True, num_conditional_frames=num_conditional_frames,
            num_conditional_actions=num_conditional_actions,
        )
        uncondition = uncondition.edit_for_inference(
            is_cfg_conditional=False, num_conditional_frames=num_conditional_frames,
            num_conditional_actions=num_conditional_actions,
        )
        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(x0, condition, None, None)
        _, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(x0, uncondition, None, None)

        if not parallel_state.is_initialized():
            assert (
                not self.dit.is_context_parallel_enabled
            ), "parallel_state is not initialized, context parallel should be turned off."

        def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor,
                  noise_a: torch.Tensor = None, sigma_a: torch.Tensor = None,
                  ) -> tuple[torch.Tensor, torch.Tensor]:
            # refer to: Predict2Video2WorldExpertModel.compute_loss_with_epsilon_and_sigma
            cond_x0_a0 = self.denoise(noise_x, sigma, condition, use_cuda_graphs=use_cuda_graphs,
                at_B_T_D=noise_a,
                a_sigma=sigma_a,
                )
            cond_x0 = cond_x0_a0.x0
            cond_a0 = cond_x0_a0.action0

            uncond_x0_a0 = self.denoise(noise_x, sigma, uncondition, use_cuda_graphs=use_cuda_graphs,
                at_B_T_D=noise_a,
                a_sigma=sigma_a,
                )
            uncond_x0 = uncond_x0_a0.x0
            uncond_a0 = uncond_x0_a0.action0

            raw_x0 = cond_x0 + guidance * (cond_x0 - uncond_x0)
            raw_a0 = cond_a0 + guidance * (cond_a0 - uncond_a0)

            if "guided_image" in data_batch:  # not used
                raise NotImplementedError("guided_image not implemented yet")
                # replacement trick that enables inpainting with base model
                assert "guided_mask" in data_batch, "guided_mask should be in data_batch if guided_image is present"
                guide_image = data_batch["guided_image"]
                guide_mask = data_batch["guided_mask"]
                raw_x0 = guide_mask * guide_image + (1 - guide_mask) * raw_x0
            return raw_x0, raw_a0

        return x0_fn

    def _get_data_batch_input(
        self,
        video: torch.Tensor,
        actions: torch.Tensor,
        agent_pos: torch.Tensor,
        prompt: Union[str, torch.Tensor],
        negative_prompt: str = "",
        num_latent_conditional_frames: int = 1,
        num_conditional_actions: int = 0,
        n_views: int = 1,
    ):
        """
        Called during inference.
        Prepares the input data batch for the diffusion model.

        Constructs a dictionary containing the video tensor, text embeddings,
        and other necessary metadata required by the model's forward pass.
        Optionally includes negative text embeddings.

        Args:
            video (torch.Tensor): The input video tensor (B, C, T, H, W).
            prompt (str or torch.Tensor): The text prompt for conditioning or precomputed text embeddings (B,512,1024)
            negative_prompt (str): Negative prompt.
            num_latent_conditional_frames (int, optional): The number of latent conditional frames. Defaults to 1.

        Returns:
            dict: A dictionary containing the prepared data batch, moved to the correct device and dtype.
        """
        B, C, T, H, W = video.shape

        if isinstance(prompt, str) and prompt == "":
            t5_text_embeddings = torch.zeros(1, 512, 1024, dtype=torch.bfloat16).cuda()
        elif isinstance(prompt, str):
            t5_text_embeddings = self.encode_prompt(prompt).to(dtype=torch.bfloat16).cuda()
        elif isinstance(prompt, torch.Tensor):
            t5_text_embeddings = prompt.cuda()  # (1, 512, 1024)
        else:
            raise NotImplementedError("prompt must be a string or a tensor of shape (B, 512, 1024)")

        # Check n_views consistency
        assert T == n_views * (agent_pos.shape[1] + actions.shape[1]), \
            f"Video frames T ({T}) causes conflict, are you sure n_views ({n_views}) is correct? )"
        # if "LIVING ROOM SCENE2" in prompt:
        #     prompt = prompt.replace("LIVING ROOM SCENE2", "")

        print(f"[DEBUG] prompt ({prompt if isinstance(prompt, str) else prompt.shape}) "
              f"t5_text_embeddings:", t5_text_embeddings.shape, t5_text_embeddings.dtype,
              t5_text_embeddings.min(), t5_text_embeddings.max())

        latent_view_indices_T = torch.repeat_interleave(torch.arange(n_views), self.config.state_t)
        latent_view_indices_B_T = latent_view_indices_T.unsqueeze(0).expand(B, -1).to(self.device)

        self.batch_size = B  # ori:1
        data_batch = {
            "dataset_name": "video_data",
            "video": video,
            # NOTE: we don't use text embeddings for action conditional video2world
            "t5_text_embeddings": t5_text_embeddings.repeat(self.batch_size, 1, 1),
            "fps": torch.ones(self.batch_size) * 20,  # ori:torch.randint(16, 32, (self.batch_size,)),  # Random FPS (might be used by model)
            "padding_mask": torch.zeros(self.batch_size, 1, H, W),  # Padding mask (assumed no padding here)
            "num_conditional_frames": num_latent_conditional_frames,  # ori:num_latent_conditional_frames,  # Specify number of conditional frames
            "num_conditional_actions": num_conditional_actions,
            "action": actions,
            "agent_pos": agent_pos,  # (T,7+1), [-1,1]
            # Multi-view related
            "sample_n_views": n_views,
            "latent_view_indices_B_T": latent_view_indices_B_T,  # in dataset is (T,), here is (B,T) like DataLoader
        }

        # Handle negative prompts for classifier-free guidance
        if negative_prompt:
            data_batch["neg_t5_text_embeddings"] = self.encode_prompt(negative_prompt).to(dtype=self.torch_dtype)

        # Move tensors to GPU and convert to bfloat16 if they are floating point
        for k, v in data_batch.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(data_batch[k]):
                data_batch[k] = v.cuda().to(dtype=torch.bfloat16)

        return data_batch

    @torch.no_grad()
    def __call__(
        self,
        first_frame: np.ndarray,  # (V*(v1+v2),H,W,C), in [0,255], uint8
        actions: np.ndarray,  # (a1+a2,D), in [-1,1], float32
        agent_pos: np.ndarray,  # (v1+v2,7+1), in [-1,1], float32
        prompt: Union[str, torch.Tensor] = "",  # text prompt or text embeddings
        negative_prompt: str = "",
        num_conditional_frames: int = 5,
        num_conditional_actions: int = 0,
        guidance: float = 7.0,
        num_sampling_step: int = 35,
        seed: int = 0,
        solver_option: str = "2ab",
        n_views: int = 1,  # Multi-view related
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Parameter check
        # width, height = VIDEO_RES_SIZE_INFO[self.config.resolution]["16:9"]  # type: ignore
        # height, width = self.check_resize_height_width(height, width)
        assert num_conditional_frames % 4 == 1, "num_conditional_frames-1 must be divisible by 4"
        num_latent_conditional_frames = self.tokenizer.get_latent_num_frames(num_conditional_frames)

        print("[DEBUG] pipeline.__call__ first_frame", first_frame.shape, "actions", actions.shape,)

        # num_video_frames = self.tokenizer.get_pixel_num_frames(self.config.state_t)

        # transform first frame and actions to tensor
        if first_frame.ndim == 4:
            # vid_input = torch.from_numpy(first_frame).permute(2, 0, 1)[None, :, None, ...]
            vid_input = torch.from_numpy(first_frame).permute(3, 0, 1, 2)  # (VT,H,W,C) -> (C,VT,H,W)
            # Cv, Tv, H, W = vid_input.shape
            # vid_back_padding = torch.zeros((Cv, 1, H, W), dtype=vid_input.dtype)  # for padding to v1+v2+1 frames
            # vid_input = torch.cat((vid_input, vid_back_padding), dim=1)   # (C, v1+v2+1, H, W)
            vid_input = vid_input[None, ...]  # Add batch dimension (1,C,v1+v2+1,H,W)
            # print("first_frame", first_frame.shape, "vid_input", vid_input.shape)
            actions_tensor = torch.from_numpy(actions).to(dtype=torch.bfloat16)[None, ...]  # (1,a1+a2,D)
            agent_pos_tensor = torch.from_numpy(agent_pos).to(dtype=torch.bfloat16)[None, ...]  # (1,v1+v2,7+1)
        else:
            assert first_frame.ndim == 5, "first_frame must be 4 or 5 dims"
            assert actions.ndim == 3, "actions must be 3 dims"
            vid_input = torch.from_numpy(first_frame).permute(0, 4, 1, 2, 3)   # (B,C,T,H,W)
            actions_tensor = torch.from_numpy(actions).to(dtype=torch.bfloat16)  # (B,a1+a2,D)
            assert vid_input.shape[0] == actions_tensor.shape[0], "first_frame and actions must have the same batch size"
            agent_pos_tensor = torch.from_numpy(agent_pos).to(dtype=torch.bfloat16)  # (B,v1+v2,7+1)

        # Check prompt type, if it is a tensor, it must be of shape (512, 1024)
        if isinstance(prompt, torch.Tensor):
            assert prompt.ndim == 2 and prompt.shape[0] == 512 and prompt.shape[1] == 1024, \
                "If prompt is a tensor, it must be of shape (B, 512, 1024)"
            assert prompt.dtype == torch.bfloat16, "If prompt is a tensor, it must be of dtype torch.bfloat16"
            prompt = prompt.unsqueeze(0)  # (1, 512, 1024)
        elif isinstance(prompt, str):
            pass

        # Prepare the data batch with text embeddings
        data_batch = self._get_data_batch_input(
            vid_input,
            actions_tensor,
            agent_pos_tensor,
            prompt,
            negative_prompt,
            num_latent_conditional_frames=num_latent_conditional_frames,
            num_conditional_actions=num_conditional_actions,
            n_views=n_views,
        )
        from debug.printer import print_batch
        print_batch('[Video2WorldExpertPipeline] data_batch', data_batch)
        '''
        [Video2WorldExpertPipeline] data_batch: Dict,keys=dict_keys(['dataset_name', 'video', 't5_text_embeddings', 'fps', 'padding_mask', 'num_conditional_frames', 'action'])
        dataset_name:<class 'str'>,len=10
        video,<class 'torch.Tensor'>,shape=torch.Size([1, 3, 13, 256, 256])
        t5_text_embeddings,<class 'torch.Tensor'>,shape=torch.Size([1, 512, 1024])
        fps,<class 'torch.Tensor'>,shape=torch.Size([1])
        padding_mask,<class 'torch.Tensor'>,shape=torch.Size([1, 1, 256, 256])
        num_conditional_frames:<class 'int'>,1
        action,<class 'torch.Tensor'>,shape=torch.Size([1, 24, 2])
        LIBERO:
        [Video2WorldExpertPipeline] data_batch: Dict,keys=dict_keys(['dataset_name', 'video', 't5_text_embeddings', 'fps', 'padding_mask', 'num_conditional_frames', 'num_conditional_actions', 'action'])
        dataset_name:<class 'str'>,len=10
        video,<class 'torch.Tensor'>,shape=torch.Size([1, 3, 33, 128, 128]),min=0.0000,max=255.0000
        t5_text_embeddings,<class 'torch.Tensor'>,shape=torch.Size([1, 512, 1024]),min=-0.7461,max=0.6172
        fps,<class 'torch.Tensor'>,shape=torch.Size([1]),min=10.0000,max=10.0000
        padding_mask,<class 'torch.Tensor'>,shape=torch.Size([1, 1, 128, 128]),min=0.0000,max=0.0000
        num_conditional_frames:<class 'int'>,3
        num_conditional_actions:<class 'int'>,0
        action,<class 'torch.Tensor'>,shape=torch.Size([1, 24, 10]),min=0.0000,max=0.0000
        '''

        # preprocess
        self._normalize_video_databatch_inplace(data_batch)  # 1st normailization
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image_batch else self.input_video_key
        n_sample = data_batch[input_key].shape[0]
        _T, _H, _W = data_batch[input_key].shape[-3:]
        state_shape = [
            self.config.state_ch,
            self.config.state_t * n_views,  # consider multi-view
            _H // self.tokenizer.spatial_compression_factor,
            _W // self.tokenizer.spatial_compression_factor,
        ]
        _Ta, _Da = data_batch['action'].shape[1:]  # action shape: (B, T, DoF)
        state_a_shape = [
            _Ta,
            _Da,
        ]
        '''
        state_shape: [16, 4, 32, 32]
        state_a_shape: [24, 2]
        '''

        x0_fn = self.get_x0_fn_from_batch(data_batch, guidance, is_negative_prompt=True)  # will call self.denoise

        log.info("Starting video generation...")

        x_sigma_max = (
            misc.arch_invariant_rand(
                (n_sample,) + tuple(state_shape),
                torch.float32,
                self.tensor_kwargs["device"],
                seed,
            )
            * self.scheduler.config.sigma_max
        )  # (B, C, T, H, W)
        a_sigma_max = (
            misc.arch_invariant_rand(
                (n_sample,) + tuple(state_a_shape),
                torch.float32,
                self.tensor_kwargs["device"],
                seed,
            )
            * self.scheduler.config.sigma_max
        )  # (B, Ta, Da)

        # Split the input data and condition for model parallelism, if context parallelism is enabled.
        if self.dit.is_context_parallel_enabled:
            x_sigma_max = split_inputs_cp(x=x_sigma_max, seq_dim=2, cp_group=self.get_context_parallel_group())
            a_sigma_max = split_inputs_cp(x=a_sigma_max, seq_dim=1, cp_group=self.get_context_parallel_group())

        # ------------------------------------------------------------------ #
        # Sampling loop driven by `RectifiedFlowAB2Scheduler`
        # ------------------------------------------------------------------ #
        scheduler = self.scheduler

        # Construct sigma schedule (L + 1 entries including simga_min) and timesteps
        scheduler.set_timesteps(num_sampling_step, device=x_sigma_max.device)

        # Bring the initial latent into the precision expected by the scheduler
        sample = x_sigma_max.to(dtype=torch.float32)
        sample_a = a_sigma_max.to(dtype=torch.float32)

        x0_prev: torch.Tensor | None = None
        a0_prev: torch.Tensor | None = None

        for i, _ in enumerate(scheduler.timesteps):
            # Current noise level (sigma_t).
            sigma_t = scheduler.sigmas[i].to(sample.device, dtype=torch.float32)

            # `x0_fn` expects `sigma` as a tensor of shape [B] or [B, T]. We
            # pass a 1-D tensor broadcastable to any later shape handling.
            sigma_in = sigma_t.repeat(sample.shape[0])

            # x0 prediction with conditional and unconditional branches
            x0_pred, a0_pred = x0_fn(
                sample, sigma_in,
                noise_a=sample_a, sigma_a=sigma_in
            )

            # Scheduler step updates the noisy sample and returns the cached x0.
            # [1] Video
            sample, x0_prev = scheduler.step(
                x0_pred=x0_pred,
                i=i,
                sample=sample,
                x0_prev=x0_prev,
            )
            # [2] Action
            sample_a, a0_prev = scheduler.step(
                x0_pred=a0_pred,
                i=i,
                sample=sample_a,
                x0_prev=a0_prev,
            )

        # Final clean pass at sigma_min.
        sigma_min = scheduler.sigmas[-1].to(sample.device, dtype=torch.float32)
        sigma_in = sigma_min.repeat(sample.shape[0])
        samples, samples_a = x0_fn(
            sample, sigma_in,
            noise_a=sample_a, sigma_a=sigma_in
        )

        # Merge context-parallel chunks back together if needed.
        if self.dit.is_context_parallel_enabled:
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.get_context_parallel_group())
            samples_a = cat_outputs_cp(samples_a, seq_dim=1, cp_group=self.get_context_parallel_group())

        # Decode
        video = self.decode(samples)  # shape: (B, C, T, H, W), possibly out of [-1, 1]
        step_action = samples_a  # (B, Ta, Da), in [-1,1]
        print("[DEBUG] pipeline.__call__ out video", video.shape, video.min(), video.max(),
              "step_action", step_action.shape, step_action.min(), step_action.max(),)

        # Run video guardrail on the generated video and apply postprocessing
        if self.video_guardrail_runner is not None:
            # Clamp to safe range before normalization
            video = video.clamp(-1.0, 1.0)
            video_normalized = (video + 1) / 2  # [0, 1]

            # Convert tensor to NumPy frames for guardrail processing
            video_squeezed = video_normalized.squeeze(0)  # (C, T, H, W)
            frames = (video_squeezed * 255).clamp(0, 255).to(torch.uint8)
            frames = frames.permute(1, 2, 3, 0).cpu().numpy()  # (T, H, W, C)

            # Run guardrail
            processed_frames = guardrail_presets.run_video_guardrail(frames, self.video_guardrail_runner)
            if processed_frames is None:
                return None
            else:
                log.success("Passed guardrail on generated video")

            # Convert processed frames back to tensor format
            processed_video = torch.from_numpy(processed_frames).float().permute(3, 0, 1, 2) / 255.0
            processed_video = processed_video * 2 - 1  # back to [-1, 1]
            processed_video = processed_video.unsqueeze(0)

            video = processed_video.to(video.device, dtype=video.dtype)

        log.success("Video generation completed successfully")
        return video, step_action
