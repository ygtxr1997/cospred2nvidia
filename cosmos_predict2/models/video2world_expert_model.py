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

from typing import Tuple, Dict, Any
import math

import torch
import numpy as np
from einops import rearrange
from megatron.core import parallel_state
from torch.distributed.device_mesh import init_device_mesh

from cosmos_predict2.models.video2world_model import Predict2Video2WorldModel, Predict2Video2WorldModelConfig
# from cosmos_predict2.pipelines.video2world_action import Video2WorldActionConditionedPipeline
from cosmos_predict2.pipelines.video2world_expert import Video2WorldExpertPipeline, DenoisePredictionWithAction
from cosmos_predict2.conditioner import ActionCondition, DataType
from cosmos_predict2.utils.vis_helpers import save_action_as_image
from imaginaire.utils.io import save_image_or_video
from imaginaire.model import ImaginaireModel
from imaginaire.utils import log


class Predict2Video2WorldExpertModel(Predict2Video2WorldModel):
    def __init__(self, config: Predict2Video2WorldModelConfig):
        super(ImaginaireModel, self).__init__()

        self.config = config

        self.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.precision]
        self.tensor_kwargs = {"device": "cuda", "dtype": self.precision}
        self.device = torch.device("cuda")

        # 1. set data keys and data information
        self.setup_data_key()

        # 4. Set up loss options, including loss masking, loss reduce and loss scaling
        self.loss_reduce = getattr(config, "loss_reduce", "mean")
        assert self.loss_reduce in ["mean", "sum"]
        self.loss_scale = getattr(config, "loss_scale", 1.0)
        log.critical(f"Using {self.loss_reduce} loss reduce with loss scale {self.loss_scale}")
        if self.config.adjust_video_noise:
            self.video_noise_multiplier = math.sqrt(self.config.pipe_config.state_t)
        else:
            self.video_noise_multiplier = 1.0

        # 7. training states
        if parallel_state.is_initialized():
            self.data_parallel_size = parallel_state.get_data_parallel_world_size()
        else:
            self.data_parallel_size = 1

        # NOTE: replace the pipeline with expert pipeline
        self.pipe: Video2WorldExpertPipeline = Video2WorldExpertPipeline.from_config(
            config.pipe_config,
            dit_path=config.model_manager_config.dit_path,
        )

        self.freeze_parameters()
        if config.train_architecture == "lora":
            self.add_lora_to_model(
                self.pipe.dit,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_target_modules=config.lora_target_modules,
                init_lora_weights=config.init_lora_weights,
            )
            if self.pipe.dit_ema:
                self.add_lora_to_model(
                    self.pipe.dit_ema,
                    lora_rank=config.lora_rank,
                    lora_alpha=config.lora_alpha,
                    lora_target_modules=config.lora_target_modules,
                    init_lora_weights=config.init_lora_weights,
                )
        else:
            self.pipe.denoising_model().requires_grad_(True)
        total_params = sum(p.numel() for p in self.parameters())
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Print the number in billions, or in the format of 1,000,000,000
        log.info(
            f"Total parameters: {total_params / 1e9:.2f}B, Frozen parameters: {frozen_params:,}, Trainable parameters: {trainable_params:,}"
        )

        if config.fsdp_shard_size != 0 and torch.distributed.is_initialized():
            if config.fsdp_shard_size == -1:
                fsdp_shard_size = torch.distributed.get_world_size()
                replica_group_size = 1
            else:
                fsdp_shard_size = min(config.fsdp_shard_size, torch.distributed.get_world_size())
                replica_group_size = torch.distributed.get_world_size() // fsdp_shard_size
            dp_mesh = init_device_mesh(
                "cuda", (replica_group_size, fsdp_shard_size), mesh_dim_names=("replicate", "shard")
            )
            log.info(f"Using FSDP with shard size {fsdp_shard_size} | device mesh: {dp_mesh}")
            self.pipe.apply_fsdp(dp_mesh)
        else:
            log.info("FSDP (Fully Sharded Data Parallel) is disabled.")

    def compute_loss_with_epsilon_and_sigma(
        self,
        x0_B_C_T_H_W: torch.Tensor,
        condition: ActionCondition,
        epsilon_B_C_T_H_W: torch.Tensor,
        sigma_B_T: torch.Tensor,
        # Action related.
        action0_B_T_D: torch.Tensor = None,
        action_epsilon_B_T_D: torch.Tensor = None,
        action_sigma_B_T: torch.Tensor = None,
    ) -> Tuple[dict, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Compute loss givee epsilon and sigma

        This method is responsible for computing loss give epsilon and sigma. It involves:
        1. Adding noise to the input data.
        2. Passing the noisy data through the network to generate predictions.
        3. Computing the loss based on the difference between the predictions and the original data, \
            considering any configured loss weighting.

        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            x0: image/video latent
            condition: text condition
            epsilon: noise
            sigma: noise level

        Returns:
            tuple: A tuple containing four elements:
                - dict: additional data that used to debug / logging / callbacks
                - Tensor 1: kendall loss,
                - Tensor 2: MSE loss,
                - Tensor 3: EDM loss

        Raises:
            AssertionError: If the class is conditional, \
                but no number of classes is specified in the network configuration.

        Notes:
            - The method handles different types of conditioning
            - The method also supports Kendall's loss
        """

        # print("[DEBUG] compute_loss_with_epsilon_and_sigma "
        #       "\nx0_B_C_T_H_W.shape:", x0_B_C_T_H_W.shape,
        #       "\nsigma_B_T.shape:", sigma_B_T.shape,
        #       "\nepsilon_B_C_T_H_W.shape:", epsilon_B_C_T_H_W.shape,
        #       "\naction0_B_T_D.shape:", action0_B_T_D.shape,
        #       "\naction_epsilon_B_T_D.shape:", action_epsilon_B_T_D.shape,
        #       "\naction_sigma_B_T.shape:", action_sigma_B_T.shape)
        '''
        x0_B_C_T_H_W.shape: torch.Size([12, 16, 5, 32, 32])                                                                                                                                                                                                          
        sigma_B_T.shape: torch.Size([12, 1])                                                                                          
        epsilon_B_C_T_H_W.shape: torch.Size([12, 16, 5, 32, 32])                                                                                                                                                                                                     
        action0_B_T_D.shape: torch.Size([12, 12, 2]) 
        action_epsilon_B_T_D.shape: torch.Size([12, 12, 2]) 
        action_sigma_B_T.shape: torch.Size([12, 1])
        '''

        ## [1] Video branch
        # Get the mean and stand deviation of the marginal probability distribution.
        mean_B_C_T_H_W, std_B_T = x0_B_C_T_H_W, sigma_B_T
        # Generate noisy observations
        xt_B_C_T_H_W = mean_B_C_T_H_W + epsilon_B_C_T_H_W * rearrange(std_B_T, "b t -> b 1 t 1 1")
        ## [2] Action branch
        action_mean_B_T_D, action_std_B_T = action0_B_T_D, action_sigma_B_T  # Action condition shares the same sigma with Video.
        actiont_B_T_D = action_mean_B_T_D + action_epsilon_B_T_D * rearrange(action_std_B_T, "b t -> b t 1")

        # make prediction
        model_pred: DenoisePredictionWithAction = self.pipe.denoise(xt_B_C_T_H_W, sigma_B_T, condition,
                                                                    at_B_T_D=actiont_B_T_D,
                                                                    a_sigma=action_sigma_B_T,
                                                                    )
        # loss weights for different noise levels
        weights_per_sigma_B_T = self.get_per_sigma_loss_weights(sigma=sigma_B_T)

        # extra loss mask for each sample, for example, human faces, hands
        # [1] Video
        pred_mse_B_C_T_H_W = (x0_B_C_T_H_W - model_pred.x0) ** 2
        edm_loss_B_C_T_H_W = pred_mse_B_C_T_H_W * rearrange(weights_per_sigma_B_T, "b t -> b 1 t 1 1")
        # [2] Action
        action_pred_mse_B_T_D = (action0_B_T_D - model_pred.action0) ** 2
        action_edm_loss_B_T_D = action_pred_mse_B_T_D * rearrange(weights_per_sigma_B_T, "b t -> b t 1")

        # # DEBUG: visualize the action prediction
        # save_action_as_image(action0_B_T_D[0].detach().float().detach().cpu().numpy(), "output/tmp_action0.png")
        # save_action_as_image(model_pred.action0[0].detach().float().cpu().numpy(), "output/tmp_action0_pred.png")
        # vis_x0_in = self.pipe.decode(x0_B_C_T_H_W[:2])  # shape: (B, C, T, H, W), possibly out of [-1, 1]
        # vis_x0_pred = self.pipe.decode(model_pred.x0[:2])  # shape: (B, C, T, H, W), possibly out of [-1, 1]
        # save_image_or_video(vis_x0_in[0, :3].detach().cpu(), "output/tmp_x0.mp4", fps=5)
        # save_image_or_video(vis_x0_pred[0, :3].detach().cpu(), "output/tmp_x0_pred.mp4", fps=5)
        # exit()

        kendall_loss = edm_loss_B_C_T_H_W
        action_kendall_loss = action_edm_loss_B_T_D

        output_batch = {
            "x0": x0_B_C_T_H_W,
            "xt": xt_B_C_T_H_W,
            "a0": action0_B_T_D,  # action label
            "at": actiont_B_T_D,  # action input
            "sigma": sigma_B_T,
            "weights_per_sigma": weights_per_sigma_B_T,
            "condition": condition,
            "model_pred": model_pred,
            "mse_loss": pred_mse_B_C_T_H_W.mean(),
            "edm_loss": edm_loss_B_C_T_H_W.mean(),
            "edm_loss_per_frame": torch.mean(edm_loss_B_C_T_H_W, dim=[1, 3, 4]),
            "action_mse_loss": action_pred_mse_B_T_D.mean(),  # action loss
            "action_edm_loss": action_edm_loss_B_T_D.mean(),  # action loss
        }
        output_batch["loss"] = kendall_loss.mean() + action_kendall_loss.mean()  # check if this is what we want
        kendall_loss_dict = {
            "kendall_loss": kendall_loss,
            "action_kendall_loss": action_kendall_loss,
        }

        return output_batch, kendall_loss_dict, pred_mse_B_C_T_H_W, edm_loss_B_C_T_H_W

    def training_step(self, data_batch: dict, data_batch_idx: int) -> tuple[dict, torch.Tensor]:
        """ Training entry point """
        self.pipe.device = self.device

        # Loss
        self._update_train_stats(data_batch)

        # Get the input data to noise and denoise~(image, video) and the corresponding conditioner.
        condition: ActionCondition
        raw_B_C_T_H_W, x0_B_C_T_H_W, condition = self.pipe.get_data_and_condition(data_batch)  # x0 is in latent space
        action0_B_T_D = condition.gt_actions  # T=horizon, D=action_dof
        # print("[DEBUG] training_step:", "x0_B_C_T_H_W.shape", x0_B_C_T_H_W.shape,
        #       "action0_B_T_D.shape", action0_B_T_D.shape)
        '''
        x0_B_C_T_H_W.shape torch.Size([12, 16, 5*, 32, 32]) 
        action0_B_T_D.shape torch.Size([12, 12*, 2])
        '''
        # DEBUG. Visualize the input data and condition
        # print("[DEBUG] training_step:", "raw_state", raw_B_C_T_H_W[0].min(), raw_B_C_T_H_W[0].max(),)
        # save_image_or_video(raw_B_C_T_H_W[0].float().detach().cpu() * 0.5 + 0.5, "output/tmp_raw_state.mp4", fps=5)
        # save_action_as_image(action0_B_T_D[0].float().detach().cpu().numpy(), "output/tmp_cond_action0.png")

        # Sample pertubation noise levels and N(0, 1) noises
        # sigma_B_T: all frames share the same sigma, that is, T=1
        sigma_B_T, epsilon_B_C_T_H_W = self.draw_training_sigma_and_epsilon(
            x0_B_C_T_H_W.size(), condition)
        # action_sigma_placeholder = torch.zeros_like(action0_B_T_D)  # adding sigma to latent?
        # action_sigma_placeholder = action_sigma_placeholder[:, : (action0_B_T_D.shape[1] // 3), :]  # T=action_latent_t
        action_sigma_B_T, action_epsilon_B_T_D = self.draw_training_sigma_and_epsilon(
            action0_B_T_D.size(), condition)
        _, Ts = sigma_B_T.shape  # Ts=1
        action_sigma_B_T[:, :Ts] = sigma_B_T  # Action condition and video share the same sigma.
        # action_epsilon_B_T_D = torch.randn_like(action0_B_T_D, device=action0_B_T_D.device)
        # print("[DEBUG] training_step:", "sigma_B_T", sigma_B_T, "action_sigma_B_T", action_sigma_B_T)
        '''
        sigma_B_T.shape: torch.Size([12, 1])
        action_sigma_B_T.shape: torch.Size([12, 1])
        '''

        # Broadcast and split the input data and condition for model parallelism
        x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T = self.pipe.broadcast_split_for_model_parallelsim(
            x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T
        )  # Not supported yet.
        output_batch, kendall_loss_dict, _, _ = self.compute_loss_with_epsilon_and_sigma(
            x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T,
            action0_B_T_D=action0_B_T_D,
            action_epsilon_B_T_D=action_epsilon_B_T_D,
            action_sigma_B_T=action_sigma_B_T,
        )

        if self.loss_reduce == "mean":
            kendall_loss = (
                kendall_loss_dict['kendall_loss'].mean() * self.loss_scale
                + kendall_loss_dict['action_kendall_loss'].mean() * self.loss_scale
            )
        elif self.loss_reduce == "sum":
            kendall_loss = (
                kendall_loss_dict['kendall_loss'].sum(dim=1).mean() * self.loss_scale  # C-dim
                + kendall_loss_dict['action_kendall_loss'].sum(dim=2).mean() * self.loss_scale  # D-dim
            )
        else:
            raise ValueError(f"Invalid loss_reduce: {self.loss_reduce}")

        return output_batch, kendall_loss

