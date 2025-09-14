from typing import Dict, Callable, List, Union, Optional
import collections
import os
import math
import numbers
import zarr
from functools import cached_property, partial

import numba
import numpy as np
import torch
import torch.nn as nn
import pytorch3d.transforms as pt  # git+https://github.com/facebookresearch/pytorch3d.git (0.7.8)

import imagecodecs  # imagecodecs-2025.3.30
import numcodecs
from numcodecs.abc import Codec
from numcodecs.registry import register_codec, get_codec


#############################
#### Dict utilities ####
#############################
def dict_apply(
    x: Dict[str, torch.Tensor], func: Callable[[torch.Tensor], torch.Tensor]
) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result


def dict_apply_split(
    x: Dict[str, torch.Tensor],
    split_func: Callable[[torch.Tensor], Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    results = collections.defaultdict(dict)
    for key, value in x.items():
        result = split_func(value)
        for k, v in result.items():
            results[k][key] = v
    return results


def dict_apply_reduce(
    x: List[Dict[str, torch.Tensor]],
    reduce_func: Callable[[List[torch.Tensor]], torch.Tensor],
) -> Dict[str, torch.Tensor]:
    result = dict()
    for key in x[0].keys():
        result[key] = reduce_func([x_[key] for x_ in x])
    return result
###############################
#### End Dict utilities ####
##############################


#####################
#### Normalizers ####
#####################
class DictOfTensorMixin(nn.Module):
    def __init__(self, params_dict=None):
        super().__init__()
        if params_dict is None:
            params_dict = nn.ParameterDict()
        self.params_dict = params_dict

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        def dfs_add(dest, keys, value: torch.Tensor):
            if len(keys) == 1:
                dest[keys[0]] = value
                return

            if keys[0] not in dest:
                dest[keys[0]] = nn.ParameterDict()
            dfs_add(dest[keys[0]], keys[1:], value)

        def load_dict(state_dict, prefix):
            out_dict = nn.ParameterDict()
            for key, value in state_dict.items():
                value: torch.Tensor
                if key.startswith(prefix):
                    param_keys = key[len(prefix) :].split(".")[1:]
                    # if len(param_keys) == 0:
                    #     import pdb; pdb.set_trace()
                    dfs_add(out_dict, param_keys, value.clone())
            return out_dict

        self.params_dict = load_dict(state_dict, prefix + "params_dict")
        self.params_dict.requires_grad_(False)
        return


class LinearNormalizer(DictOfTensorMixin):
    avaliable_modes = ["limits", "gaussian"]

    @torch.no_grad()
    def fit(
        self,
        data: Union[Dict, torch.Tensor, np.ndarray, zarr.Array],
        last_n_dims=1,
        dtype=torch.float32,
        mode="limits",
        output_max=1.0,
        output_min=-1.0,
        range_eps=1e-4,
        fit_offset=True,
    ):
        if isinstance(data, dict):
            for key, value in data.items():
                self.params_dict[key] = _fit(
                    value,
                    last_n_dims=last_n_dims,
                    dtype=dtype,
                    mode=mode,
                    output_max=output_max,
                    output_min=output_min,
                    range_eps=range_eps,
                    fit_offset=fit_offset,
                )
        else:
            self.params_dict["_default"] = _fit(
                data,
                last_n_dims=last_n_dims,
                dtype=dtype,
                mode=mode,
                output_max=output_max,
                output_min=output_min,
                range_eps=range_eps,
                fit_offset=fit_offset,
            )

    def __call__(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)

    def __getitem__(self, key: str):
        return SingleFieldLinearNormalizer(self.params_dict[key])

    def __setitem__(self, key: str, value: "SingleFieldLinearNormalizer"):
        self.params_dict[key] = value.params_dict

    def _normalize_impl(self, x, forward=True):
        if isinstance(x, dict):
            result = dict()
            for key, value in x.items():
                params = self.params_dict[key]
                result[key] = _normalize(value, params, forward=forward)
            return result
        else:
            if "_default" not in self.params_dict:
                raise RuntimeError("Not initialized")
            params = self.params_dict["_default"]
            return _normalize(x, params, forward=forward)

    def normalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self._normalize_impl(x, forward=True)

    def unnormalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self._normalize_impl(x, forward=False)

    def get_input_stats(self) -> Dict:
        if len(self.params_dict) == 0:
            raise RuntimeError("Not initialized")
        if len(self.params_dict) == 1 and "_default" in self.params_dict:
            return self.params_dict["_default"]["input_stats"]

        result = dict()
        for key, value in self.params_dict.items():
            if key != "_default":
                result[key] = value["input_stats"]
        return result

    def get_output_stats(self, key="_default"):
        input_stats = self.get_input_stats()
        if "min" in input_stats:
            # no dict
            return dict_apply(input_stats, self.normalize)

        result = dict()
        for key, group in input_stats.items():
            this_dict = dict()
            for name, value in group.items():
                this_dict[name] = self.normalize({key: value})[key]
            result[key] = this_dict
        return result


class SingleFieldLinearNormalizer(DictOfTensorMixin):
    avaliable_modes = ["limits", "gaussian"]

    @torch.no_grad()
    def fit(
        self,
        data: Union[torch.Tensor, np.ndarray, zarr.Array],
        last_n_dims=1,
        dtype=torch.float32,
        mode="limits",
        output_max=1.0,
        output_min=-1.0,
        range_eps=1e-4,
        fit_offset=True,
    ):
        self.params_dict = _fit(
            data,
            last_n_dims=last_n_dims,
            dtype=dtype,
            mode=mode,
            output_max=output_max,
            output_min=output_min,
            range_eps=range_eps,
            fit_offset=fit_offset,
        )

    @classmethod
    def create_fit(cls, data: Union[torch.Tensor, np.ndarray, zarr.Array], **kwargs):
        obj = cls()
        obj.fit(data, **kwargs)
        return obj

    @classmethod
    def create_manual(
        cls,
        scale: Union[torch.Tensor, np.ndarray],
        offset: Union[torch.Tensor, np.ndarray],
        input_stats_dict: Dict[str, Union[torch.Tensor, np.ndarray]],
    ):
        def to_tensor(x):
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x)
            x = x.flatten()
            return x

        # check
        for x in [offset] + list(input_stats_dict.values()):
            assert x.shape == scale.shape
            assert x.dtype == scale.dtype

        params_dict = nn.ParameterDict(
            {
                "scale": to_tensor(scale),
                "offset": to_tensor(offset),
                "input_stats": nn.ParameterDict(
                    dict_apply(input_stats_dict, to_tensor)
                ),
            }
        )
        return cls(params_dict)

    @classmethod
    def create_identity(cls, dtype=torch.float32):
        scale = torch.tensor([1], dtype=dtype)
        offset = torch.tensor([0], dtype=dtype)
        input_stats_dict = {
            "min": torch.tensor([-1], dtype=dtype),
            "max": torch.tensor([1], dtype=dtype),
            "mean": torch.tensor([0], dtype=dtype),
            "std": torch.tensor([1], dtype=dtype),
        }
        return cls.create_manual(scale, offset, input_stats_dict)

    def normalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=True)

    def unnormalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=False)

    def get_input_stats(self):
        return self.params_dict["input_stats"]

    def get_output_stats(self):
        return dict_apply(self.params_dict["input_stats"], self.normalize)

    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)


def _fit(
    data: Union[torch.Tensor, np.ndarray, zarr.Array],
    last_n_dims=1,
    dtype=torch.float32,
    mode="limits",
    output_max=1.0,
    output_min=-1.0,
    range_eps=1e-4,
    fit_offset=True,
):
    assert mode in ["limits", "gaussian"]
    assert last_n_dims >= 0
    assert output_max > output_min

    # convert data to torch and type
    if isinstance(data, zarr.Array):
        data = data[:]
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if dtype is not None:
        data = data.type(dtype)

    # convert shape
    dim = 1
    if last_n_dims > 0:
        dim = np.prod(data.shape[-last_n_dims:])
    data = data.reshape(-1, dim)

    # compute input stats min max mean std
    input_min, _ = data.min(axis=0)
    input_max, _ = data.max(axis=0)
    input_mean = data.mean(axis=0)
    input_std = data.std(axis=0)

    # compute scale and offset
    if mode == "limits":
        if fit_offset:
            # unit scale
            input_range = input_max - input_min
            ignore_dim = input_range < range_eps
            input_range[ignore_dim] = output_max - output_min
            scale = (output_max - output_min) / input_range
            offset = output_min - scale * input_min
            offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
            # ignore dims scaled to mean of output max and min
        else:
            # use this when data is pre-zero-centered.
            assert output_max > 0
            assert output_min < 0
            # unit abs
            output_abs = min(abs(output_min), abs(output_max))
            input_abs = torch.maximum(torch.abs(input_min), torch.abs(input_max))
            ignore_dim = input_abs < range_eps
            input_abs[ignore_dim] = output_abs
            # don't scale constant channels
            scale = output_abs / input_abs
            offset = torch.zeros_like(input_mean)
    elif mode == "gaussian":
        ignore_dim = input_std < range_eps
        scale = input_std.clone()
        scale[ignore_dim] = 1
        scale = 1 / scale

        if fit_offset:
            offset = -input_mean * scale
        else:
            offset = torch.zeros_like(input_mean)

    # save
    this_params = nn.ParameterDict(
        {
            "scale": scale,
            "offset": offset,
            "input_stats": nn.ParameterDict(
                {
                    "min": input_min,
                    "max": input_max,
                    "mean": input_mean,
                    "std": input_std,
                }
            ),
        }
    )
    for p in this_params.parameters():
        p.requires_grad_(False)
    return this_params


def _normalize(x, params, forward=True):
    assert "scale" in params
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    scale = params["scale"]
    offset = params["offset"]
    x = x.to(device=scale.device, dtype=scale.dtype)
    src_shape = x.shape
    x = x.reshape(-1, scale.shape[0])
    if forward:
        x = x * scale + offset
    else:
        x = (x - offset) / scale
    x = x.reshape(src_shape)
    return x


def get_range_normalizer_from_stat(stat, output_max=1, output_min=-1, range_eps=1e-7):
    # -1, 1 normalization
    input_max = stat["max"]
    input_min = stat["min"]
    input_range = input_max - input_min
    ignore_dim = input_range < range_eps
    input_range[ignore_dim] = output_max - output_min
    scale = (output_max - output_min) / input_range
    offset = output_min - scale * input_min
    offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )


def get_image_range_normalizer():
    scale = np.array([2], dtype=np.float32)
    offset = np.array([-1], dtype=np.float32)
    stat = {
        "min": np.array([0], dtype=np.float32),
        "max": np.array([1], dtype=np.float32),
        "mean": np.array([0.5], dtype=np.float32),
        "std": np.array([np.sqrt(1 / 12)], dtype=np.float32),
    }
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )


def get_image_identity_normalizer():
    scale = np.array([1], dtype=np.float32)
    offset = np.array([0], dtype=np.float32)
    stat = {
        "min": np.array([0], dtype=np.float32),
        "max": np.array([1], dtype=np.float32),
        "mean": np.array([0.5], dtype=np.float32),
        "std": np.array([np.sqrt(1 / 12)], dtype=np.float32),
    }
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )


def get_identity_normalizer_from_stat(stat):
    scale = np.ones_like(stat["min"])
    offset = np.zeros_like(stat["min"])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )


def robomimic_abs_action_normalizer_from_stat(stat, rotation_transformer):
    result = dict_apply_split(
        stat, lambda x: {"pos": x[..., :3], "rot": x[..., 3:6], "gripper": x[..., 6:]}
    )

    def get_pos_param_info(stat, output_max=1, output_min=-1, range_eps=1e-7):
        # -1, 1 normalization
        input_max = stat["max"]
        input_min = stat["min"]
        input_range = input_max - input_min
        ignore_dim = input_range < range_eps
        input_range[ignore_dim] = output_max - output_min
        scale = (output_max - output_min) / input_range
        offset = output_min - scale * input_min
        offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

        return {"scale": scale, "offset": offset}, stat

    def get_rot_param_info(stat):
        example = rotation_transformer.forward(stat["mean"])
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {
            "max": np.ones_like(example),
            "min": np.full_like(example, -1),
            "mean": np.zeros_like(example),
            "std": np.ones_like(example),
        }
        return {"scale": scale, "offset": offset}, info

    def get_gripper_param_info(stat):
        example = stat["max"]
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {
            "max": np.ones_like(example),
            "min": np.full_like(example, -1),
            "mean": np.zeros_like(example),
            "std": np.ones_like(example),
        }
        return {"scale": scale, "offset": offset}, info

    pos_param, pos_info = get_pos_param_info(result["pos"])
    rot_param, rot_info = get_rot_param_info(result["rot"])
    gripper_param, gripper_info = get_gripper_param_info(result["gripper"])

    param = dict_apply_reduce(
        [pos_param, rot_param, gripper_param], lambda x: np.concatenate(x, axis=-1)
    )
    info = dict_apply_reduce(
        [pos_info, rot_info, gripper_info], lambda x: np.concatenate(x, axis=-1)
    )

    return SingleFieldLinearNormalizer.create_manual(
        scale=param["scale"], offset=param["offset"], input_stats_dict=info
    )


def robomimic_abs_action_only_normalizer_from_stat(stat):
    result = dict_apply_split(stat, lambda x: {"pos": x[..., :3], "other": x[..., 3:]})

    def get_pos_param_info(stat, output_max=1, output_min=-1, range_eps=1e-7):
        # -1, 1 normalization
        input_max = stat["max"]
        input_min = stat["min"]
        input_range = input_max - input_min
        ignore_dim = input_range < range_eps
        input_range[ignore_dim] = output_max - output_min
        scale = (output_max - output_min) / input_range
        offset = output_min - scale * input_min
        offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

        return {"scale": scale, "offset": offset}, stat

    def get_other_param_info(stat):
        example = stat["max"]
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {
            "max": np.ones_like(example),
            "min": np.full_like(example, -1),
            "mean": np.zeros_like(example),
            "std": np.ones_like(example),
        }
        return {"scale": scale, "offset": offset}, info

    pos_param, pos_info = get_pos_param_info(result["pos"])
    other_param, other_info = get_other_param_info(result["other"])

    param = dict_apply_reduce(
        [pos_param, other_param], lambda x: np.concatenate(x, axis=-1)
    )
    info = dict_apply_reduce(
        [pos_info, other_info], lambda x: np.concatenate(x, axis=-1)
    )

    return SingleFieldLinearNormalizer.create_manual(
        scale=param["scale"], offset=param["offset"], input_stats_dict=info
    )


def robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat):
    Da = stat["max"].shape[-1]
    Dah = Da // 2
    result = dict_apply_split(
        stat,
        lambda x: {
            "pos0": x[..., :3],
            "other0": x[..., 3:Dah],
            "pos1": x[..., Dah : Dah + 3],
            "other1": x[..., Dah + 3 :],
        },
    )

    def get_pos_param_info(stat, output_max=1, output_min=-1, range_eps=1e-7):
        # -1, 1 normalization
        input_max = stat["max"]
        input_min = stat["min"]
        input_range = input_max - input_min
        ignore_dim = input_range < range_eps
        input_range[ignore_dim] = output_max - output_min
        scale = (output_max - output_min) / input_range
        offset = output_min - scale * input_min
        offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

        return {"scale": scale, "offset": offset}, stat

    def get_other_param_info(stat):
        example = stat["max"]
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {
            "max": np.ones_like(example),
            "min": np.full_like(example, -1),
            "mean": np.zeros_like(example),
            "std": np.ones_like(example),
        }
        return {"scale": scale, "offset": offset}, info

    pos0_param, pos0_info = get_pos_param_info(result["pos0"])
    pos1_param, pos1_info = get_pos_param_info(result["pos1"])
    other0_param, other0_info = get_other_param_info(result["other0"])
    other1_param, other1_info = get_other_param_info(result["other1"])

    param = dict_apply_reduce(
        [pos0_param, other0_param, pos1_param, other1_param],
        lambda x: np.concatenate(x, axis=-1),
    )
    info = dict_apply_reduce(
        [pos0_info, other0_info, pos1_info, other1_info],
        lambda x: np.concatenate(x, axis=-1),
    )

    return SingleFieldLinearNormalizer.create_manual(
        scale=param["scale"], offset=param["offset"], input_stats_dict=info
    )


def array_to_stats(arr: np.ndarray):
    stat = {
        "min": np.min(arr, axis=0),
        "max": np.max(arr, axis=0),
        "mean": np.mean(arr, axis=0),
        "std": np.std(arr, axis=0),
    }
    return stat


def concatenate_normalizer(normalizers: list):
    scale = torch.concatenate(
        [normalizer.params_dict["scale"] for normalizer in normalizers], axis=-1
    )
    offset = torch.concatenate(
        [normalizer.params_dict["offset"] for normalizer in normalizers], axis=-1
    )
    input_stats_dict = dict_apply_reduce(
        [normalizer.params_dict["input_stats"] for normalizer in normalizers],
        lambda x: torch.concatenate(x, axis=-1),
    )
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=input_stats_dict
    )
##########################
#### End Normalizers ####
#########################


class RotationTransformer:
    valid_reps = ["axis_angle", "euler_angles", "quaternion", "rotation_6d", "matrix"]

    def __init__(
        self,
        from_rep="axis_angle",
        to_rep="rotation_6d",
        from_convention=None,
        to_convention=None,
    ):
        """
        Valid representations

        Always use matrix as intermediate representation.
        """
        assert from_rep != to_rep
        assert from_rep in self.valid_reps
        assert to_rep in self.valid_reps
        if from_rep == "euler_angles":
            assert from_convention is not None
        if to_rep == "euler_angles":
            assert to_convention is not None

        forward_funcs = list()
        inverse_funcs = list()

        if from_rep != "matrix":
            funcs = [
                getattr(pt, f"{from_rep}_to_matrix"),
                getattr(pt, f"matrix_to_{from_rep}"),
            ]
            if from_convention is not None:
                funcs = [
                    partial(func, convention=from_convention)
                    for func in funcs
                ]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        if to_rep != "matrix":
            funcs = [
                getattr(pt, f"matrix_to_{to_rep}"),
                getattr(pt, f"{to_rep}_to_matrix"),
            ]
            if to_convention is not None:
                funcs = [
                    partial(func, convention=to_convention) for func in funcs
                ]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        inverse_funcs = inverse_funcs[::-1]

        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs

    @staticmethod
    def _apply_funcs(
        x: Union[np.ndarray, torch.Tensor], funcs: list
    ) -> Union[np.ndarray, torch.Tensor]:
        x_ = x
        if isinstance(x, np.ndarray):
            x_ = torch.from_numpy(x)
        x_: torch.Tensor
        for func in funcs:
            x_ = func(x_)
        y = x_
        if isinstance(x, np.ndarray):
            y = x_.numpy()
        return y

    def forward(
        self, x: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.forward_funcs)

    def inverse(
        self, x: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.inverse_funcs)


def log_warning(msg, *args, **kwargs):
    """Log message with level WARNING."""
    import logging

    logging.getLogger(__name__).warning(msg, *args, **kwargs)


def register_codecs(codecs=None, force=False, verbose=True):
    """Register codecs in this module with numcodecs."""
    for name, cls in globals().items():
        if not hasattr(cls, "codec_id") or name == "Codec":
            continue
        if codecs is not None and cls.codec_id not in codecs:
            continue
        try:
            try:
                get_codec({"id": cls.codec_id})
            except TypeError:
                # registered, but failed
                pass
        except ValueError:
            # not registered yet
            pass
        else:
            if not force:
                if verbose:
                    log_warning(f"numcodec {cls.codec_id!r} already registered")
                continue
            if verbose:
                log_warning(f"replacing registered numcodec {cls.codec_id!r}")
        register_codec(cls)


def protective_squeeze(x: np.ndarray):
    """
    Squeeze dim only if it's not the last dim.
    Image dim expected to be *, H, W, C
    """
    img_shape = x.shape[-3:]
    if len(x.shape) > 3:
        n_imgs = np.prod(x.shape[:-3])
        if n_imgs > 1:
            img_shape = (-1,) + img_shape
    return x.reshape(img_shape)


class Jpeg2k(Codec):
    """JPEG 2000 codec for numcodecs."""

    codec_id = "imagecodecs_jpeg2k"

    def __init__(
        self,
        level=None,
        codecformat=None,
        colorspace=None,
        tile=None,
        reversible=None,
        bitspersample=None,
        resolutions=None,
        numthreads=None,
        verbose=0,
    ):
        self.level = level
        self.codecformat = codecformat
        self.colorspace = colorspace
        self.tile = None if tile is None else tuple(tile)
        self.reversible = reversible
        self.bitspersample = bitspersample
        self.resolutions = resolutions
        self.numthreads = numthreads
        self.verbose = verbose

    def encode(self, buf):
        buf = protective_squeeze(np.asarray(buf))
        return imagecodecs.jpeg2k_encode(
            buf,
            level=self.level,
            codecformat=self.codecformat,
            colorspace=self.colorspace,
            tile=self.tile,
            reversible=self.reversible,
            bitspersample=self.bitspersample,
            resolutions=self.resolutions,
            numthreads=self.numthreads,
            verbose=self.verbose,
        )

    def decode(self, buf, out=None):
        return imagecodecs.jpeg2k_decode(
            buf, verbose=self.verbose, numthreads=self.numthreads, out=out
        )


###########################
#### Replay Buffer ####
###########################
def check_chunks_compatible(chunks: tuple, shape: tuple):
    assert len(shape) == len(chunks)
    for c in chunks:
        assert isinstance(c, numbers.Integral)
        assert c > 0


def rechunk_recompress_array(
    group, name, chunks=None, chunk_length=None, compressor=None, tmp_key="_temp"
):
    old_arr = group[name]
    if chunks is None:
        if chunk_length is not None:
            chunks = (chunk_length,) + old_arr.chunks[1:]
        else:
            chunks = old_arr.chunks
    check_chunks_compatible(chunks, old_arr.shape)

    if compressor is None:
        compressor = old_arr.compressor

    if (chunks == old_arr.chunks) and (compressor == old_arr.compressor):
        # no change
        return old_arr

    # rechunk recompress
    group.move(name, tmp_key)
    old_arr = group[tmp_key]
    n_copied, n_skipped, n_bytes_copied = zarr.copy(
        source=old_arr,
        dest=group,
        name=name,
        chunks=chunks,
        compressor=compressor,
    )
    del group[tmp_key]
    arr = group[name]
    return arr


def get_optimal_chunks(shape, dtype, target_chunk_bytes=2e6, max_chunk_length=None):
    """
    Common shapes
    T,D
    T,N,D
    T,H,W,C
    T,N,H,W,C
    """
    itemsize = np.dtype(dtype).itemsize
    # reversed
    rshape = list(shape[::-1])
    if max_chunk_length is not None:
        rshape[-1] = int(max_chunk_length)
    split_idx = len(shape) - 1
    for i in range(len(shape) - 1):
        this_chunk_bytes = itemsize * np.prod(rshape[:i])
        next_chunk_bytes = itemsize * np.prod(rshape[: i + 1])
        if (
            this_chunk_bytes <= target_chunk_bytes
            and next_chunk_bytes > target_chunk_bytes
        ):
            split_idx = i

    rchunks = rshape[:split_idx]
    item_chunk_bytes = itemsize * np.prod(rshape[:split_idx])
    this_max_chunk_length = rshape[split_idx]
    next_chunk_length = min(
        this_max_chunk_length, math.ceil(target_chunk_bytes / item_chunk_bytes)
    )
    rchunks.append(next_chunk_length)
    len_diff = len(shape) - len(rchunks)
    rchunks.extend([1] * len_diff)
    chunks = tuple(rchunks[::-1])
    # print(np.prod(chunks) * itemsize / target_chunk_bytes)
    return chunks


class ReplayBuffer:
    """
    Zarr-based temporal datastructure.
    Assumes first dimension to be time. Only chunk in time dimension.
    """

    def __init__(self, root: Union[zarr.Group, Dict[str, dict]]):
        """
        Dummy constructor. Use copy_from* and create_from* class methods instead.
        """
        assert "data" in root
        assert "meta" in root
        assert "episode_ends" in root["meta"]
        for key, value in root["data"].items():
            assert value.shape[0] == root["meta"]["episode_ends"][-1]
        self.root = root

    # ============= create constructors ===============
    @classmethod
    def create_empty_zarr(cls, storage=None, root=None):
        if root is None:
            if storage is None:
                storage = zarr.MemoryStore()
            root = zarr.group(store=storage)
        data = root.require_group("data", overwrite=False)
        meta = root.require_group("meta", overwrite=False)
        if "episode_ends" not in meta:
            episode_ends = meta.zeros(
                "episode_ends",
                shape=(0,),
                dtype=np.int64,
                compressor=None,
                overwrite=False,
            )
        return cls(root=root)

    @classmethod
    def create_empty_numpy(cls):
        root = {
            "data": dict(),
            "meta": {"episode_ends": np.zeros((0,), dtype=np.int64)},
        }
        return cls(root=root)

    @classmethod
    def create_from_group(cls, group, **kwargs):
        if "data" not in group:
            # create from stratch
            buffer = cls.create_empty_zarr(root=group, **kwargs)
        else:
            # already exist
            buffer = cls(root=group, **kwargs)
        return buffer

    @classmethod
    def create_from_path(cls, zarr_path, mode="r", **kwargs):
        """
        Open a on-disk zarr directly (for dataset larger than memory).
        Slower.
        """
        group = zarr.open(os.path.expanduser(zarr_path), mode)
        return cls.create_from_group(group, **kwargs)

    # ============= copy constructors ===============
    @classmethod
    def copy_from_store(
            cls,
            src_store,
            store=None,
            keys=None,
            chunks: Dict[str, tuple] = dict(),
            compressors: Union[dict, str, numcodecs.abc.Codec] = dict(),
            if_exists="replace",
            **kwargs,
    ):
        """
        Load to memory.
        """

        src_root = zarr.group(src_store)
        root = None
        if store is None:
            # numpy backend
            meta = dict()
            for key, value in src_root["meta"].items():
                if len(value.shape) == 0:
                    meta[key] = np.array(value)
                else:
                    meta[key] = value[:]

            if keys is None:
                keys = src_root["data"].keys()
            data = dict()
            for key in keys:
                arr = src_root["data"][key]
                # data[key] = arr[:]
                data[key] = arr

            root = {"meta": meta, "data": data}

        else:
            root = zarr.group(store=store)
            # copy without recompression
            n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                source=src_store,
                dest=store,
                source_path="/meta",
                dest_path="/meta",
                if_exists=if_exists,
            )
            data_group = root.create_group("data", overwrite=True)
            if keys is None:
                keys = src_root["data"].keys()
            for key in keys:
                value = src_root["data"][key]
                cks = cls._resolve_array_chunks(chunks=chunks, key=key, array=value)
                cpr = cls._resolve_array_compressor(
                    compressors=compressors, key=key, array=value
                )
                if cks == value.chunks and cpr == value.compressor:
                    # copy without recompression
                    this_path = "/data/" + key
                    n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                        source=src_store,
                        dest=store,
                        source_path=this_path,
                        dest_path=this_path,
                        if_exists=if_exists,
                    )
                else:
                    # copy with recompression
                    n_copied, n_skipped, n_bytes_copied = zarr.copy(
                        source=value,
                        dest=data_group,
                        name=key,
                        chunks=cks,
                        compressor=cpr,
                        if_exists=if_exists,
                    )
        buffer = cls(root=root)
        return buffer

    @classmethod
    def copy_from_path(
            cls,
            zarr_path,
            backend=None,
            store=None,
            keys=None,
            chunks: Dict[str, tuple] = dict(),
            compressors: Union[dict, str, numcodecs.abc.Codec] = dict(),
            if_exists="replace",
            **kwargs,
    ):
        """
        Copy a on-disk zarr to in-memory compressed.
        Recommended
        """

        if backend == "numpy":
            print("backend argument is deprecated!")
            store = None
        group = zarr.open(os.path.expanduser(zarr_path), "r")
        return cls.copy_from_store(
            src_store=group.store,
            store=store,
            keys=keys,
            chunks=chunks,
            compressors=compressors,
            if_exists=if_exists,
            **kwargs,
        )

    # ============= save methods ===============
    def save_to_store(
            self,
            store,
            chunks: Optional[Dict[str, tuple]] = dict(),
            compressors: Union[str, numcodecs.abc.Codec, dict] = dict(),
            if_exists="replace",
            **kwargs,
    ):

        root = zarr.group(store)
        if self.backend == "zarr":
            # recompression free copy
            n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                source=self.root.store,
                dest=store,
                source_path="/meta",
                dest_path="/meta",
                if_exists=if_exists,
            )
        else:
            meta_group = root.create_group("meta", overwrite=True)
            # save meta, no chunking
            for key, value in self.root["meta"].items():
                _ = meta_group.array(
                    name=key, data=value, shape=value.shape, chunks=value.shape
                )

        # save data, chunk
        data_group = root.create_group("data", overwrite=True)
        for key, value in self.root["data"].items():
            cks = self._resolve_array_chunks(chunks=chunks, key=key, array=value)
            cpr = self._resolve_array_compressor(
                compressors=compressors, key=key, array=value
            )
            if isinstance(value, zarr.Array):
                if cks == value.chunks and cpr == value.compressor:
                    # copy without recompression
                    this_path = "/data/" + key
                    n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                        source=self.root.store,
                        dest=store,
                        source_path=this_path,
                        dest_path=this_path,
                        if_exists=if_exists,
                    )
                else:
                    # copy with recompression
                    n_copied, n_skipped, n_bytes_copied = zarr.copy(
                        source=value,
                        dest=data_group,
                        name=key,
                        chunks=cks,
                        compressor=cpr,
                        if_exists=if_exists,
                    )
            else:
                # numpy
                _ = data_group.array(name=key, data=value, chunks=cks, compressor=cpr)
        return store

    def save_to_path(
            self,
            zarr_path,
            chunks: Optional[Dict[str, tuple]] = dict(),
            compressors: Union[str, numcodecs.abc.Codec, dict] = dict(),
            if_exists="replace",
            **kwargs,
    ):
        store = zarr.DirectoryStore(os.path.expanduser(zarr_path))
        return self.save_to_store(
            store, chunks=chunks, compressors=compressors, if_exists=if_exists, **kwargs
        )

    @staticmethod
    def resolve_compressor(compressor="default"):
        if compressor == "default":
            compressor = numcodecs.Blosc(
                cname="lz4", clevel=5, shuffle=numcodecs.Blosc.NOSHUFFLE
            )
        elif compressor == "disk":
            compressor = numcodecs.Blosc(
                "zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE
            )
        return compressor

    @classmethod
    def _resolve_array_compressor(
            cls, compressors: Union[dict, str, numcodecs.abc.Codec], key, array
    ):
        # allows compressor to be explicitly set to None
        cpr = "nil"
        if isinstance(compressors, dict):
            if key in compressors:
                cpr = cls.resolve_compressor(compressors[key])
            elif isinstance(array, zarr.Array):
                cpr = array.compressor
        else:
            cpr = cls.resolve_compressor(compressors)
        # backup default
        if cpr == "nil":
            cpr = cls.resolve_compressor("default")
        return cpr

    @classmethod
    def _resolve_array_chunks(cls, chunks: Union[dict, tuple], key, array):
        cks = None
        if isinstance(chunks, dict):
            if key in chunks:
                cks = chunks[key]
            elif isinstance(array, zarr.Array):
                cks = array.chunks
        elif isinstance(chunks, tuple):
            cks = chunks
        else:
            raise TypeError(f"Unsupported chunks type {type(chunks)}")
        # backup default
        if cks is None:
            cks = get_optimal_chunks(shape=array.shape, dtype=array.dtype)
        # check
        check_chunks_compatible(chunks=cks, shape=array.shape)
        return cks

    # ============= properties =================
    @cached_property
    def data(self):
        return self.root["data"]

    @cached_property
    def meta(self):
        return self.root["meta"]

    def update_meta(self, data):
        # sanitize data
        np_data = dict()
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                np_data[key] = value
            else:
                arr = np.array(value)
                if arr.dtype == object:
                    raise TypeError(f"Invalid value type {type(value)}")
                np_data[key] = arr

        meta_group = self.meta
        if self.backend == "zarr":
            for key, value in np_data.items():
                _ = meta_group.array(
                    name=key,
                    data=value,
                    shape=value.shape,
                    chunks=value.shape,
                    overwrite=True,
                )
        else:
            meta_group.update(np_data)

        return meta_group

    @property
    def episode_ends(self):
        return self.meta["episode_ends"]

    def get_episode_idxs(self):
        import numba

        numba.jit(nopython=True)

        def _get_episode_idxs(episode_ends):
            result = np.zeros((episode_ends[-1],), dtype=np.int64)
            for i in range(len(episode_ends)):
                start = 0
                if i > 0:
                    start = episode_ends[i - 1]
                end = episode_ends[i]
                for idx in range(start, end):
                    result[idx] = i
            return result

        return _get_episode_idxs(self.episode_ends)

    @property
    def backend(self):
        backend = "numpy"
        if isinstance(self.root, zarr.Group):
            backend = "zarr"
        return backend

    # =========== dict-like API ==============
    def __repr__(self) -> str:
        if self.backend == "zarr":
            return str(self.root.tree())
        else:
            return super().__repr__()

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data

    # =========== our API ==============
    @property
    def n_steps(self):
        if len(self.episode_ends) == 0:
            return 0
        return self.episode_ends[-1]

    @property
    def n_episodes(self):
        return len(self.episode_ends)

    @property
    def chunk_size(self):
        if self.backend == "zarr":
            return next(iter(self.data.arrays()))[-1].chunks[0]
        return None

    @property
    def episode_lengths(self):
        ends = self.episode_ends[:]
        ends = np.insert(ends, 0, 0)
        lengths = np.diff(ends)
        return lengths

    def add_episode(
            self,
            data: Dict[str, np.ndarray],
            chunks: Optional[Dict[str, tuple]] = dict(),
            compressors: Union[str, numcodecs.abc.Codec, dict] = dict(),
    ):
        assert len(data) > 0
        is_zarr = self.backend == "zarr"

        curr_len = self.n_steps
        episode_length = None
        for key, value in data.items():
            assert len(value.shape) >= 1
            if episode_length is None:
                episode_length = len(value)
            else:
                assert episode_length == len(value)
        new_len = curr_len + episode_length

        for key, value in data.items():
            new_shape = (new_len,) + value.shape[1:]
            # create array
            if key not in self.data:
                if is_zarr:
                    cks = self._resolve_array_chunks(
                        chunks=chunks, key=key, array=value
                    )
                    cpr = self._resolve_array_compressor(
                        compressors=compressors, key=key, array=value
                    )
                    arr = self.data.zeros(
                        name=key,
                        shape=new_shape,
                        chunks=cks,
                        dtype=value.dtype,
                        compressor=cpr,
                    )
                else:
                    # copy data to prevent modify
                    arr = np.zeros(shape=new_shape, dtype=value.dtype)
                    self.data[key] = arr
            else:
                arr = self.data[key]
                assert value.shape[1:] == arr.shape[1:]
                # same method for both zarr and numpy
                if is_zarr:
                    arr.resize(new_shape)
                else:
                    arr.resize(new_shape, refcheck=False)
            # copy data
            arr[-value.shape[0]:] = value

        # append to episode ends
        episode_ends = self.episode_ends
        if is_zarr:
            episode_ends.resize(episode_ends.shape[0] + 1)
        else:
            episode_ends.resize(episode_ends.shape[0] + 1, refcheck=False)
        episode_ends[-1] = new_len

        # rechunk
        if is_zarr:
            if episode_ends.chunks[0] < episode_ends.shape[0]:
                rechunk_recompress_array(
                    self.meta,
                    "episode_ends",
                    chunk_length=int(episode_ends.shape[0] * 1.5),
                )

    def drop_episode(self):
        is_zarr = self.backend == "zarr"
        episode_ends = self.episode_ends[:].copy()
        assert len(episode_ends) > 0
        start_idx = 0
        if len(episode_ends) > 1:
            start_idx = episode_ends[-2]
        for key, value in self.data.items():
            new_shape = (start_idx,) + value.shape[1:]
            if is_zarr:
                value.resize(new_shape)
            else:
                value.resize(new_shape, refcheck=False)
        if is_zarr:
            self.episode_ends.resize(len(episode_ends) - 1)
        else:
            self.episode_ends.resize(len(episode_ends) - 1, refcheck=False)

    def pop_episode(self):
        assert self.n_episodes > 0
        episode = self.get_episode(self.n_episodes - 1, copy=True)
        self.drop_episode()
        return episode

    def extend(self, data):
        self.add_episode(data)

    def get_episode(self, idx, copy=False):
        idx = list(range(len(self.episode_ends)))[idx]
        start_idx = 0
        if idx > 0:
            start_idx = self.episode_ends[idx - 1]
        end_idx = self.episode_ends[idx]
        result = self.get_steps_slice(start_idx, end_idx, copy=copy)
        return result

    def get_episode_slice(self, idx):
        start_idx = 0
        if idx > 0:
            start_idx = self.episode_ends[idx - 1]
        end_idx = self.episode_ends[idx]
        return slice(start_idx, end_idx)

    def get_steps_slice(self, start, stop, step=None, copy=False):
        _slice = slice(start, stop, step)

        result = dict()
        for key, value in self.data.items():
            x = value[_slice]
            if copy and isinstance(value, np.ndarray):
                x = x.copy()
            result[key] = x
        return result

    # =========== chunking =============
    def get_chunks(self) -> dict:
        assert self.backend == "zarr"
        chunks = dict()
        for key, value in self.data.items():
            chunks[key] = value.chunks
        return chunks

    def set_chunks(self, chunks: dict):
        assert self.backend == "zarr"
        for key, value in chunks.items():
            if key in self.data:
                arr = self.data[key]
                if value != arr.chunks:
                    check_chunks_compatible(chunks=value, shape=arr.shape)
                    rechunk_recompress_array(self.data, key, chunks=value)

    def get_compressors(self) -> dict:
        assert self.backend == "zarr"
        compressors = dict()
        for key, value in self.data.items():
            compressors[key] = value.compressor
        return compressors

    def set_compressors(self, compressors: dict):
        assert self.backend == "zarr"
        for key, value in compressors.items():
            if key in self.data:
                arr = self.data[key]
                compressor = self.resolve_compressor(value)
                if compressor != arr.compressor:
                    rechunk_recompress_array(self.data, key, compressor=compressor)
###########################
#### End Replay Buffer ####
###########################


##########################
#### Samplers ###########
##########################
@numba.jit(nopython=True)
def create_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    episode_mask: np.ndarray,
    pad_before: int = 0,
    pad_after: int = 0,
    debug: bool = True,
) -> np.ndarray:

    episode_mask.shape == episode_ends.shape
    pad_before = min(max(pad_before, 0), sequence_length - 1)  # 1
    pad_after = min(max(pad_after, 0), sequence_length - 1)  # 7

    indices = list()
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            # skip episode
            continue
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before  # -1
        max_start = episode_length - sequence_length + pad_after  # 93

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert start_offset >= 0
                assert end_offset >= 0
                assert (sample_end_idx - sample_start_idx) == (
                    buffer_end_idx - buffer_start_idx
                )
            indices.append(
                [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
            )
    indices = np.array(indices)
    return indices


def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    print("---------------------------------------------------------")
    print(f"Validation episodes: {val_idxs}")
    print("---------------------------------------------------------")
    val_mask[val_idxs] = True
    return val_mask


class SequenceSampler:
    def __init__(
            self,
            replay_buffer: ReplayBuffer,
            sequence_length: int,
            pad_before: int = 0,
            pad_after: int = 0,
            keys=None,
            key_first_k=dict(),
            episode_mask: Optional[np.ndarray] = None,
    ):
        """
        key_first_k: dict str: int
            Only take first k data from these keys (to improve perf)
        """

        super().__init__()
        assert sequence_length >= 1
        if keys is None:
            keys = list(replay_buffer.keys())

        episode_ends = replay_buffer.episode_ends[:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        if np.any(episode_mask):
            indices = create_indices(
                episode_ends,
                sequence_length=sequence_length,
                pad_before=pad_before,
                pad_after=pad_after,
                episode_mask=episode_mask,
            )
        else:
            indices = np.zeros((0, 4), dtype=np.int64)

        self.indices = indices
        self.keys = list(keys)  # prevent OmegaConf list performance problem
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.key_first_k = key_first_k

    def __len__(self):
        return len(self.indices)

    def sample_sequence(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = (
            self.indices[idx]
        )

        result = dict()
        for key in self.keys:
            input_arr = self.replay_buffer[key]
            # performance optimization, avoid small allocation if possible
            if key not in self.key_first_k:
                sample = input_arr[buffer_start_idx:buffer_end_idx]
            else:
                # performance optimization, only load used obs steps
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(self.key_first_k[key], n_data)
                # fill value with Nan to catch bugs
                # the non-loaded region should never be used
                sample = np.full(
                    (n_data,) + input_arr.shape[1:],
                    fill_value=np.nan,
                    dtype=input_arr.dtype,
                )
                try:
                    sample[:k_data] = input_arr[
                        buffer_start_idx: buffer_start_idx + k_data
                    ]
                except Exception as e:
                    raise e
            data = sample
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                data = np.zeros(
                    shape=(self.sequence_length,) + input_arr.shape[1:],
                    dtype=input_arr.dtype,
                )
                if sample_start_idx > 0:
                    data[:sample_start_idx] = sample[0]
                if sample_end_idx < self.sequence_length:
                    data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            result[key] = data
        return result
##########################
#### End Samplers ###########
##########################

