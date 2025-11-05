import numpy as np
import torch


def get_frames_from_multiview_video(video_B_VT_H_W_C, sample_n_views: int, start_idx=0, end_idx=None,
                                    skipped_indices: np.ndarray = None,  # indicate which frames to select
                                    ):
    # video_B_VT_H_W_C: (B,V*T,H,W,C), in [0,255]
    B = video_B_VT_H_W_C.shape[0]
    T = video_B_VT_H_W_C.shape[1] // sample_n_views
    V = sample_n_views
    assert video_B_VT_H_W_C.shape[1] == T * V, \
        f"Expected first dimension to be divisible by {V}, got {video_B_VT_H_W_C.shape[1]}"
    if end_idx is None:
        end_idx = T

    view_starts = np.arange(V) * T  # [0, T, 2T, ..., (V-1)*T]
    if skipped_indices is None:
        time_offsets = np.arange(start_idx, end_idx)  # [start_idx, start_idx+1, ..., end_idx-1]
    else:
        assert np.all((skipped_indices >= start_idx) & (skipped_indices < end_idx)), \
            f"skipped_indices must be in the range [{start_idx}, {end_idx})"
        time_offsets = skipped_indices  # [start_idx, ..., now_idx, (skipped), now_idx+skip, ..., end_idx-1]
    select_indices = (view_starts[:, None] + time_offsets).flatten()  # (V*(end_idx-start_idx),)

    select_frames_B_VT_H_W_C = video_B_VT_H_W_C[:, select_indices]  # (B, V*(end_idx-start_idx), H, W, C)
    return select_frames_B_VT_H_W_C


def get_skipped_indices(start_idx: int, end_idx: int, obs_len: int, future_frame_skip: int = 1) -> np.ndarray:
    # [start_idx, end_idx)
    assert end_idx > start_idx, f"end_idx {end_idx} must be greater than start_idx {start_idx}"
    assert start_idx + obs_len <= end_idx, \
        f"obs_len {obs_len} exceeds the range from start_idx {start_idx} to end_idx {end_idx}"
    pad_before = obs_len - 1
    skipped_indices = np.concatenate([
        np.arange(start_idx, start_idx + pad_before),  # v1-1
        np.arange(start_idx + pad_before, end_idx)[::future_frame_skip],  # 1+v2
    ])
    one_plus_skipped_v2 = (end_idx - start_idx - pad_before) // future_frame_skip
    assert one_plus_skipped_v2 * future_frame_skip == (end_idx - start_idx - obs_len), \
        (f"Expected (end_idx - start_idx) to be divisible by future_frame_skip {future_frame_skip},"
         f"got end_idx={end_idx}, start_idx={start_idx}, obs_len={obs_len}")
    assert skipped_indices[-1] == end_idx - 1, \
        f"Last skipped index {skipped_indices[-1]} does not match end_idx-1 {end_idx - 1}"
    return skipped_indices


def cat_multiview_video_with_zeros(video_B_VT_H_W_C: np.ndarray, sample_n_views: int, zero_length: int):
    V = sample_n_views
    B = video_B_VT_H_W_C.shape[0]
    total_frames = video_B_VT_H_W_C.shape[1]
    T = total_frames // V
    H, W, C = video_B_VT_H_W_C.shape[2:]

    assert total_frames == T * V, f"Expected first dimension to be divisible by {V}, got {total_frames}"

    video_reshaped = video_B_VT_H_W_C.reshape(B, V, T, H, W, C)
    zeros = np.zeros((B, V, zero_length, H, W, C), dtype=np.uint8)

    # Concatenate along time dimension: (V, T+zero_length, H, W, C)
    video_with_zeros = np.concatenate([video_reshaped, zeros], axis=2)

    # Reshape back: (V, T+zero_length, H, W, C) -> (V*(T+zero_length), H, W, C)
    result = video_with_zeros.reshape(B, V * (T + zero_length), H, W, C)
    return result


def replace_multiview_video_back_with_another(
        ori_video_B_C_VT_H_W: torch.Tensor,
        new_video_B_C_VT_H_W: torch.Tensor,
        sample_n_views: int,
        replace_length: int,
):
    """ Both sub-videos extracted from last frames """
    B, C, VT_ori, H, W = ori_video_B_C_VT_H_W.shape
    VT_new = new_video_B_C_VT_H_W.shape[2]
    T_ori = VT_ori // sample_n_views
    T_new = VT_new // sample_n_views
    V = sample_n_views

    assert VT_ori == T_ori * V, \
        f"Expected third dimension to be divisible by {V}, got {VT_ori}"
    assert VT_new == T_new * V, \
        f"Expected third dimension to be divisible by {V}, got {VT_new}"
    assert replace_length <= T_ori and replace_length <= T_new, \
        f"replace_length {replace_length} exceeds original or new video length"
    assert ori_video_B_C_VT_H_W.shape[0] == new_video_B_C_VT_H_W.shape[0], \
        f"Shape mismatch: {ori_video_B_C_VT_H_W.shape[0]} vs {new_video_B_C_VT_H_W.shape[0]}"

    # Reshape to (B, C, V, T, H, W)
    ori_reshaped = ori_video_B_C_VT_H_W.view(B, C, V, T_ori, H, W)
    new_reshaped = new_video_B_C_VT_H_W.view(B, C, V, T_new, H, W)

    # Replace the last `replace_length` frames along the time dimension
    ori_reshaped[:, :, :, -replace_length:, :, :] = new_reshaped[:, :, :, -replace_length:, :, :]

    # Reshape back to (B, C, V*T, H, W)
    result = ori_reshaped.view(B, C, VT_ori, H, W)
    return result

