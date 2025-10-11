import numpy as np


def get_frames_from_multiview_video(video_B_VT_H_W_C, sample_n_views: int, start_idx=0, end_idx=None):
    # video_B_VT_H_W_C: (B,V*T,H,W,C), in [0,255]
    B = video_B_VT_H_W_C.shape[0]
    T = video_B_VT_H_W_C.shape[1] // sample_n_views
    V = sample_n_views
    assert video_B_VT_H_W_C.shape[1] == T * V, \
        f"Expected first dimension to be divisible by {V}, got {video_B_VT_H_W_C.shape[1]}"
    if end_idx is None:
        end_idx = T

    view_starts = np.arange(V) * T  # [0, T, 2T, ..., (V-1)*T]
    time_offsets = np.arange(start_idx, end_idx)  # [start_idx, start_idx+1, ..., end_idx-1]
    select_indices = (view_starts[:, None] + time_offsets).flatten()  # (V*(end_idx-start_idx),)

    select_frames_B_VT_H_W_C = video_B_VT_H_W_C[:, select_indices]  # (B, V*(end_idx-start_idx), H, W, C)
    return select_frames_B_VT_H_W_C


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
