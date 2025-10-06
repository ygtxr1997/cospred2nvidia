import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from imaginaire.utils import distributed, log, misc


def save_action_as_image(action_T_D: np.ndarray, save_path):
    """
    Save action sequence as trajectory image.
    action_T_D: (T,D) numpy array, expected shape (T,2) or (T,3) with values in [-1,1]
    """
    if action_T_D.shape[1] == 1:
        save_1d_action_as_image(action_T_D, save_path)
    elif action_T_D.shape[1] == 2:
        save_2d_action_as_image(action_T_D, save_path)
    elif action_T_D.shape[1] == 3:
        save_3d_action_as_image(action_T_D, save_path)
    else:
        raise ValueError(f"Unsupported action dimension: {action_T_D.shape[1]}")


def save_1d_action_as_image(action_T_D: np.ndarray, save_path):
    """
    Save 1D action sequence as time series plot with time-based color gradient.
    action_T_D: (T,1) numpy array, expected shape (T,1) with values in [-1,1]
    """
    print("[DEBUG] save_1d_action_as_image:", action_T_D.shape, action_T_D.dtype, action_T_D.min(), action_T_D.max())

    # Validate input shape
    if len(action_T_D.shape) != 2 or action_T_D.shape[1] != 1:
        raise ValueError(f"Expected shape (T,1), got {action_T_D.shape}")

    # Extract the 1D values
    values = action_T_D[:, 0]  # (T,)
    T = len(action_T_D)
    time_steps = np.arange(T)

    # Create 2D plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create color map from light to dark over time
    colors = plt.cm.plasma(np.linspace(0.2, 1.0, T))

    # Plot trajectory points
    for i in range(T):
        ax.scatter(time_steps[i], values[i], c=[colors[i]], s=50, alpha=0.8, edgecolors='white', linewidth=0.5)

    # Plot trajectory lines
    if T > 1:
        for i in range(T - 1):
            ax.plot([time_steps[i], time_steps[i + 1]],
                    [values[i], values[i + 1]],
                    color=colors[i], alpha=0.6, linewidth=2)

    # Set up the plot
    y_min, y_max = -0.2, 0.2
    y_min = min(values.min(), y_min)  # in case action out of our expected range: [-0.2,1.2]
    y_max = max(values.max(), y_max)

    ax.set_xlim(-0.5, T - 0.5)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Action Value')
    ax.set_title('1D Action Trajectory Over Time')
    ax.grid(True, alpha=0.3)

    # Add colorbar to show time progression
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=mcolors.Normalize(vmin=0, vmax=T - 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=10)
    cbar.set_label('Time Step')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved 1D action trajectory visualization to {save_path}")


def save_2d_action_as_image(action_T_D: np.ndarray, save_path):
    """
    Save action sequence as trajectory on 512x512 image with time-based color gradient.
    action_T_D: (T,D) numpy array, expected shape (T,2) with values in [-1,1]
    """
    # Create 512x512 figure
    print("[DEBUG] save_action_as_image:", action_T_D.shape, action_T_D.dtype, action_T_D.min(), action_T_D.max())
    fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=100)

    # Convert normalized coordinates [0,1] to pixel coordinates [0,512]
    trajectory = action_T_D * 256. + 256.

    # Create color map from light to dark over time
    T = len(action_T_D)
    colors = plt.cm.plasma(np.linspace(0.2, 1.0, T))  # plasma colormap from light to dark

    # Plot trajectory points
    for i in range(T):
        x, y = trajectory[i]
        ax.scatter(x, y, c=[colors[i]], s=20, alpha=0.8, edgecolors='white', linewidth=0.5)

    # Plot trajectory lines
    if T > 1:
        for i in range(T - 1):
            ax.plot([trajectory[i][0], trajectory[i + 1][0]],
                    [trajectory[i][1], trajectory[i + 1][1]],
                    color=colors[i], alpha=0.6, linewidth=1.5)

    # Set up the plot
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 512)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Invert y-axis to match image coordinates
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Action Trajectory Over Time')

    # Add colorbar to show time progression
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=mcolors.Normalize(vmin=0, vmax=T - 1))
    sm.set_array([])
    # cbar = plt.colorbar(sm, ax=ax)
    # cbar.set_label('Time Step')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    log.info(f"Saved action trajectory visualization to {save_path}")


def save_3d_action_as_image(action_T_D: np.ndarray, save_path):
    """
    Save 3D action sequence as trajectory with time-based color gradient.
    action_T_D: (T,D) numpy array, expected shape (T,3) with values in [-1,1]
    """
    print("[DEBUG] save_3d_action_as_image:", action_T_D.shape, action_T_D.dtype, action_T_D.min(), action_T_D.max())

    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Convert normalized coordinates [-1,1] to visualization coordinates
    # trajectory = action_T_D * 256. + 256.  # [0, 512] range for consistency
    trajectory = action_T_D  # do not shift to [0,512], keep in [-1,1] for better view

    # Create color map from light to dark over time
    T = len(action_T_D)
    colors = plt.cm.plasma(np.linspace(0.2, 1.0, T))

    # Plot trajectory points
    for i in range(T):
        x, y, z = trajectory[i]
        ax.scatter(x, y, z, c=[colors[i]], s=30, alpha=0.8, edgecolors='white', linewidth=0.5)

    # Plot trajectory lines
    if T > 1:
        for i in range(T - 1):
            ax.plot([trajectory[i][0], trajectory[i + 1][0]],
                    [trajectory[i][1], trajectory[i + 1][1]],
                    [trajectory[i][2], trajectory[i + 1][2]],
                    color=colors[i], alpha=0.6, linewidth=2)

    # Set up the plot
    xyz_min, xyz_max = -0.2, 1.2
    xyz_min = min(trajectory.min(), xyz_min)  # in case actio out of our expected range: [-0.2,1.2]
    xyz_max = max(trajectory.max(), xyz_max)
    ax.set_xlim(xyz_min, xyz_max)
    ax.set_ylim(xyz_min, xyz_max)
    ax.set_zlim(xyz_min, xyz_max)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('3D Action Trajectory Over Time')

    # Add colorbar to show time progression
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=mcolors.Normalize(vmin=0, vmax=T - 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Time Step')

    # Set viewing angle for better visualization
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    log.info(f"Saved 3D action trajectory visualization to {save_path}")

