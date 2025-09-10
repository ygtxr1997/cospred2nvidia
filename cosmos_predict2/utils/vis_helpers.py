import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from imaginaire.utils import distributed, log, misc


def save_action_as_image(action_T_D: np.ndarray, save_path):
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