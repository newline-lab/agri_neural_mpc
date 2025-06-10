import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import seaborn as sns
from PIL import Image
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import io
from cairosvg import svg2png
png_bytes = svg2png(url="/home/pantheon/drea/neural_mpc/ros/src/nmpc_ros/results/sun_orientation.svg")
logo_arr = mpimg.imread(io.BytesIO(png_bytes), format='PNG')


paired_palette = sns.color_palette("Paired")
paired_palette[0]  # light blue
paired_palette[1]  # orange
paired_palette[2]  # green
paired_palette[3]  # red
dark_paired_palette = sns.color_palette("dark")

def get_latest_csv(mode, directory, suffix="plot_data.csv"):
    pattern = os.path.join(directory, f"{mode}*{suffix}")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def load_csv_data(csv_path):
    with open(csv_path, 'r') as f:
        first_line = f.readline().strip()
    parts = first_line.split(',')
    if parts[0] == "tree_positions":
        tree_positions_list = [float(x) for x in parts[1:] if x] # ensure x is not empty string
        if tree_positions_list:
            num_trees = len(tree_positions_list) // 2
            tree_positions = np.array(tree_positions_list).reshape(num_trees, 2)
        else:
            tree_positions = np.array([]).reshape(0,2) # Empty array with correct shape
    else:
        tree_positions = None

    df = pd.read_csv(csv_path, skiprows=2)
    time_history = df["time"].values
    x_trajectory = df["x"].values
    y_trajectory = df["y"].values
    theta_trajectory = df["theta"].values
    entropy_history = df["entropy"].values

    lambda_columns = [col for col in df.columns if col.startswith("lambda_")]
    lambda_history = df[lambda_columns].values

    return time_history, x_trajectory, y_trajectory, theta_trajectory, entropy_history, lambda_history, tree_positions

def plot_trajectory_subplot(ax, x, y, theta, trees, lambda_history, custom_cmap,
                            tree_label_fontsize, legend_fontsize_param,
                            axis_label_fontsize, tick_label_fontsize,
                            subplot_title_fontsize,
                            mode_label_for_title, is_single_mode_plot):
    # Plot trajectory line
    ax.plot(x, y, marker='o', color='orange', linewidth=1, markersize=1.5, label="Drone Trajectory")
    ax.scatter(x[0], y[0], color='crimson', s=250, marker='X', label="Initial Position", zorder=4)
    ax.scatter(x[-1], y[-1], color='gold', s=250, marker='*', label="Final Position", zorder=4)

    # Plot trees
    tree_handles_for_legend = [] # Initialize here
    if trees is not None and trees.size > 0:
        num_trees = trees.shape[0]
        # This specific handle is for legend construction, generic representation
        tree_handles_for_legend = [
             Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                    markersize=10, label='Ripe(Red)-Raw(Green) Trees Belief')
        ]
        
        for i in range(num_trees):
            if lambda_history.size > 0 and lambda_history.shape[0] > 0 and lambda_history.shape[1] > i:
                final_lambda = lambda_history[-1, i]
            else:
                final_lambda = 0.5
            tree_color = custom_cmap(final_lambda)
            ax.scatter(trees[i, 0], trees[i, 1], color=tree_color, s=100, marker='o', zorder=3)


    if is_single_mode_plot:
        base_handles = [
            Line2D([0], [0], color='orange', lw=4, marker='o', markersize=5, label='Drone Trajectory'),
            Line2D([0], [0], marker='X', color='w', markerfacecolor='crimson', markersize=15, label='Initial Position'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markersize=20, label='Final Position'),
        ]
        all_handles = base_handles + tree_handles_for_legend
        ax.legend(handles=all_handles, loc='lower center',
                  bbox_to_anchor=(0.5, -0.35), # Adjusted for single plot better legend placement
                  fontsize=legend_fontsize_param, ncol=2, frameon=True) # Reduced ncol for better fit


    arrow_length = 1.5
    step = max(1, len(x) // 300) # Reduced number of arrows for clarity, e.g. max 20 arrows
    if len(x) > 1: # Ensure there's more than one point to draw arrows
        for idx_arr in range(0, len(x), step):
            x0, y0, t = x[idx_arr], y[idx_arr], theta[idx_arr]
            x1 = x0 + arrow_length * np.cos(t)
            y1 = y0 + arrow_length * np.sin(t)
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="orange", linewidth=1.0))

    #ax.set_xlabel("X(m)", fontsize=axis_label_fontsize)
    #ax.set_ylabel("Y(m)", fontsize=axis_label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_label_fontsize)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-4.0, 18.5)  # Replace -10, 10 with your desired limits
    ax.set_ylim(-20.5, +1.5)
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(5))

    if is_single_mode_plot:
        pass
    else:
        ax.set_title(mode_label_for_title, fontsize=subplot_title_fontsize)


def plot_entropy_subplot(ax, time_history, entropy,
                         axis_label_fontsize, tick_label_fontsize, subplot_title_fontsize,
                         mode_label_for_title,
                         is_single_mode_plot, legend_fontsize_param):
    ax.plot(time_history, entropy, marker='o', color='blue', linewidth=4, markersize=5, label="Entropy")
    ax.set_xlabel("Time (s)", fontsize=axis_label_fontsize)
    ax.set_ylabel("Entropy (bits)", fontsize=axis_label_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)

    if is_single_mode_plot:
        ax.set_title("Entropy Over Time", fontsize=subplot_title_fontsize)
        ax.legend(fontsize=legend_fontsize_param)
    else:
        ax.set_title(mode_label_for_title, fontsize=subplot_title_fontsize)
        ax.legend(fontsize=legend_fontsize_param * 0.8)

def main():
    # Test with 5 modes to check centering
    modes = ["mower_good", "mower_bad"] 
    #modes = ["mpc", "greedy", "linear"]
    # modes = ["mpc", "greedy"] # Test with 2 modes
    # modes = ["mpc"] # Test with 1 mode
    baselines_dir = "to_plot" 

    axis_label_fontsize = 15
    tick_label_fontsize = 12
    subplot_title_fontsize = 16
    legend_fontsize = 14
    suptitle_fontsize = 18
    tree_label_fontsize = 10

    # Verde (unripe) ? Giallo ? Rosso (ripe)
    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_cmap", [paired_palette[3], (0.5, 0.5, 0.5), dark_paired_palette[1]]
    )
    algorithm_labels = {
        "linear": "Linear Path",
        "greedy": "Greedy Approach",
        "mower" : "Mower Path", # General Mower
        "mower_good": "Mower Path - Good View",
        "mower_bad": "Mower Path - Poor View", # Changed from mower_poor
        "mpc": "Neural MPC",
    }

    num_modes_to_plot = len(modes)

    if num_modes_to_plot == 0:
        print("No modes specified to plot.")
        return

    if num_modes_to_plot == 1:
        mode = modes[0]
        csv_file = get_latest_csv(mode, baselines_dir)
        if not csv_file:
            print(f"No CSV found for {mode}")
            return

        try:
            (time_history, x, y, theta, entropy,
             lambda_history, trees) = load_csv_data(csv_file)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            return

        mode_desc = algorithm_labels.get(mode, mode.capitalize())

        fig_traj, ax_traj = plt.subplots(figsize=(8, 7)) # Adjusted figsize

        plot_trajectory_subplot(ax_traj, x, y, theta, trees, lambda_history, custom_cmap,
                                tree_label_fontsize, legend_fontsize,
                                axis_label_fontsize, tick_label_fontsize,
                                subplot_title_fontsize,
                                mode_desc, is_single_mode_plot=True)
        fig_traj.suptitle(f"{mode_desc} Trajectory", fontsize=suptitle_fontsize)
        fig_traj.tight_layout(rect=[0, 0.1, 1, 0.93]) # Adjust rect for suptitle and legend

        output_path_traj = os.path.join(baselines_dir, f"{mode}_trajectory.png")
        fig_traj.savefig(output_path_traj, format='png', bbox_inches='tight',  dpi=350)
        print(f"Saved trajectory plot for {mode} to: {output_path_traj}")

        fig_entropy, ax_entropy = plt.subplots(figsize=(10,6))
        plot_entropy_subplot(ax_entropy, time_history, entropy,
                             axis_label_fontsize, tick_label_fontsize, subplot_title_fontsize,
                             mode_desc, is_single_mode_plot=True, legend_fontsize_param=legend_fontsize)
        fig_entropy.suptitle(f"{mode_desc}: Entropy Trend", fontsize=suptitle_fontsize)
        fig_entropy.tight_layout(rect=[0, 0.03, 1, 0.93])

        output_path_entropy = os.path.join(baselines_dir, f"{mode}_entropy.eps",  dpi=350)
        fig_entropy.savefig(output_path_entropy, format='eps', bbox_inches='tight')
        print(f"Saved entropy plot for {mode} to: {output_path_entropy}")

        plt.show()

    elif num_modes_to_plot > 1:
        # For Trajectory Plots, use a 2x3 grid (max 6 plots)
        # If num_modes is 5, the 2nd row should have its 2 plots centered.
        # If num_modes is 4, the 2nd row has 1 plot on the left.
        # If num_modes <=3, only the 1st row is used.
        num_rows_traj = 1
        num_cols_traj = 2
        fig_trajectories, axes_trajectories = plt.subplots(num_rows_traj, num_cols_traj, figsize=(11, 6), squeeze=False)
        axes_trajectories_flat = axes_trajectories.flatten()

        fig_entropy_combined, ax_entropy_combined = plt.subplots(figsize=(12, 7))

        num_colors_needed = num_modes_to_plot
        # Using a more diverse colormap if many modes
        if num_colors_needed <= 10:
            color_map = plt.cm.get_cmap('tab10', num_colors_needed)
        elif num_colors_needed <=20:
            color_map = plt.cm.get_cmap('tab20', num_colors_needed)
        else: # Fallback for more than 20
            color_map_base = plt.cm.get_cmap('nipy_spectral', num_colors_needed) # More distinct colors
        colors_for_modes = [color_map(i) for i in range(num_colors_needed)]


        active_plot_count_traj = 0
        any_mode_had_trees = False
        
        # Keep track of which axes in the grid are used
        used_axes_indices = []

        for idx, mode in enumerate(modes):
            mode_desc = algorithm_labels.get(mode, mode.capitalize())
            
            # --- Determine target axis for trajectory plot ---
            target_axis_flat_idx = idx # Default: 0th mode -> 0th subplot, 1st -> 1st, etc.

            if num_modes_to_plot == 5: # Special centering for 5 plots
                if idx == 3: # The 4th mode (0-indexed)
                    target_axis_flat_idx = 4 # Moves to the middle slot of the second row (original index for 5th subplot)
                elif idx == 4: # The 5th mode
                    target_axis_flat_idx = 5 # Moves to the right slot of the second row (original index for 6th subplot)
            
            # If num_modes_to_plot is 4, modes[3] will go to axes_flat[3] (bottom-left) - this is fine.
            # If num_modes_to_plot <= 3, they fill the first row.
            
            ax_traj_current = None
            if target_axis_flat_idx < len(axes_trajectories_flat): # Ensure it's within 2x3 grid
                 ax_traj_current = axes_trajectories_flat[target_axis_flat_idx]
                 used_axes_indices.append(target_axis_flat_idx)
            else: # Should not happen if modes <= 6
                print(f"Warning: Skipping trajectory plot for mode '{mode}' as it exceeds grid capacity.")


            csv_file = get_latest_csv(mode, baselines_dir)
            if not csv_file:
                print(f"No CSV found for {mode}")
                if ax_traj_current:
                    ax_traj_current.text(0.5, 0.5, f"No data for\n{mode_desc}",
                                         ha='center', va='center', fontsize=12)
                    ax_traj_current.axis('off')
                continue

            try:
                (time_h, x_t, y_t, theta_t, entropy_val,
                 lambda_h, trees_pos) = load_csv_data(csv_file)
                if trees_pos is not None and trees_pos.size > 0:
                    any_mode_had_trees = True
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
                if ax_traj_current:
                    ax_traj_current.text(0.5, 0.5, f"Error loading\n{mode_desc}",
                                         ha='center', va='center', fontsize=12)
                    ax_traj_current.axis('off')
                continue

            if ax_traj_current:
                plot_trajectory_subplot(ax_traj_current, x_t, y_t, theta_t, trees_pos, lambda_h, custom_cmap,
                                        tree_label_fontsize, legend_fontsize,
                                        axis_label_fontsize, tick_label_fontsize,
                                        subplot_title_fontsize,
                                        mode_desc, is_single_mode_plot=False)
                active_plot_count_traj += 1
            oim = OffsetImage(logo_arr, zoom=0.075)  # tweak `zoom` so it?s the right size
            # 0.05, 0.95 are in Axes?fraction coordinates (5% from left, 95% from bottom),
            # which places it near the top?left corner?adjust as needed.
            ab = AnnotationBbox(
                oim,
                (-0.01, 1.01),
                xycoords="axes fraction",
                frameon=False
            )
            ax_traj_current.add_artist(ab)
            ax_entropy_combined.plot(time_h, entropy_val, marker='o', color=colors_for_modes[idx],
                                     linewidth=1.5, markersize=3, label=mode_desc) # Slightly thicker lines

        # Finalize Trajectory Plots
        if active_plot_count_traj > 0:
            legend_handles = [
                Line2D([0], [0], color='orange', lw=1.5, marker='o', markersize=2, label='Drone Trajectory'),
                Line2D([0], [0], marker='X', color='w', markerfacecolor='crimson', markersize=15, label='Initial Position'),
                Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markersize=25, label='Final Position'),
            ]
            if any_mode_had_trees:
                legend_handles.append(
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                           markersize=8, label='Ripe(Red)-Raw(Green) Trees Belief')
                )
            
            fig_trajectories.suptitle("Algorithm Trajectories Comparison", fontsize=suptitle_fontsize, y=0.98) # Adjust y for suptitle

            # Delete all unused axes from the 2x3 grid
            for i_ax, ax_curr in enumerate(axes_trajectories_flat):
                if i_ax not in used_axes_indices:
                    fig_trajectories.delaxes(ax_curr)
            
            #fig_trajectories.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust rect for suptitle and fig legend
            fig_trajectories.legend(handles=legend_handles, loc='lower center',
                                    bbox_to_anchor=(0.5, 0.01), # Adjust y to be just above bottom
                                    ncol=len(legend_handles), fontsize=legend_fontsize, frameon=False)
            output_path_trajectories = os.path.join(baselines_dir, "comparison_trajectories.png")
            fig_trajectories.savefig(output_path_trajectories, format='png', dpi=350)
            print(f"Saved trajectories comparison plot to: {output_path_trajectories}")
        else:
            plt.close(fig_trajectories)

        # Finalize Combined Entropy Plot
        if num_modes_to_plot > 0 and any(ax_entropy_combined.lines): # Check if anything was plotted
            ax_entropy_combined.set_xlabel("Time (s)", fontsize=axis_label_fontsize)
            ax_entropy_combined.set_ylabel("Entropy (bits)", fontsize=axis_label_fontsize)
            ax_entropy_combined.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
            ax_entropy_combined.legend(fontsize=legend_fontsize, loc='best')
            ax_entropy_combined.grid(True, linestyle='--', alpha=0.7)

            fig_entropy_combined.suptitle("Entropy Trends Comparison", fontsize=suptitle_fontsize)
            fig_entropy_combined.tight_layout(rect=[0, 0.03, 1, 0.95])
            output_path_entropies = os.path.join(baselines_dir, "comparison_entropies.eps")
            fig_entropy_combined.savefig(output_path_entropies, format='eps', bbox_inches='tight')
            print(f"Saved combined entropy plot to: {output_path_entropies}")
        else:
            plt.close(fig_entropy_combined)

        if active_plot_count_traj > 0 or (num_modes_to_plot > 0 and any(ax_entropy_combined.lines)):
            plt.show()

if __name__ == "__main__":
    main()