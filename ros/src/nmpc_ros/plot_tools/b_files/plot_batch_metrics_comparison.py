#!/usr/bin/env python
import os
import glob
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.style.use('seaborn-v0_8')

def load_performance_metrics(file_path):
    metrics = {}
    try:
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    key = row[0].strip()
                    try:
                        value_str = row[1].strip().strip('"').strip("'")
                        value = float(value_str)
                        metrics[key] = value
                    except ValueError:
                        continue
    except FileNotFoundError:
        return None
    except Exception as e:
        return None
    required_keys = [
        "Total Execution Time (s)", "Total Distance (m)",
        "Average Waypoint-to-Waypoint Time (s)", "Final Entropy",
        "Total Commands"
    ]
    for rkey in required_keys:
        if rkey not in metrics:
             pass
    return metrics

def load_average_velocity(vel_file_path):
    try:
        df = pd.read_csv(vel_file_path)
        if df.empty or 'x_velocity' not in df.columns or 'y_velocity' not in df.columns:
            pass
        df['x_velocity'] = pd.to_numeric(df['x_velocity'], errors='coerce')
        df['y_velocity'] = pd.to_numeric(df['y_velocity'], errors='coerce')
        df.dropna(subset=['x_velocity', 'y_velocity'], inplace=True)
        if df.empty:
            return np.nan
        df['Linear Velocity'] = np.sqrt(df['x_velocity']**2 + df['y_velocity']**2)
        df.dropna(subset=['Linear Velocity'], inplace=True)
        if df.empty:
             return np.nan
        return df['Linear Velocity'].median()
    except FileNotFoundError:
        return np.nan
    except Exception as e:
        return np.nan

def gather_metrics(base_dir):
    run_folders = sorted(glob.glob(os.path.join(base_dir, "run_*")))
    if not run_folders:
        pass
    metrics_lists = {
        "Total Execution Time (s)": [],
        "Final Entropy": [],
        "Total Distance (m)": [],
        "Average Velocity (m/s)": [],
        "Average NMPC Step Execution Time (s)": []
    }
    tests_found_metrics = 0
    dt_stats_list = []
    for run_idx, run in enumerate(run_folders):
        perf_files = glob.glob(os.path.join(run, "*performance_metrics.csv"))
        if not perf_files:
            continue
        perf_file = perf_files[0]
        perf = load_performance_metrics(perf_file)
        if perf is None:
            continue
        tests_found_metrics += 1
        metrics_lists["Total Execution Time (s)"].append(perf.get("Total Execution Time (s)", np.nan))
        metrics_lists["Final Entropy"].append(perf.get("Final Entropy", np.nan))
        metrics_lists["Total Distance (m)"].append(perf.get("Total Distance (m)", np.nan))
        metrics_lists["Average NMPC Step Execution Time (s)"].append(perf.get("Average Waypoint-to-Waypoint Time (s)", np.nan))
        vel_files = glob.glob(os.path.join(run, "*_velocity_commands.csv"))
        if not vel_files:
            metrics_lists["Average Velocity (m/s)"].append(np.nan)
            metrics_lists["Average Velocity (m/s)"].append(np.nan)
            dt_stats_list.append({
                'run_folder': os.path.basename(run),
                'dt_std': np.nan,
                'dt_min': np.nan,
                'dt_max': np.nan
            })
        else:
            vel_file = vel_files[0]
            avg_vel = load_average_velocity(vel_file)
            metrics_lists["Average Velocity (m/s)"].append(avg_vel)
            dt_std, dt_min, dt_max, dt_mean, dt_median = compute_dt_stats(vel_file)
            dt_stats_list.append({
                'run_folder': os.path.basename(run),
                'dt_std': dt_std,
                'dt_min': dt_min,
                'dt_max': dt_max,
                'dt_mean': dt_mean,
                'dt_median': dt_median
            })
    if tests_found_metrics > 0 :
        print(f"Processed {base_dir}: Found metrics in {tests_found_metrics}/{len(run_folders)} runs.")
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name  = f"dt_stats_{timestamp}_{base_dir}.csv"
    pd.DataFrame(dt_stats_list).to_csv(out_name, index=False)
    print(f"Written ?t stats per run to: {out_name}")
    return metrics_lists, tests_found_metrics

def compute_statistics(values):
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    if arr.size < 4:
         filtered_arr = arr
    else:
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1
        if iqr == 0:
            lower_bound = q1 - 1e-9
            upper_bound = q3 + 1e-9
        else:
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
        filtered_arr = arr[(arr >= lower_bound) & (arr <= upper_bound)]
    if filtered_arr.size == 0:
         return np.nan, np.nan, np.nan, np.nan
    median_val = np.median(filtered_arr)
    std_val = np.std(filtered_arr)
    min_val = np.min(filtered_arr)
    max_val = np.max(filtered_arr)
    return np.mean(filtered_arr), median_val, std_val, min_val, max_val

def compute_dt_stats(vel_file_path):
    try:
        df = pd.read_csv(vel_file_path)
    except FileNotFoundError:
        return np.nan, np.nan, np.nan
    if df.shape[0] < 2 or 'Time (s)' not in df.columns:
        return np.nan, np.nan, np.nan
    t = df['Time (s)'].iloc[1:].values
    dt = np.diff(t)
    dt = dt[dt > 0]
    if dt.size == 0:
        return np.nan, np.nan, np.nan
    return dt.std(), dt.min(), dt.max(), dt.mean(), np.median(dt)


def run_batch_analysis(base_dirs_dict):
    """ Analyzes multiple algorithms, generates comparison plots, exports summary. """
    all_metrics = {}
    test_counts = {}

    print("\n--- Starting Batch Analysis ---")
    for algo_name, base_dir in base_dirs_dict.items():
        print(f"\n--- Processing Algorithm: {algo_name} (Source: {base_dir}) ---")
        if not os.path.isdir(base_dir):
            print(f"Warning: Directory '{base_dir}' not found. Skipping '{algo_name}'.")
            continue
        metrics_data, num_tests = gather_metrics(base_dir)
        if num_tests == 0:
            print(f"Warning: No valid test data found for '{algo_name}'. Excluding from comparison.")
            continue

        all_metrics[algo_name] = metrics_data
        test_counts[algo_name] = num_tests
        # Optional: Run single plot generation here if desired for each algo
        # run_single_analysis(algo_name, base_dir) # Calls plot_grouped_metrics

    if not all_metrics:
        print("\nError: No algorithms processed successfully with data. Exiting batch analysis.")
        return

    print("\n--- Generating Comparison Plots & Summary Table ---")
    plot_comparison_metrics(all_metrics, test_counts)
    export_summary_table(all_metrics, test_counts)
    print("\n--- Batch Analysis Complete ---")


def plot_comparison_metrics(all_metrics, test_counts):
    """ Plots grouped bar charts comparing algorithms, with bar clipping and fixed legend. """
    group1_metrics = ["Total Execution Time (s)", "Total Distance (m)"]
    group2_metrics = ["Final Entropy", "Average Velocity (m/s)"]

    algorithms = list(all_metrics.keys())
    num_algorithms = len(algorithms)
    if num_algorithms == 0:
        print("No algorithms have data for comparison plotting.")
        return

    # Determine bar width dynamically
    total_width = 0.8 # Total width allocated for bars within a group
    bar_width = total_width / num_algorithms
    cmap = plt.get_cmap("tab10")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7)) # Slightly wider

    legend_handles = [] # To store one handle per algorithm for the legend
    legend_labels = []

    # --- Helper function to plot one group ---
    def plot_group(ax, group_metrics, title, set_y_limit=None, is_first_plot=False):
        num_metrics = len(group_metrics)
        x_indices = np.arange(num_metrics)
        clip_threshold = set_y_limit if set_y_limit is not None else np.inf # Use limit for clipping if set

        all_plot_upper_bounds = [] # Track y-values for axis limits

        for i, algo in enumerate(algorithms):
            algo_medians = []
            algo_stds = []
            algo_raw_values = []

            for metric in group_metrics:
                 values = all_metrics.get(algo, {}).get(metric, [])
                 algo_raw_values.append(values)
                 median, std, _, _ = compute_statistics(values) # Get the TRUE median/std
                 algo_medians.append(median)
                 algo_stds.append(std)

            plot_medians = np.array(algo_medians, dtype=float)
            plot_stds = np.array(algo_stds, dtype=float)
            plot_stds_safe = np.nan_to_num(plot_stds) # Use 0 std for NaN error bars

            # Calculate positions
            offset = (i - num_algorithms / 2 + 0.5) * bar_width # Center group around tick
            bar_positions = x_indices + offset

            # --- Bar Clipping Logic ---
            plot_heights = plot_medians.copy() # Start with actual medians
            clipped_mask = plot_heights > clip_threshold # Find bars exceeding threshold
            plot_heights[clipped_mask] = clip_threshold # Cap their height

            bars = ax.bar(bar_positions, plot_heights, width=bar_width * 0.9, # Slightly thinner bars
                          label=f"{algo} (n={test_counts.get(algo, 0)})", # Label needed for initial handle collection
                          yerr=plot_stds_safe,
                          capsize=4,
                          alpha=0.9, color=cmap(i % cmap.N),
                          error_kw={'alpha': 0.6}) # Make error bars slightly transparent

            # Store legend info only from the first plot group
            if is_first_plot:
                if bars: # Ensure bars were actually created (data wasn't all NaN)
                    legend_handles.append(bars[0]) # Get handle from first bar of this algo
                    legend_labels.append(f"{algo} (n={test_counts.get(algo, 0)})")

            # Add hatching and text for clipped bars
            for j, bar in enumerate(bars):
                original_median = plot_medians[j] # The actual calculated median
                is_clipped = clipped_mask[j]

                if np.isnan(original_median): # Skip annotation for NaN bars
                    continue

                # Apply hatching if clipped
                if is_clipped:
                    bar.set_hatch('///')
                    bar.set_edgecolor('grey') # Make edge visible over hatch

                # --- Text Annotation ---
                text_y = plot_heights[j] # Start text position at (potentially clipped) bar top
                # Add error bar height to text position if std exists
                if not np.isnan(plot_stds[j]) and plot_stds[j] > 0:
                    # Make sure not to position based on error bar if the bar itself was clipped much lower
                    # If clipped, put text just above clip line. Otherwise, above error bar.
                     if is_clipped:
                         text_y = clip_threshold # Position text right above the clip line
                     else:
                         text_y += plot_stds_safe[j] # Position above error bar end

                # Add a small absolute offset based on current axis scale for clarity
                y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
                text_y += y_offset

                # Use the ORIGINAL median for the text label!
                ax.text(bar.get_x() + bar.get_width() / 2., text_y,
                        f"{original_median:.2f}", ha='center', va='bottom', fontsize=7, rotation=0, fontweight='bold')


            # Track max values for y-limit setting (use capped height + error bar for visible extent)
            valid_bounds = [h + s for h, s in zip(plot_heights, plot_stds_safe) if not np.isnan(h)]
            if valid_bounds:
                all_plot_upper_bounds.extend(valid_bounds)


        ax.set_xticks(x_indices)
        ax.set_xticklabels(group_metrics, rotation=30, ha='right')
        ax.set_ylabel("Median Value")
        ax.set_title(title)
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        if set_y_limit is not None:
            ax.set_ylim(0, set_y_limit * 1.1)
        elif all_plot_upper_bounds:
             y_upper = max(all_plot_upper_bounds) * 1.15
             ax.set_ylim(0, y_upper if not np.isnan(y_upper) and y_upper > 0 else 1)
        else:
             ax.set_ylim(0, 1)

    plot_group(ax1, group1_metrics, "Execution Time & Distance", is_first_plot=True)
    plot_group(ax2, group2_metrics, "Entropy & Velocity", set_y_limit=4.0)

    fig.suptitle("Performance Comparison Across Algorithms", fontsize=18)

    if legend_handles:
         fig.legend(handles=legend_handles, labels=legend_labels,
                   loc='upper center', ncol=min(num_algorithms, 4),
                   bbox_to_anchor=(0.5, 0.96), fontsize='medium')

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])

    plot_filename = "comparison_performance_summary_clipped.png"
    try:
        plt.savefig(plot_filename)
        print(f"\nSaved comparison plot: {plot_filename}")
    except Exception as e:
        print(f"\nError saving comparison plot {plot_filename}: {e}")

    plt.show()
    plt.close(fig)

def export_summary_table(all_metrics, test_counts, output_path="summary_metrics_comparison.csv"):
    """ Creates a CSV summary table with median, std, min, max. """
    rows = []
    algorithms = list(all_metrics.keys())
    if not algorithms:
        print("Cannot export summary table: No algorithm data.")
        return

    all_metric_keys = set()
    for metrics in all_metrics.values():
        all_metric_keys.update(metrics.keys())
    sorted_metric_keys = sorted(list(all_metric_keys))

    for algo in algorithms:
        metrics = all_metrics.get(algo, {})
        t_count = test_counts.get(algo, 0)
        for metric in sorted_metric_keys:
            values = metrics.get(metric, [])
            median, std, min_val, max_val = compute_statistics(values)
            rows.append({
                "Metric": metric,
                "Algorithm": algo,
                "Median": round(median, 4) if not np.isnan(median) else 'N/A',
                "Std Dev": round(std, 4) if not np.isnan(std) else 'N/A',
                "Min": round(min_val, 4) if not np.isnan(min_val) else 'N/A',
                "Max": round(max_val, 4) if not np.isnan(max_val) else 'N/A',
                "Test Count": t_count
            })

    if not rows:
         print("No data rows generated for the summary table.")
         return

    df = pd.DataFrame(rows)
    df = df[["Metric", "Algorithm", "Median", "Std Dev", "Min", "Max", "Test Count"]]
    df = df.sort_values(by=["Metric", "Algorithm"])

    try:
        df.to_csv(output_path, index=False)
        print(f"\n✅ Summary table saved: {output_path}")
        print("\n--- Summary Statistics Table ---")
        print(df.to_string(index=False, na_rep='N/A', float_format="%.4f"))
    except Exception as e:
        print(f"\n❌ Error saving summary table to {output_path}: {e}")

if __name__ == "__main__":
    base_dirs_to_analyze = {
        'neural mpc 100 elements': 'test_hz_nmpc',
        'neural mpc 10 elements': 'test_hz_10_nmpc',
        'neural mpc 625 elemnts': 'test_hz_625_nmpc',
        'neural mpc 625 elemnts with 10 elements': 'test_hz_625_10_nmpc'
    }
    run_batch_analysis(base_dirs_to_analyze)
