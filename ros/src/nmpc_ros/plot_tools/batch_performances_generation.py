#!/usr/bin/env python
import os
import glob
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker  # For formatting y-axis if needed

# ===================== Configuration =====================
# Distance is computed after resampling time,x,y to a uniform grid:
DIST_SAMPLE_DT = 1.0            # seconds, e.g. 1.0 for 1 Hz
DIST_SKIP_FIRST_SECONDS = 0.0    # e.g. 20.0 to ignore first 20s warm-up
# =========================================================

# Make plots look a bit nicer
plt.style.use('seaborn-v0_8')

def load_performance_metrics(file_path):
    """Loads performance metrics from a CSV file, tentando la conversione numerica."""
    metrics = {}
    try:
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    key = row[0].strip()
                    try:
                        value_str = row[1].strip().strip('"').strip("'")
                        try:
                            value = float(value_str)
                        except ValueError:
                            value = value_str
                        metrics[key] = value
                    except ValueError:
                        print(f"Warning: Could not parse value '{row[1]}' for '{key}' in {file_path}. Skipping row.")
                        continue
    except FileNotFoundError:
        print(f"Error: Performance metrics file not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading performance metrics file {file_path}: {e}")
        return None

    # required keys (if missing, leave absent)
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
    """Loads velocity CSV and computes median linear velocity."""
    try:
        df = pd.read_csv(vel_file_path)
        if df.empty or 'x_velocity' not in df.columns or 'y_velocity' not in df.columns:
            return np.nan

        df['x_velocity'] = pd.to_numeric(df['x_velocity'], errors='coerce')
        df['y_velocity'] = pd.to_numeric(df['y_velocity'], errors='coerce')
        df.dropna(subset=['x_velocity', 'y_velocity'], inplace=True)

        if df.empty:
            print(f"Warning: Velocity file {vel_file_path} contains no valid numeric x/y velocity data.")
            return np.nan

        df['Linear Velocity'] = np.sqrt(df['x_velocity']**2 + df['y_velocity']**2)
        df.dropna(subset=['Linear Velocity'], inplace=True)
        if df.empty:
            print(f"Warning: No valid linear velocities computed for {vel_file_path}.")
            return np.nan

        return float(df['Linear Velocity'].median())
    except FileNotFoundError:
        return np.nan
    except Exception as e:
        print(f"Error loading or processing velocity file {vel_file_path}: {e}")
        return np.nan

def compute_distance_from_plot_data_resampled(plot_data_file_path, dt=DIST_SAMPLE_DT, skip_first_seconds=DIST_SKIP_FIRST_SECONDS):
    """
    Compute total path length from *_plot_data.csv by:
      1) parsing time, x, y
      2) optionally skipping the first `skip_first_seconds`
      3) resampling at uniform dt seconds via linear interpolation (np.interp)
      4) summing Euclidean step lengths between consecutive resampled points

    Returns: float (meters) or np.nan
    """
    try:
        with open(plot_data_file_path, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        # Find header
        header_idx = -1
        for i, ln in enumerate(lines):
            if ln.startswith("time,x,y,theta"):
                header_idx = i
                break
        if header_idx == -1:
            print(f"Warning: header 'time,x,y,theta,...' not found in {plot_data_file_path}")
            return np.nan

        header = lines[header_idx].split(",")
        try:
            ix_t = header.index("time")
            ix_x = header.index("x")
            ix_y = header.index("y")
        except ValueError:
            print(f"Warning: missing time/x/y columns in header of {plot_data_file_path}")
            return np.nan

        # Parse numeric rows
        t_list, x_list, y_list = [], [], []
        for ln in lines[header_idx + 1:]:
            parts = ln.split(",")
            if len(parts) <= max(ix_t, ix_x, ix_y):
                continue
            try:
                t_list.append(float(parts[ix_t]))
                x_list.append(float(parts[ix_x]))
                y_list.append(float(parts[ix_y]))
            except ValueError:
                continue

        if len(t_list) < 2:
            return np.nan

        t = np.asarray(t_list, dtype=float)
        x = np.asarray(x_list, dtype=float)
        y = np.asarray(y_list, dtype=float)

        # Ensure strictly increasing time (deduplicate)
        t, uniq_idx = np.unique(t, return_index=True)
        x = x[uniq_idx]
        y = y[uniq_idx]
        if t.size < 2:
            return np.nan

        # Optional skip of the initial window
        t0 = t[0]
        if skip_first_seconds > 0:
            start_time = t0 + float(skip_first_seconds)
            if start_time >= t[-1]:
                return np.nan
        else:
            start_time = t0

        # Uniform time grid at dt seconds
        t_new = np.arange(start_time, t[-1] + 1e-9, dt)
        if t_new.size < 2:
            return np.nan

        # Linear interpolation (numpy only)
        x_new = np.interp(t_new, t, x)
        y_new = np.interp(t_new, t, y)

        dx = np.diff(x_new)
        dy = np.diff(y_new)
        steps = np.hypot(dx, dy)

        return float(np.nansum(steps))

    except FileNotFoundError:
        return np.nan
    except Exception as e:
        print(f"Error computing resampled distance from {plot_data_file_path}: {e}")
        return np.nan

def load_false_positives(plot_data_file_path, num_trees):
    """
    Carica plot_data.csv, estrae GT e lambdas finali e calcola:
      - False Positives (%), Not Detected (%), Percentage of Classified Trees (%),
        Percentage of Correct Classifications on Classified Trees (%)
    Regola di decisione:
      - corretto: |gt - lambda| < 0.1
      - sbagliato netto (considerato 'false positive'): |gt - lambda| > 0.8
      - non classificato (zona grigia): altrimenti
    """
    gt_ids = []
    lambdas_final = []

    try:
        with open(plot_data_file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        if len(lines) < 4:
            print(f"Warning: Plot data file {plot_data_file_path} has too few lines ({len(lines)}).")
            return np.nan

        # Parse trees_gt_id
        gt_line_found = False
        for line_idx, line_content in enumerate(lines):
            if line_content.startswith("trees_gt_id,"):
                gt_line_parts = line_content.split(',')
                if len(gt_line_parts) > 1:
                    gt_values_str = gt_line_parts[1:]
                    if len(gt_values_str) == num_trees:
                        try:
                            gt_ids = [int(val) for val in gt_values_str]
                        except ValueError:
                            print(f"Warning: Non-integer GT values in {plot_data_file_path}")
                            return np.nan
                        gt_line_found = True
                        break
                    else:
                        print(f"Warning: Expected {num_trees} GT IDs, found {len(gt_values_str)} in {plot_data_file_path}")
                        return np.nan
                else:
                    print(f"Warning: 'trees_gt_id' line malformed (no values) in {plot_data_file_path}")
                    return np.nan

        if not gt_line_found:
            print(f"Warning: 'trees_gt_id,' line not found in {plot_data_file_path}")
            return np.nan

        # Timeseries header
        header_line_index = -1
        header_parts = []
        for i, line_content in enumerate(lines):
            if "time,x,y,theta,entropy,lambda_0" in line_content:
                header_line_index = i
                header_parts = line_content.split(',')
                break

        if header_line_index == -1:
            print(f"Warning: Timeseries header (time,x,y,...) not found in {plot_data_file_path}")
            return np.nan

        # Indices of lambda_i
        lambda_indices = []
        try:
            for i in range(num_trees):
                lambda_col_name = f"lambda_{i}"
                lambda_indices.append(header_parts.index(lambda_col_name))
        except ValueError as e:
            print(f"Warning: Column {e} (one of lambda_i) not found in header of {plot_data_file_path}")
            return np.nan

        # Last non-empty data row
        data_lines_content = [line for line in lines[header_line_index+1:] if line]
        if not data_lines_content:
            print(f"Warning: No data rows found after header in {plot_data_file_path}")
            return np.nan

        last_data_line_str = data_lines_content[-1]
        last_data_parts = last_data_line_str.split(',')

        if len(last_data_parts) < max(lambda_indices) + 1:
            print(f"Warning: Last data line in {plot_data_file_path} has {len(last_data_parts)} columns, "
                  f"expected at least {max(lambda_indices) + 1}.")
            return np.nan

        lambdas_final_str = [last_data_parts[idx] for idx in lambda_indices]
        try:
            lambdas_final = [float(val) for val in lambdas_final_str]
        except ValueError as e:
            print(f"Warning: Could not parse lambda values as float in {plot_data_file_path}: {e}")
            return np.nan

        if len(lambdas_final) != num_trees:
            print(f"Warning: Extracted {len(lambdas_final)} lambda values, expected {num_trees} in {plot_data_file_path}")
            return np.nan

        predicted_tree_states = [l for l in lambdas_final]

        false_positives = 0
        not_detected = 0
        correct_classifications = 0
        classified_trees = 0

        if len(gt_ids) == len(predicted_tree_states):
            for i in range(len(gt_ids)):
                lambda_val = predicted_tree_states[i]
                gt = gt_ids[i]

                if np.abs(gt - lambda_val) < 0.1:
                    correct_classifications += 1
                    classified_trees += 1
                elif np.abs(gt - lambda_val) > 0.8:
                    false_positives += 1
                    classified_trees += 1
                else:
                    not_detected += 1
        else:
            print(f"Critical Warning: Mismatch GT IDs ({len(gt_ids)}) and predictions ({len(predicted_tree_states)}) in {plot_data_file_path}")
            return np.nan

        total = len(gt_ids) if len(gt_ids) > 0 else np.nan
        if not np.isnan(total):
            false_positives_perc = (false_positives / total) * 100
            not_detected_perc = (not_detected / total) * 100
            perc_classified = (classified_trees / total) * 100
            perc_correct_on_classified = (correct_classifications / classified_trees) * 100 if classified_trees > 0 else 0.0
        else:
            false_positives_perc = not_detected_perc = perc_classified = perc_correct_on_classified = np.nan

        return false_positives_perc, not_detected_perc, perc_classified, perc_correct_on_classified

    except FileNotFoundError:
        return np.nan
    except Exception as e:
        print(f"Error processing plot data file {plot_data_file_path} for FPs: {e}")
        import traceback
        traceback.print_exc()
        return np.nan

def compute_statistics(values):
    """Removes NaNs/outliers (IQR), computes median, std, min, max."""
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
        print(f"Warning: Outlier filtering removed all {arr.size} data points. Returning NaN stats. Original data (first 5): {arr[:5]}")
        return np.nan, np.nan, np.nan, np.nan

    median_val = np.median(filtered_arr)
    std_val = np.std(filtered_arr)
    min_val = np.min(filtered_arr)
    max_val = np.max(filtered_arr)
    return median_val, std_val, min_val, max_val

def plot_grouped_metrics(metrics_data, num_tests, algo_name="Algorithm"):
    """Creates a figure with two subplots for a SINGLE algorithm's metrics."""
    group1_keys = ["Average Velocity (m/s)", "Final Entropy",
                   "Average NMPC Step Execution Time (s)", "False Positives (%)"]
    group2_keys = ["Total Execution Time (s)", "Total Distance (m)"]

    def get_stats(keys):
        medians, stds, mins, maxs = [], [], []
        for key in keys:
            if key in metrics_data and metrics_data[key]:
                median, std, min_val, max_val = compute_statistics(metrics_data[key])
            else:
                median, std, min_val, max_val = np.nan, np.nan, np.nan, np.nan
            medians.append(median)
            stds.append(std)
            mins.append(min_val)
            maxs.append(max_val)
        return medians, stds, mins, maxs

    medians1, stds1, mins1, maxs1 = get_stats(group1_keys)
    medians2, stds2, mins2, maxs2 = get_stats(group2_keys)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7))
    cmap1 = plt.get_cmap('Set2')
    cmap2 = plt.get_cmap('Set3')

    # Group 1
    x1 = np.arange(len(group1_keys))
    plot_medians1 = np.array(medians1, dtype=float)
    plot_stds1 = np.array(stds1, dtype=float)
    plot_stds1_safe = np.nan_to_num(plot_stds1)

    bars1 = ax1.bar(x1, plot_medians1, yerr=plot_stds1_safe, align='center', alpha=0.85,
                    color=[cmap1(i % cmap1.N) for i in range(len(group1_keys))],
                    error_kw={'ecolor': 'black', 'capsize': 8})
    ax1.set_xticks(x1)
    ax1.set_xticklabels(group1_keys, rotation=40, ha="right")
    ax1.set_ylabel("Median Value")
    ax1.set_title("Group 1: Velocity, Entropy, NMPC Time, False Positives (%)")
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    valid_upper_bounds1 = [m + s for m, s in zip(plot_medians1, plot_stds1_safe) if not np.isnan(m)]
    ax1.set_ylim(0, max(valid_upper_bounds1) * 1.25 if valid_upper_bounds1 else 1)

    for i, bar in enumerate(bars1):
        height = bar.get_height()
        median_val = plot_medians1[i]
        if not np.isnan(median_val):
            text_y_pos = height + (plot_stds1_safe[i] * 1.05)
            ax1.text(bar.get_x() + bar.get_width()/2., text_y_pos,
                     f"{median_val:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height / 2,
                         f"Min: {mins1[i]:.2f}\nMax: {maxs1[i]:.2f}",
                         ha='center', va='center', fontsize=7, alpha=0.8)

    # Group 2
    x2 = np.arange(len(group2_keys))
    plot_medians2 = np.array(medians2, dtype=float)
    plot_stds2 = np.array(stds2, dtype=float)
    plot_stds2_safe = np.nan_to_num(plot_stds2)

    bars2 = ax2.bar(x2, plot_medians2, yerr=plot_stds2_safe, align='center', alpha=0.85,
                    color=[cmap2(i % cmap2.N) for i in range(len(group2_keys))],
                    error_kw={'ecolor': 'black', 'capsize': 8})
    ax2.set_xticks(x2)
    ax2.set_xticklabels(group2_keys, rotation=40, ha="right")
    ax2.set_ylabel("Median Value")
    ax2.set_title("Group 2: Total Time & Total Distance")
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    valid_upper_bounds2 = [m + s for m, s in zip(plot_medians2, plot_stds2_safe) if not np.isnan(m)]
    ax2.set_ylim(0, max(valid_upper_bounds2) * 1.25 if valid_upper_bounds2 else 1)

    for i, bar in enumerate(bars2):
        height = bar.get_height()
        median_val = plot_medians2[i]
        if not np.isnan(median_val):
            text_y_pos = height + (plot_stds2_safe[i] * 1.05)
            ax2.text(bar.get_x() + bar.get_width()/2., text_y_pos,
                     f"{median_val:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height / 2,
                         f"Min: {mins2[i]:.2f}\nMax: {maxs2[i]:.2f}",
                         ha='center', va='center', fontsize=7, alpha=0.8)

    fig.suptitle(f"Performance Statistics for {algo_name} (Tests: {num_tests})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = f"performance_summary_{algo_name}.png"
    try:
        plt.savefig(plot_filename)
        print(f"Saved single algorithm plot: {plot_filename}")
    except Exception as e:
        print(f"Error saving plot {plot_filename}: {e}")
    plt.close(fig)

def run_single_analysis(algo_name, base_dir):
    """Helper function to run analysis for one algorithm and plot it"""
    print(f"\n--- Running Single Analysis for {algo_name} ---")
    metrics_data, num_tests = gather_metrics(base_dir)
    if num_tests > 0:
        plot_grouped_metrics(metrics_data, num_tests, algo_name)
    else:
        print(f"Skipping plot for {algo_name}: No valid test data.")
    return metrics_data, num_tests

def plot_comparison_metrics(all_metrics, test_counts):
    """Plots grouped bar charts comparing algorithms, with bar clipping and fixed legend."""
    group1_metrics = ["Total Execution Time (s)", "Total Distance (m)"]
    group2_metrics = ["Final Entropy", "Average Velocity (m/s)", "False Positives (%)"]

    algorithms = list(all_metrics.keys())
    num_algorithms = len(algorithms)
    if num_algorithms == 0:
        print("No algorithms have data for comparison plotting.")
        return

    total_width = 0.8
    bar_width = total_width / num_algorithms
    cmap = plt.get_cmap("tab10")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7))

    legend_handles = []
    legend_labels = []

    def plot_group(ax, group_metrics, title, set_y_limit=None, is_first_plot=False):
        num_metrics = len(group_metrics)
        x_indices = np.arange(num_metrics)
        clip_threshold = set_y_limit if set_y_limit is not None else np.inf

        all_plot_upper_bounds = []

        for i, algo in enumerate(algorithms):
            algo_medians = []
            algo_stds = []

            for metric in group_metrics:
                values = all_metrics.get(algo, {}).get(metric, [])
                median, std, _, _ = compute_statistics(values)
                algo_medians.append(median)
                algo_stds.append(std)

            plot_medians = np.array(algo_medians, dtype=float)
            plot_stds = np.array(algo_stds, dtype=float)
            plot_stds_safe = np.nan_to_num(plot_stds)

            offset = (i - num_algorithms / 2 + 0.5) * bar_width
            bar_positions = x_indices + offset

            plot_heights = plot_medians.copy()
            clipped_mask = plot_heights > clip_threshold
            plot_heights[clipped_mask] = clip_threshold

            bars = ax.bar(bar_positions, plot_heights, width=bar_width * 0.9,
                          label=f"{algo} (n={test_counts.get(algo, 0)})",
                          yerr=plot_stds_safe,
                          capsize=4,
                          alpha=0.9, color=cmap(i % cmap.N),
                          error_kw={'alpha': 0.6})

            if is_first_plot and len(bars) > 0:
                legend_handles.append(bars[0])
                legend_labels.append(f"{algo} (n={test_counts.get(algo, 0)})")

            for j, bar in enumerate(bars):
                original_median = plot_medians[j]
                is_clipped = clipped_mask[j]
                if np.isnan(original_median):
                    continue
                if is_clipped:
                    bar.set_hatch('///')
                    bar.set_edgecolor('grey')
                text_y = plot_heights[j]
                if not np.isnan(plot_stds[j]) and plot_stds[j] > 0:
                    if is_clipped:
                        text_y = clip_threshold
                    else:
                        text_y += plot_stds_safe[j]
                y_axis_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                if y_axis_range == 0:
                    y_axis_range = 1
                y_offset = y_axis_range * 0.02
                text_y += y_offset
                ax.text(bar.get_x() + bar.get_width() / 2., text_y,
                        f"{original_median:.2f}", ha='center', va='bottom', fontsize=7, rotation=0, fontweight='bold')

            valid_bounds = [h + s for h, s in zip(plot_heights, plot_stds_safe) if not np.isnan(h)]
            if valid_bounds:
                all_plot_upper_bounds.extend(valid_bounds)

        ax.set_xticks(x_indices)
        ax.set_xticklabels(group_metrics, rotation=30, ha='right')
        ax.set_ylabel("Median Value")
        ax.set_title(title)
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        if set_y_limit is not None:
            ax.set_ylim(0, set_y_limit * 1.15)
        elif all_plot_upper_bounds:
            y_upper = max(all_plot_upper_bounds) * 1.20
            ax.set_ylim(0, y_upper if not np.isnan(y_upper) and y_upper > 0 else 1)
        else:
            ax.set_ylim(0, 1)

    plot_group(ax1, group1_metrics, "Execution Time & Distance", is_first_plot=True)
    plot_group(ax2, group2_metrics, "Entropy, Velocity & False Positives (%)")

    fig.suptitle("Performance Comparison Across Algorithms", fontsize=18)
    if legend_handles:
        fig.legend(handles=legend_handles, labels=legend_labels,
                   loc='upper center', ncol=num_algorithms,
                   bbox_to_anchor=(0.5, 0.96), fontsize='medium')

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])

    plot_filename = "___comparison_performance_summary_clipped_100_trees.eps"
    try:
        plt.savefig(plot_filename)
        print(f"\nSaved comparison plot: {plot_filename}")
    except Exception as e:
        print(f"\nError saving comparison plot {plot_filename}: {e}")
    plt.close(fig)

def export_summary_table(all_metrics, test_counts, num_trees, output_path="__summary_metrics_comparison_trees"):
    """Creates a CSV summary table with mean, median, std, min, max e stampa in console senza errori."""
    rows = []
    algorithms = list(all_metrics.keys())
    if not algorithms:
        print("Cannot export summary table: No algorithm data.")
        return

    all_metric_keys = set()
    for metrics in all_metrics.values():
        all_metric_keys.update(metrics.keys())
    sorted_metric_keys = sorted(all_metric_keys)

    for algo in algorithms:
        metrics = all_metrics[algo]
        t_count = test_counts.get(algo, 0)
        for metric_key in sorted_metric_keys:
            values = metrics.get(metric_key, [])
            arr = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy()
            arr = arr[~np.isnan(arr)]
            mean_val = np.mean(arr) if arr.size > 0 else np.nan
            median, std, min_val, max_val = compute_statistics(values)
            rows.append({
                "Metric": metric_key,
                "Algorithm": algo,
                "Mean": round(mean_val, 4) if not np.isnan(mean_val) else 'N/A',
                "Median": round(median, 4) if not np.isnan(median) else 'N/A',
                "Std Dev": round(std, 4) if not np.isnan(std) else 'N/A',
                "Min": round(min_val, 4) if not np.isnan(min_val) else 'N/A',
                "Max": round(max_val, 4) if not np.isnan(max_val) else 'N/A',
                "Test Count": t_count
            })

    df = pd.DataFrame(rows, columns=[
        "Metric", "Algorithm", "Mean", "Median", "Std Dev", "Min", "Max", "Test Count"
    ])
    df.sort_values(by=["Metric", "Algorithm"], inplace=True)

    try:
        csv_path = output_path + str(num_trees) + ".csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Summary table saved: {csv_path}")
        print("\n--- Summary Statistics Table ---")

        df_to_print = df.copy()
        num_cols = ["Mean", "Median", "Std Dev", "Min", "Max"]
        for c in num_cols:
            df_to_print[c] = pd.to_numeric(df_to_print[c], errors='coerce')

        formatters = {col: (lambda x: f"{x:.4f}") for col in num_cols}
        print(df_to_print.to_string(index=False, na_rep='N/A', formatters=formatters))
    except Exception as e:
        print(f"\n❌ Error saving summary table to {output_path+str(num_trees)+'.csv'}: {e}")

def gather_metrics(base_dir, num_trees=100, has_gt_ids=True):
    run_folders = sorted(glob.glob(os.path.join(base_dir, "run_*")))
    if not run_folders:
        print(f"Warning: No 'run_*' directories found in {base_dir}")

    metrics_lists = {
        "Total Execution Time (s)": [],
        "Final Entropy": [],
        "Total Distance (m)": [],
        "Average Velocity (m/s)": [],
        "Average NMPC Step Execution Time (s)": [],
        "False Positives (%)": [],
        "Not Detected (%)": [],
        "Percentage of Classified Trees (%)": [],
        "Percentage of Correct Classifications on Classified Trees (%)": []
    }

    tests_found_metrics = 0
    for run_idx, run_folder_path in enumerate(run_folders):
        perf_files = glob.glob(os.path.join(run_folder_path, "*performance_metrics.csv"))
        if not perf_files:
            continue

        perf_file = perf_files[0]
        perf = load_performance_metrics(perf_file)
        if perf is None:
            continue

        tests_found_metrics += 1
        metrics_lists["Total Execution Time (s)"].append(perf.get("Total Execution Time (s)", np.nan))
        metrics_lists["Final Entropy"].append(perf.get("Final Entropy", np.nan))
        metrics_lists["Average NMPC Step Execution Time (s)"].append(perf.get("Average Waypoint-to-Waypoint Time (s)", np.nan))

        # Average velocity from *_velocity_commands.csv (median of sqrt(vx^2+vy^2))
        vel_files = glob.glob(os.path.join(run_folder_path, "*_velocity_commands.csv"))
        if not vel_files:
            metrics_lists["Average Velocity (m/s)"].append(np.nan)
        else:
            vel_file = vel_files[0]
            avg_vel = load_average_velocity(vel_file)
            metrics_lists["Average Velocity (m/s)"].append(avg_vel)

        # Total Distance from *_plot_data.csv using uniform resample at DIST_SAMPLE_DT
        plot_data_files = glob.glob(os.path.join(run_folder_path, "*_plot_data.csv"))
        if plot_data_files:
            computed_distance = compute_distance_from_plot_data_resampled(
                plot_data_files[0],
                dt=DIST_SAMPLE_DT,
                skip_first_seconds=DIST_SKIP_FIRST_SECONDS
            )
            metrics_lists["Total Distance (m)"].append(computed_distance)
        else:
            metrics_lists["Total Distance (m)"].append(np.nan)

        # FP metrics
        if not plot_data_files or not has_gt_ids:
            metrics_lists["False Positives (%)"].append(np.nan)
            metrics_lists["Not Detected (%)"].append(np.nan)
            metrics_lists["Percentage of Classified Trees (%)"].append(np.nan)
            metrics_lists["Percentage of Correct Classifications on Classified Trees (%)"].append(np.nan)
        else:
            plot_data_file = plot_data_files[0]
            fp_data = load_false_positives(plot_data_file, num_trees)
            if fp_data and not isinstance(fp_data, float):
                fp, notdet, perc_class, perc_corr = fp_data
                metrics_lists["False Positives (%)"].append(fp)
                metrics_lists["Not Detected (%)"].append(notdet)
                metrics_lists["Percentage of Classified Trees (%)"].append(perc_class)
                metrics_lists["Percentage of Correct Classifications on Classified Trees (%)"].append(perc_corr)
            else:
                metrics_lists["False Positives (%)"].append(np.nan)
                metrics_lists["Not Detected (%)"].append(np.nan)
                metrics_lists["Percentage of Classified Trees (%)"].append(np.nan)
                metrics_lists["Percentage of Correct Classifications on Classified Trees (%)"].append(np.nan)

    if tests_found_metrics > 0:
        print(f"Processed {base_dir}: Found metrics in {tests_found_metrics}/{len(run_folders)} runs.")
    return metrics_lists, tests_found_metrics

def run_batch_analysis(base_dirs_dict, num_trees):
    all_metrics = {}
    test_counts = {}

    print("\n--- Starting Batch Analysis ---")
    for algo_name, config in base_dirs_dict.items():
        base_dir = os.path.join("/home/pantheon/drea/neural_mpc/ros/src/nmpc_ros/batch_tests", config["path"])
        has_gt_ids = config.get("has_gt_ids", True)

        print(f"\n--- Processing Algorithm: {algo_name} (Source: {base_dir}) ---")
        if not os.path.isdir(base_dir):
            print(f"Warning: Directory '{base_dir}' not found. Skipping '{algo_name}'.")
            continue
        metrics_data, num_tests = gather_metrics(base_dir, num_trees, has_gt_ids)
        if num_tests == 0:
            print(f"Warning: No valid test data found for '{algo_name}'. Excluding from comparison.")
            continue

        all_metrics[algo_name] = metrics_data
        test_counts[algo_name] = num_tests

    if not all_metrics:
        print("\nError: No algorithms processed successfully with data. Exiting batch analysis.")
        return

    print("\n--- Generating Comparison Plots & Summary Table ---")
    plot_comparison_metrics(all_metrics, test_counts)
    export_summary_table(all_metrics, test_counts, num_trees)
    print("\n--- Batch Analysis Complete ---")

if __name__ == "__main__":
    num_trees = 100
    base_dirs_to_analyze = {
        f'mower_{num_trees}_slow': {"path": f'batch_test_{num_trees}trees_mower_slow_gt', "has_gt_ids": True},
        f'mower_{num_trees}'     : {"path": f'batch_test_{num_trees}trees_mower_gt', "has_gt_ids": True},
        f'linear_{num_trees}'    : {"path": f'batch_test_{num_trees}trees_linear_gt', "has_gt_ids": True},
        f'greedy_{num_trees}'    : {"path": f'batch_test_{num_trees}trees_greedy_gt', "has_gt_ids": True},
        f'nn_i_mpc_{num_trees}'  : {"path": f'batch_test_{num_trees}trees_nmpc', "has_gt_ids": False},
    }

    print("\n--- Setting up Test Environment  ---")
    run_batch_analysis(base_dirs_to_analyze, num_trees)
