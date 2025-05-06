#!/usr/bin/env python3
"""
Visualize Training Dynamics from Weights & Biases

This script fetches training data from Weights & Biases and generates
a visualization of the training dynamics for different model ablations.

Usage:
    python visualize_wandb_training.py [--project PROJECT_NAME] [--entity ENTITY_NAME]
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Try importing wandb with error handling
try:
    import wandb
    from wandb import Api
    wandb_available = True
    print("Successfully imported wandb module")
except ImportError:
    wandb_available = False
    print("WARNING: Could not import wandb. Make sure it's installed with 'pip install wandb'")

# Constants for visualization
BACKGROUND_COLOR = "#FAFAFA"
MISTRAL_COLORS = ["#7B61FF", "#4EAEFF", "#36D399", "#F1FA8C", "#FF6E6E"]
HIGHLIGHT_COLOR = "#FF79C6"

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize training dynamics from W&B runs")
    parser.add_argument("--project", type=str, default="mistral-nli-sft",
                        help="W&B project name")
    parser.add_argument("--entity", type=str, default=None,
                        help="W&B entity (username or team name)")
    parser.add_argument("--runs", type=str, nargs='+', default=None,
                        help="Specific run IDs to include (if not provided, will use the latest 3 runs)")
    parser.add_argument("--output", type=str, default="metrics/training_dynamics.png",
                        help="Output path for the visualization")
    parser.add_argument("--labels", type=str, nargs='+', default=None,
                        help="Labels for each run (if not provided, will use run names)")
    return parser.parse_args()

def get_wandb_runs(project, entity=None, run_ids=None, max_runs=3):
    """Fetch run data from W&B API"""
    if not wandb_available:
        print("ERROR: wandb is not available. Cannot fetch runs.")
        return []
    
    # Initialize W&B API
    try:
        # Try both possible API initialization methods
        try:
            api = wandb.Api()
            print("Using wandb.Api() method")
        except AttributeError:
            try:
                api = Api()
                print("Using Api() method")
            except:
                api = wandb.api()
                print("Using wandb.api() method")
    except Exception as e:
        print(f"ERROR: Failed to initialize wandb API: {e}")
        print("Make sure you're logged in with 'wandb login'")
        return []
    
    # Get runs from the project
    try:
        if run_ids:
            runs = []
            for run_id in run_ids:
                try:
                    run_path = f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}"
                    print(f"Fetching run: {run_path}")
                    run = api.run(run_path)
                    runs.append(run)
                except Exception as e:
                    print(f"Error fetching run {run_id}: {e}")
        else:
            # Get all runs and sort by creation time (newest first)
            print(f"Fetching all runs for project {entity}/{project}" if entity else project)
            all_runs = api.runs(f"{entity}/{project}" if entity else project)
            # Filter completed runs
            completed_runs = [run for run in all_runs if run.state == "finished"]
            # Sort by creation time (newest first)
            sorted_runs = sorted(completed_runs, key=lambda r: r.created_at, reverse=True)
            # Take the latest N runs
            runs = sorted_runs[:max_runs]
    except Exception as e:
        print(f"ERROR: Failed to fetch runs: {e}")
        return []
    
    print(f"Fetched {len(runs)} runs from W&B")
    for run in runs:
        print(f"  - {run.name} (ID: {run.id})")
    
    return runs

def fetch_run_history(run):
    """Fetch history data for a run"""
    # Convert history to dataframe
    try:
        print(f"Fetching history for run {run.name} (ID: {run.id})")
        history_df = run.history()
        print(f"Successfully fetched history with {len(history_df)} records")
    except Exception as e:
        print(f"ERROR: Failed to fetch history for run {run.name}: {e}")
        return pd.DataFrame(), None
    
    # Check if we have the necessary metrics
    required_metrics = ["train/loss", "eval/loss"]
    optional_metrics = ["train/mean_token_accuracy", "eval/accuracy", "eval/mean_token_accuracy"]
    
    # Find available metrics
    available_metrics = history_df.columns.tolist()
    print(f"Available metrics: {available_metrics}")
    
    # Check required metrics
    for metric in required_metrics:
        if metric not in available_metrics:
            print(f"Warning: Required metric '{metric}' not found in run {run.name}")
    
    # Find first available accuracy metric
    accuracy_metric = None
    for metric in optional_metrics:
        if metric in available_metrics:
            accuracy_metric = metric
            print(f"Using accuracy metric: {accuracy_metric}")
            break
    
    if not accuracy_metric:
        print(f"Warning: No accuracy metric found in run {run.name}")
    
    return history_df, accuracy_metric

def create_training_dynamics_visualization(runs_data, output_path, run_labels=None):
    """Create visualization of training dynamics across runs"""
    if not runs_data:
        print("No run data available to visualize. Creating placeholder visualization.")
        # Create placeholder visualization
        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
        fig.patch.set_facecolor(BACKGROUND_COLOR)
        ax.text(0.5, 0.5, "No W&B training data available.\nPlease check W&B connection and run IDs.",
               ha='center', va='center', fontsize=14, fontweight='bold', transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=BACKGROUND_COLOR)
        plt.close()
        return output_path
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=200)
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    
    # Process each run
    for i, (run, history_df, accuracy_metric) in enumerate(runs_data):
        try:
            color = MISTRAL_COLORS[i % len(MISTRAL_COLORS)]
            label = run_labels[i] if run_labels and i < len(run_labels) else run.name
            
            # Get training steps (x-axis) - use global_step if available, otherwise index
            if "global_step" in history_df.columns:
                steps = history_df["global_step"].values
            else:
                steps = np.arange(len(history_df))
            
            # Plot training loss
            if "train/loss" in history_df.columns:
                train_loss = history_df["train/loss"].values
                # Apply smoothing to reduce noise
                window_size = max(1, len(train_loss) // 10)
                smoothed_train_loss = pd.Series(train_loss).rolling(window=window_size, min_periods=1).mean().values
                ax1.plot(steps, smoothed_train_loss, color=color, linestyle="-", alpha=0.4, label=f"{label} (train)")
            
            # Plot validation loss
            if "eval/loss" in history_df.columns:
                eval_loss = history_df["eval/loss"].values
                # Filter out NaN values which might occur when eval wasn't run
                valid_indices = ~np.isnan(eval_loss)
                if np.any(valid_indices):
                    eval_steps = steps[valid_indices]
                    valid_eval_loss = eval_loss[valid_indices]
                    ax1.plot(eval_steps, valid_eval_loss, color=color, linestyle="-", linewidth=2, label=f"{label} (eval)")
                    
                    # Highlight the minimum point
                    min_idx = np.argmin(valid_eval_loss)
                    ax1.scatter([eval_steps[min_idx]], [valid_eval_loss[min_idx]], 
                                color=HIGHLIGHT_COLOR, s=80, zorder=10, edgecolor='white')
            
            # Plot accuracy if available
            if accuracy_metric and accuracy_metric in history_df.columns:
                accuracy = history_df[accuracy_metric].values
                # Convert to percentage if needed
                if np.max(accuracy) <= 1.0:
                    accuracy = accuracy * 100
                    
                # Filter out NaN values
                valid_indices = ~np.isnan(accuracy)
                if np.any(valid_indices):
                    acc_steps = steps[valid_indices]
                    valid_accuracy = accuracy[valid_indices]
                    ax2.plot(acc_steps, valid_accuracy, color=color, linewidth=2, label=label)
                    
                    # Highlight the maximum point
                    max_idx = np.argmax(valid_accuracy)
                    ax2.scatter([acc_steps[max_idx]], [valid_accuracy[max_idx]], 
                                color=HIGHLIGHT_COLOR, s=80, zorder=10, edgecolor='white')
        except Exception as e:
            print(f"Error processing run {run.name}: {e}")
            continue
    
    # Customize loss plot
    ax1.set_title('Training & Validation Loss', fontsize=14)
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9)
    
    # Customize accuracy plot
    ax2.set_title('Model Accuracy', fontsize=14)
    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(50, 100)  # Set y-axis limit to focus on the relevant range
    ax2.legend(fontsize=9)
    
    # Title for the whole figure
    fig.suptitle('Training Dynamics of Model Ablations', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save figure
    try:
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=BACKGROUND_COLOR)
        plt.close()
        print(f"Training dynamics visualization saved to: {output_path}")
    except Exception as e:
        print(f"Error saving visualization: {e}")
    
    return output_path

def main():
    """Main function to generate training dynamics visualization from W&B data"""
    args = parse_args()
    
    print(f"Fetching data from W&B project: {args.project}")
    
    # Get W&B runs
    runs = get_wandb_runs(args.project, args.entity, args.runs)
    
    if not runs:
        print("No runs found. Please check your W&B project and credentials.")
        # Create empty visualization
        create_training_dynamics_visualization([], args.output)
        return
    
    # Fetch data for each run
    runs_data = []
    for run in runs:
        print(f"Fetching history for run: {run.name}")
        history_df, accuracy_metric = fetch_run_history(run)
        if not history_df.empty:
            runs_data.append((run, history_df, accuracy_metric))
    
    if not runs_data:
        print("No valid data found in the runs.")
        # Create empty visualization
        create_training_dynamics_visualization([], args.output)
        return
    
    # Create visualization
    output_path = create_training_dynamics_visualization(runs_data, args.output, args.labels)
    print(f"Visualization saved to: {output_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: Script execution failed: {e}")
        import traceback
        traceback.print_exc() 