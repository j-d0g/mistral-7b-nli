#!/usr/bin/env python3
"""
Script to generate visualizations for the Hugging Face model and dataset cards.
This script creates polished visualizations based on the metrics data.

Usage:
    python generate_card_visualizations.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import sys
from matplotlib.patches import Ellipse, FancyBboxPatch

# Set style for all plots
plt.style.use('ggplot')
sns.set_style("whitegrid")

# Directory paths
METRICS_DIR = "metrics"
OUTPUT_DIR = METRICS_DIR  # Save in the same directory

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Custom color palettes - Improving color contrast
MISTRAL_COLORS = ["#7e57c2", "#3949ab", "#e91e63", "#009688"]  # More contrast between colors
HIGHLIGHT_COLOR = "#ff6e40"
BACKGROUND_COLOR = '#f8f9fa'

def load_metrics():
    """Load metrics from the JSON file."""
    metrics_path = os.path.join(METRICS_DIR, "card_metrics.json")
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading metrics: {e}")
        return {}

def create_dataset_banner():
    """Create a banner image for the dataset card."""
    fig, ax = plt.figure(figsize=(12, 3), dpi=300), plt.gca()
    
    # Set background color
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)
    
    # Add title text
    ax.text(0.5, 0.6, "Mistral-7B NLI", fontsize=36, weight='bold', ha='center', color="#311b92")
    ax.text(0.5, 0.3, "Chain-of-Thought Dataset", fontsize=28, ha='center', color="#5e35b1")
    
    # Add decorative elements
    ax.axhline(y=0.15, xmin=0.15, xmax=0.85, color="#7e57c2", linewidth=4)
    
    # Remove axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, "dataset_banner.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=BACKGROUND_COLOR)
    plt.close()
    
    return output_path

def create_dataset_statistics(metrics):
    """Create a visualization of dataset statistics."""
    if not metrics or "dataset" not in metrics:
        return None
    
    dataset = metrics.get("dataset", {})
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=200)
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    
    # Plot 1: Dataset splits
    splits = ['Training', 'Validation', 'Test']
    
    # Use hardcoded values as specified by the user
    split_values = [
         dataset.get("train_examples", 0), # Use values from metrics
         dataset.get("dev_examples", 0),   # Use values from metrics
         dataset.get("test_examples", 0)    # Use values from metrics
    ]
    
    # Calculate total for percentage calculation
    total_examples = sum(split_values)
    
    # Calculate percentages manually
    train_percentage = (split_values[0] / total_examples) * 100 if total_examples > 0 else 0
    dev_percentage = (split_values[1] / total_examples) * 100 if total_examples > 0 else 0
    test_percentage = (split_values[2] / total_examples) * 100 if total_examples > 0 else 0
    
    split_percentages = [
        dataset.get("train_percentage", train_percentage), # Prefer metrics if available
        dataset.get("dev_percentage", dev_percentage),
        dataset.get("test_percentage", test_percentage)
    ]
    
    # Create color map
    cmap = LinearSegmentedColormap.from_list("", MISTRAL_COLORS[:3])
    colors = [cmap(i) for i in np.linspace(0, 1, len(splits))]
    
    # Plot bar chart with percentages
    bars = ax1.bar(splits, split_values, color=colors)
    ax1.set_title('Dataset Split Distribution', fontsize=16, pad=15)
    ax1.set_ylabel('Number of Examples', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add count and percentage labels
    for i, (bar, percentage) in enumerate(zip(bars, split_percentages)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=10)
        ax1.text(bar.get_x() + bar.get_width()/2., height/2 if height > 0 else 0.5,
                f'{percentage:.1f}%',
                ha='center', va='center', fontsize=12, fontweight='bold', color='white' if height > 0 else 'black')
    
    # Plot 2: Label distribution
    label_dist = dataset.get("label_distribution", {})
    labels = ['Entailment', 'No Entailment']
    label_values = [
        label_dist.get("entailment", 0),
        label_dist.get("no_entailment", 0)
    ]
    
    # Check if we have valid data for the pie chart
    if sum(label_values) == 0:
        # No valid data, display a placeholder message
        ax2.text(0.5, 0.5, "No label distribution data available",
               ha='center', va='center', fontsize=12, transform=ax2.transAxes)
        ax2.axis('off')
    else:
        # We have data, create the pie chart
        label_percentages = [
            label_dist.get("entailment_percentage", 0),
            label_dist.get("no_entailment_percentage", 0)
        ]
        
        # Create color map for labels
        pie_colors = [MISTRAL_COLORS[0], MISTRAL_COLORS[2]]  # Using colors with more contrast
        
        # Plot pie chart with percentages
        wedges, texts, autotexts = ax2.pie(
            label_values,
            labels=labels,
            colors=pie_colors,
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops={'edgecolor': 'w', 'linewidth': 1},
            textprops={'fontsize': 12}
        )
        
        # Update text properties
        for autotext in autotexts:
            autotext.set_fontweight('bold')
        
        ax2.set_title('Label Distribution', fontsize=16, pad=15)
    
    # Final adjustments
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, "dataset_statistics.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=BACKGROUND_COLOR)
    plt.close()
    
    return output_path

def create_token_length_visualization(metrics):
    """Create visualization of token length statistics."""
    if not metrics or "tokens" not in metrics:
        return None
    
    tokens = metrics.get("tokens", {})
    
    # Extract data - Removed 'Reflection'
    components = ['Premise', 'Hypothesis', 'Reasoning Chain']
    avg_tokens = [
        tokens.get("premise", {}).get("avg", 0),
        tokens.get("hypothesis", {}).get("avg", 0),
        tokens.get("thought_process", {}).get("avg", 0)
    ]
    
    # Check if we have valid data
    if all(val == 0 for val in avg_tokens):
        # No valid data, create a placeholder
        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
        fig.patch.set_facecolor(BACKGROUND_COLOR)
        ax.text(0.5, 0.5, "No token length data available",
               ha='center', va='center', fontsize=16, fontweight='bold', transform=ax.transAxes)
        ax.axis('off')
        
        # Save figure
        output_path = os.path.join(OUTPUT_DIR, "token_lengths.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=BACKGROUND_COLOR)
        plt.close()
        
        return output_path
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    
    # Create color map with better contrast - adjust for fewer components
    colors = MISTRAL_COLORS[:len(components)]  # Use distinct colors
    
    # Plot horizontal bar chart
    bars = ax.barh(components, avg_tokens, color=colors)
    ax.set_title('Average Token Length by Component', fontsize=16, pad=15)
    ax.set_xlabel('Number of Tokens (estimated)', fontsize=12) # Added (estimated)
    ax.grid(axis='x', alpha=0.3)
    
    # Add count labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 3, bar.get_y() + bar.get_height()/2.,
               f'{width:.1f}',
               ha='left', va='center', fontsize=12)
    
    # Check if we have reasoning chain data for the box plot
    thought_tokens_data = tokens.get("thought_process", {})
    thought_min = thought_tokens_data.get("min", 0)
    thought_max = thought_tokens_data.get("max", 0)
    thought_quartiles = thought_tokens_data.get("quartiles", [0, 0, 0])
    
    # Only add the box plot if we have non-zero values for thought_process
    if thought_max > 0:
        # Create a secondary plot for token ranges
        axins = ax.inset_axes([0.6, 0.15, 0.35, 0.3])
        
        # Box plot data
        boxplot_data = [[thought_min, thought_quartiles[0], thought_quartiles[1], thought_quartiles[2], thought_max]]
        
        # Create box plot
        axins.boxplot(boxplot_data, vert=False, widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor=MISTRAL_COLORS[2 % len(MISTRAL_COLORS)], color=MISTRAL_COLORS[3 % len(MISTRAL_COLORS)]),
                    medianprops=dict(color=HIGHLIGHT_COLOR, linewidth=2))
        
        axins.set_title('Reasoning Chain Token Distribution', fontsize=10)
        axins.set_yticklabels([])
        axins.grid(axis='x', alpha=0.3)
    
    # Final adjustments
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, "token_lengths.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=BACKGROUND_COLOR)
    plt.close()
    
    return output_path

def create_data_pipeline_visualization():
    """Create a visual representation of the data pipeline."""
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)
    
    # Define stages and their positions
    stages = ['Original\nData', 'Generate\nThoughts', 'Generate\nReflections', 'Prepare\nFine-tuning\nData']
    x_positions = [1, 3, 5, 7]
    y_position = 2
    
    # Draw boxes for each stage - make them larger to fit multi-line text
    for i, (stage, x) in enumerate(zip(stages, x_positions)):
        color = MISTRAL_COLORS[i % len(MISTRAL_COLORS)]
        # Increased height from 1 to 1.2 and width to 1.8
        rect = plt.Rectangle((x-0.9, y_position-0.6), 1.8, 1.2, 
                           facecolor=color, alpha=0.8, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        # Use smaller font with multi-line text
        ax.text(x, y_position, stage, ha='center', va='center', color='white', fontweight='bold', fontsize=10)
        
        # Add arrows between stages - adjust position due to larger boxes
        if i < len(stages) - 1:
            ax.arrow(x+1.0, y_position, 0.1, 0, head_width=0.1, head_length=0.1, 
                    fc='black', ec='black', length_includes_head=True)
    
    # Add file types below each stage
    file_types = ['(CSV)', '(JSON)', '(JSON)', '(JSONL)']
    for i, (file_type, x) in enumerate(zip(file_types, x_positions)):
        ax.text(x, y_position-1.0, file_type, ha='center', va='center', 
               fontsize=9, color='black')
    
    # Add pipeline flow explanation above
    ax.text(4, y_position+1.5, 'Data Preparation Pipeline', ha='center', va='center',
           fontsize=16, fontweight='bold', color=MISTRAL_COLORS[1])
    
    # Add detailed descriptions - further simplified and reduced font size
    descriptions = [
        'Original premise-\nhypothesis pairs',
        'Generate Chain-of-\nThought reasoning',
        'Improve via\nreflection',
        'Format for\nfine-tuning'
    ]
    
    for i, (desc, x) in enumerate(zip(descriptions, x_positions)):
        ax.text(x, y_position-1.5, desc, ha='center', va='center',
               fontsize=8, color='black', style='italic',
               bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=2))
    
    # Remove axes
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, "data_pipeline.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=BACKGROUND_COLOR)
    plt.close()
    
    return output_path

def create_reflection_process_visualization():
    """Create a visualization of the reflection process."""
    # Create figure with more whitespace on sides for centering
    fig, ax = plt.subplots(figsize=(12, 5), dpi=200)  # Increased width from 10 to 12
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)
    
    # Define the process steps - using shorter text where possible
    steps = [
        'Initial\nThought',
        'Reflection\nPrompt',
        'Self-\nCritique',
        'Improved\nReasoning'
    ]
    
    # Create visual flow with larger circles
    radius = 0.6
    # Adjust starting position to center the entire flow
    start_x = 2.5  # Shifted right from 0
    # Calculate total width of visualization to center it
    total_width = (len(steps) - 1) * 2.5
    
    # Center-aligned positions with more spacing
    circle_positions = [(start_x + i*2.5, 1.5) for i in range(len(steps))]
    
    for i, (step, position) in enumerate(zip(steps, circle_positions)):
        x, y = position
        circle = plt.Circle((x, y), radius, facecolor=MISTRAL_COLORS[i % len(MISTRAL_COLORS)], 
                           alpha=0.8, edgecolor='black')
        ax.add_patch(circle)
        # Position text in the center of the circle with smaller font
        ax.text(x, y, step, ha='center', va='center', color='white', fontweight='bold', fontsize=9)
        
        # Add connecting arrows
        if i < len(steps) - 1:
            next_x = circle_positions[i+1][0]
            ax.arrow(x + radius + 0.05, y, next_x - x - 2*radius - 0.1, 0,
                    head_width=0.2, head_length=0.2, fc='black', ec='black', length_includes_head=True)
    
    # Add examples below circles - make them more compact and move further down
    examples = [
        '"Penguins are birds,\nso they can fly."',
        '"Review your reasoning.\nIs there a flaw in\nyour logic?"',
        '"I need to consider that\npenguins are flightless\nbirds."',
        '"If premise is that\nall birds fly, then\npenguins must fly."'
    ]
    
    for i, (example, position) in enumerate(zip(examples, circle_positions)):
        x, y = position
        # Move text boxes much further down from circles to avoid overlap
        ax.text(x, y-1.5, example, ha='center', va='center', fontsize=8,
               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # Add title
    mid_x = start_x + total_width / 2  # Calculate center x position
    ax.text(mid_x, 3, 'Reflection Process for Improving Reasoning', 
           ha='center', va='center', fontsize=16, fontweight='bold', color=MISTRAL_COLORS[1])
    
    # Adjust axes limits to center everything
    ax.set_xlim(1, start_x + total_width + 1.5)
    ax.set_ylim(0, 3.5)
    ax.axis('off')
    
    # Add tight layout to remove excess margins
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, "reflection_process.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=BACKGROUND_COLOR)
    plt.close()
    
    return output_path

def create_model_performance_visualization(metrics):
    """Create visualization of model performance comparison."""
    if not metrics or "models" not in metrics or not metrics.get("models"):
        # Create placeholder visualization
        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
        fig.patch.set_facecolor(BACKGROUND_COLOR)
        
        # Set title indicating this is a placeholder
        ax.text(0.5, 0.5, "Model Performance Comparison\nPlaceholder - Add actual model data",
               ha='center', va='center', fontsize=20, fontweight='bold', color=MISTRAL_COLORS[1])
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Save figure
        output_path = os.path.join(OUTPUT_DIR, "model_performance.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=BACKGROUND_COLOR)
        plt.close()
        
        return output_path
    
    models = metrics.get("models", {})
    
    # Extract data
    model_names = list(models.keys())
    if not model_names:
        # If no models found, create placeholder
        return create_model_performance_visualization(None)
    
    # Extract metrics for each model
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
    metrics_data = {metric: [] for metric in metrics_names}
    
    for model in model_names:
        model_data = models.get(model, {})
        for metric in metrics_names:
            metrics_data[metric].append(model_data.get(metric, 0))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    
    # Set width of bars
    bar_width = 0.2
    positions = np.arange(len(model_names))
    
    # Create grouped bar chart with more contrasting colors
    for i, metric in enumerate(metrics_names):
        offset = (i - 1.5) * bar_width
        bars = ax.bar(positions + offset, metrics_data[metric], bar_width, 
                     label=metric.capitalize(), color=MISTRAL_COLORS[i % len(MISTRAL_COLORS)])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9, rotation=0)
    
    # Customize chart
    ax.set_title('Model Performance Comparison', fontsize=16, pad=15)
    ax.set_xlabel('Model Variant', fontsize=12, labelpad=10)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_xticks(positions)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0, 110)  # Set y-axis limit to accommodate labels
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(metrics_names))
    
    # Add grid
    ax.grid(axis='y', alpha=0.3)
    
    # Final adjustments
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, "model_performance.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=BACKGROUND_COLOR)
    plt.close()
    
    return output_path

def create_token_vs_accuracy_visualization(metrics=None):
    """Create visualization showing relationship between token length and accuracy from real data."""
    # Check if we have token_vs_accuracy data in metrics
    if metrics and "token_vs_accuracy" in metrics and metrics["token_vs_accuracy"]:
        # Use real data from metrics
        token_data = metrics["token_vs_accuracy"]
        token_ranges = token_data.get("ranges", ['0-100', '101-200', '201-300', '301+'])
        accuracies = token_data.get("accuracies", [0, 0, 0, 0])
        counts = token_data.get("counts", [0, 0, 0, 0])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
        fig.patch.set_facecolor(BACKGROUND_COLOR)
        
        # Normalize counts for plotting
        max_count = max(counts) if counts and max(counts) > 0 else 1
        normalized_counts = [count/max_count * 50 for count in counts]
        
        # Plot scatter points with size representing count
        for i, (token_range, accuracy, count, norm_count) in enumerate(zip(token_ranges, accuracies, counts, normalized_counts)):
            ax.scatter(i, accuracy, s=norm_count*20, alpha=0.7, color=MISTRAL_COLORS[i % len(MISTRAL_COLORS)], 
                      edgecolor='black', linewidth=1)
            
            # Add count labels
            ax.text(i, accuracy+3, f'{count} examples', ha='center', va='center', fontsize=10)
        
        # Connect points with line
        ax.plot(range(len(token_ranges)), accuracies, color='gray', linestyle='--', alpha=0.5)
        
        # Add annotations
        for i, (token_range, accuracy) in enumerate(zip(token_ranges, accuracies)):
            ax.text(i, accuracy-4, f'{accuracy:.1f}%', ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Get model name from metrics if available
        model_name = "Fine-tuned Model"
        if "model" in metrics and metrics["model"]:
            model_name = metrics["model"].split("/")[-1]
        
        # Customize chart
        ax.set_title(f'Reasoning Chain Length vs. Accuracy ({model_name})', fontsize=16, pad=15)
        ax.set_xlabel('Token Range in Reasoning Chain', fontsize=12, labelpad=10)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_xticks(range(len(token_ranges)))
        ax.set_xticklabels(token_ranges)
        ax.set_ylim(50, 100)  # Set y-axis limit to focus on the relevant range
        
        # Add grid
        ax.grid(alpha=0.3)
    else:
        # Create placeholder visualization with a message that no data is available
        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
        fig.patch.set_facecolor(BACKGROUND_COLOR)
        
        ax.text(0.5, 0.5, "No token vs accuracy data available.\nRun analyze_token_accuracy.py to generate this data.",
               ha='center', va='center', fontsize=14, fontweight='bold', transform=ax.transAxes)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    # Final adjustments
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, "token_vs_accuracy.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=BACKGROUND_COLOR)
    plt.close()
    
    return output_path

def create_reasoning_benefits_visualization():
    """Create visualization highlighting the benefits of Chain-of-Thought reasoning using a bullet-point style."""
    # Create figure - adjust height for vertical list
    fig, ax = plt.subplots(figsize=(10, 10), dpi=200) # Increased height for more vertical space
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)
  
    benefits = [
        'Step-by-Step Logic',
        'Transparency',
        'Educational Value',
        'Error Detection'
    ]
  
    descriptions = [
        'Breaks complex problems into discrete logical steps, making the model\'s process clear and auditable.',
        'Makes model decisions explainable and interpretable to users, fostering trust and understanding.',
        'Helps users understand logical inference patterns and identify how the model arrives at conclusions.',
        'Enables identification of specific flaws or biases in reasoning, facilitating targeted improvements.'
    ]
 
    # Main title
    ax.text(0.5, 0.95, 'Benefits of Chain-of-Thought Reasoning',
            ha='center', va='center', fontsize=18, fontweight='bold', color=MISTRAL_COLORS[1], transform=fig.transFigure)
 
    box_width = 0.8
    box_height = 0.15
    start_y = 0.85 # Adjusted from 0.80 to give more space at the bottom
    y_spacing = 0.05
 
    for i, (benefit, description) in enumerate(zip(benefits, descriptions)):
        current_y = start_y - i * (box_height + y_spacing)
        
        rect_x = (1 - box_width) / 2
        box = FancyBboxPatch((rect_x, current_y - box_height),
                             box_width, box_height,
                             boxstyle="round,pad=0.03,rounding_size=0.02",
                             facecolor=MISTRAL_COLORS[i % len(MISTRAL_COLORS)],
                             edgecolor="#333333",
                             linewidth=1.5,
                             alpha=0.8,
                             transform=fig.transFigure)
        ax.add_patch(box)
 
        ax.text(0.5, current_y - box_height / 4, benefit,
                ha='center', va='center', fontsize=14, fontweight='bold', color='white', 
                transform=fig.transFigure)
 
        ax.text(0.5, current_y - box_height * 0.65,
                description,
                ha='center', va='center', fontsize=10, color='white',
                wrap=True, transform=fig.transFigure, 
                bbox=dict(boxstyle='round,pad=0.2', fc='none', ec='none'))
 
    ax.axis('off')
    plt.tight_layout(pad=1.0) # Added tight_layout for better spacing
  
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, "reasoning_benefits.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=BACKGROUND_COLOR)
    plt.close()
    
    return output_path

def create_model_architecture_visualization():
    """Create visualization of the model architecture with LoRA adaptation."""
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)
    
    # Define elements of the architecture
    base_height = 4
    base_width = 3
    adapter_height = 1.2
    adapter_width = 2
    
    # Draw base model (Mistral-7B)
    rect_base = plt.Rectangle((-base_width/2, 0), base_width, base_height, 
                             facecolor=MISTRAL_COLORS[-1], alpha=0.8, edgecolor='black', linewidth=2)
    ax.add_patch(rect_base)
    
    # Add base model text
    ax.text(0, base_height/2, "Mistral-7B\nBase Model\n(Frozen)", ha='center', va='center', 
           color='white', fontsize=14, fontweight='bold')
    ax.text(0, base_height/5, "7 Billion Parameters", ha='center', va='center', 
           color='white', fontsize=10)
    
    # Draw LoRA adapters
    adapter_positions = [
        (-adapter_width/2, base_height + 0.5),  # Top adapter
        (-base_width/2 - adapter_width - 0.5, base_height/2 - adapter_height/2),  # Left adapter
        (base_width/2 + 0.5, base_height/2 - adapter_height/2)  # Right adapter
    ]
    
    adapter_labels = [
        "Value & Output\nProjections",
        "Query\nProjection",
        "Key\nProjection"
    ]
    
    for i, (pos, label) in enumerate(zip(adapter_positions, adapter_labels)):
        x, y = pos
        # Draw adapter
        adapter = plt.Rectangle((x, y), adapter_width, adapter_height,
                              facecolor=MISTRAL_COLORS[i % len(MISTRAL_COLORS)], alpha=0.8, 
                              edgecolor='black', linewidth=1)
        ax.add_patch(adapter)
        
        # Add adapter text
        ax.text(x + adapter_width/2, y + adapter_height/2, label, ha='center', va='center',
               color='white', fontsize=10, fontweight='bold')
        
        # Connect adapter to base model with arrows
        if i == 0:  # Top adapter
            ax.arrow(0, base_height + 0.05, 0, 0.25, head_width=0.1, head_length=0.1,
                    fc='black', ec='black')
        elif i == 1:  # Left adapter
            ax.arrow(x + adapter_width + 0.05, y + adapter_height/2, 0.25, 0, head_width=0.1, head_length=0.1,
                    fc='black', ec='black')
        else:  # Right adapter
            ax.arrow(x - 0.05, y + adapter_height/2, -0.25, 0, head_width=0.1, head_length=0.1,
                    fc='black', ec='black')
    
    # Add LoRA explanation
    lora_text = "Low-Rank Adaptation (LoRA)\nRank: 16-32 | Alpha: 32-64\n<1% of base parameters"
    ax.text(0, base_height + adapter_height + 1, lora_text, ha='center', va='center',
           fontsize=12, fontweight='bold', color=MISTRAL_COLORS[0],
           bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # Add input and output
    input_text = "Input:\nPremise + Hypothesis"
    ax.text(-base_width/2 - 2, 0.5, input_text, ha='center', va='center',
           fontsize=10, bbox=dict(facecolor='lightgray', alpha=0.7, boxstyle='round,pad=0.5'))
    
    output_text = "Output:\nChain-of-Thought\n+ Label"
    ax.text(base_width/2 + 2, 0.5, output_text, ha='center', va='center',
           fontsize=10, bbox=dict(facecolor='lightgray', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # Add arrows for input/output
    ax.arrow(-base_width/2 - 1.5, 0.5, 1, 0, head_width=0.1, head_length=0.1,
            fc='black', ec='black')
    ax.arrow(base_width/2, 0.5, 1, 0, head_width=0.1, head_length=0.1,
            fc='black', ec='black')
    
    # Add title
    ax.text(0, -1, 'QLoRA Fine-tuning Architecture', ha='center', va='center',
           fontsize=16, fontweight='bold', color=MISTRAL_COLORS[1])
    
    # Add 4-bit quantization note
    quant_text = "4-bit Quantization (NF4)\nwith Double Quantization"
    ax.text(0, -0.5, quant_text, ha='center', va='center',
           fontsize=10, style='italic', color='black',
           bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    
    # Remove axes
    ax.set_xlim(-5, 5)
    ax.set_ylim(-1.5, base_height + adapter_height + 2)
    ax.axis('off')
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, "model_architecture.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=BACKGROUND_COLOR)
    plt.close()
    
    return output_path

def create_training_dynamics_visualization():
    """Create visualization of training dynamics across ablations."""
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=200)
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    
    # Sample data for illustration (replace with actual data)
    steps = np.arange(0, 2000, 100)
    
    # Loss curves
    loss_abl0 = 2.5 * np.exp(-steps/1000) + 0.8 + 0.1 * np.random.randn(len(steps))
    loss_abl1 = 2.3 * np.exp(-steps/1200) + 0.7 + 0.1 * np.random.randn(len(steps))
    loss_abl2 = 2.2 * np.exp(-steps/800) + 0.6 + 0.08 * np.random.randn(len(steps))
    
    # Accuracy curves
    acc_abl0 = 70 + 25 * (1 - np.exp(-steps/800)) + 2 * np.random.randn(len(steps))
    acc_abl1 = 72 + 24 * (1 - np.exp(-steps/1000)) + 2 * np.random.randn(len(steps))
    acc_abl2 = 75 + 23 * (1 - np.exp(-steps/600)) + 1.5 * np.random.randn(len(steps))
    
    # Plot loss curves - using more distinct colors
    ax1.plot(steps, loss_abl0, color=MISTRAL_COLORS[0], label='Ablation0', linewidth=2)
    ax1.plot(steps, loss_abl1, color=MISTRAL_COLORS[1], label='Ablation1', linewidth=2)
    ax1.plot(steps, loss_abl2, color=MISTRAL_COLORS[2], label='Ablation2', linewidth=2)
    
    # Add early stopping points
    ax1.scatter([1400, 1600, 1800], [loss_abl0[14], loss_abl1[16], loss_abl2[18]], 
                color=HIGHLIGHT_COLOR, s=80, zorder=10, edgecolor='white')
    
    # Customize loss plot
    ax1.set_title('Training Loss', fontsize=14)
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.legend()
    
    # Plot accuracy curves
    ax2.plot(steps, acc_abl0, color=MISTRAL_COLORS[0], label='Ablation0', linewidth=2)
    ax2.plot(steps, acc_abl1, color=MISTRAL_COLORS[1], label='Ablation1', linewidth=2)
    ax2.plot(steps, acc_abl2, color=MISTRAL_COLORS[2], label='Ablation2', linewidth=2)
    
    # Add early stopping points
    ax2.scatter([1400, 1600, 1800], [acc_abl0[14], acc_abl1[16], acc_abl2[18]], 
                color=HIGHLIGHT_COLOR, s=80, zorder=10, edgecolor='white')
    
    # Customize accuracy plot
    ax2.set_title('Validation Accuracy', fontsize=14)
    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(65, 100)
    ax2.legend()
    
    # Add note that this is placeholder data
    fig.text(0.5, 0.01, "Placeholder visualization - Replace with actual training dynamics data",
            ha='center', va='bottom', fontsize=10, style='italic')
    
    # Title for the whole figure
    fig.suptitle('Training Dynamics Comparison', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, "training_dynamics.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=BACKGROUND_COLOR)
    plt.close()
    
    return output_path

def main():
    """Main function to generate all visualizations."""
    print("Generating visualizations for Hugging Face cards...")
    
    # Load metrics data
    metrics = load_metrics()
    
    # Generate visualizations
    visualizations = {}
    
    # Dataset visualizations
    print("Creating dataset banner...")
    visualizations["dataset_banner"] = create_dataset_banner()
    
    print("Creating dataset statistics visualization...")
    visualizations["dataset_statistics"] = create_dataset_statistics(metrics)
    
    print("Creating token length visualization...")
    visualizations["token_lengths"] = create_token_length_visualization(metrics)
    
    print("Creating data pipeline visualization...")
    visualizations["data_pipeline"] = create_data_pipeline_visualization()
    
    print("Creating reflection process visualization...")
    visualizations["reflection_process"] = create_reflection_process_visualization()
    
    print("Creating reasoning benefits visualization...")
    visualizations["reasoning_benefits"] = create_reasoning_benefits_visualization()
    
    # Model visualizations
    print("Creating model performance visualization...")
    visualizations["model_performance"] = create_model_performance_visualization(metrics)
    
    print("Creating token vs accuracy visualization...")
    visualizations["token_vs_accuracy"] = create_token_vs_accuracy_visualization(metrics)
    
    print("Creating model architecture visualization...")
    visualizations["model_architecture"] = create_model_architecture_visualization()
    
    print("Creating training dynamics visualization...")
    visualizations["training_dynamics"] = create_training_dynamics_visualization()
    
    print("All visualizations generated and saved to the metrics directory.")
    print("Done!")

if __name__ == "__main__":
    main() 