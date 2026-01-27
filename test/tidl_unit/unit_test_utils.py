import os
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box1: List containing [x_min, y_min, x_max, y_max]
        box2: List containing [x_min, y_min, x_max, y_max]
        
    Returns:
        float: IoU value
    """
    try:
        # Calculate intersection coordinates
        x_min_inter = max(box1[0], box2[0])
        y_min_inter = max(box1[1], box2[1])
        x_max_inter = min(box1[2], box2[2])
        y_max_inter = min(box1[3], box2[3])
        
        # Check if boxes intersect
        if x_max_inter <= x_min_inter or y_max_inter <= y_min_inter:
            return 0.0
        
        # Calculate area of intersection
        intersection_area = (x_max_inter - x_min_inter) * (y_max_inter - y_min_inter)
        
        # Calculate area of both boxes
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Calculate union area
        union_area = box1_area + box2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0.0
        
        return iou
    except Exception as e:
        print(f"Error calculating IoU: {e}")
        return 0.0

def generate_plot(binary_results=None, image_results=None, output_dir=None, max_plot_points=10000, plot_name="plot", save_image=True):
    """
    Generate a figure with multiple subplots showing:
    1. Binary results comparison (if binary_results is provided)
    2. Image results (if image_results is provided)
    
    Args:
        binary_results: Dict with keys 'expected_outputs', 'outputs', 'nmse', 'mse', 'delta'
        image_results: Dict with keys 'expected_outputs' (can be None) and 'outputs'
        output_dir: Directory to save the plots
        max_plot_points: Maximum number of points to plot per output (to avoid overcrowding)
        plot_name: Base name for the output files
        save_image: Save plot as png
    
    Returns:
        tuple: (plot_path, plot_base64) for the combined plot
    """
    if output_dir is None:
        output_dir = os.getcwd()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check which inputs are provided
    has_binary_results = binary_results is not None
    has_image_results = image_results is not None
    
    # Initialize variables for binary results
    reference_outputs = []
    actual_outputs = []
    nmse = []
    mse = []
    delta = []
    
    # Extract binary results if provided
    if has_binary_results:
        reference_outputs = binary_results['expected_outputs']
        actual_outputs = binary_results['outputs']
        nmse = binary_results['nmse']
        mse = binary_results['mse']
        delta = binary_results['delta']

    # Check for expected and actual images in image_results
    has_actual_images = False
    has_expected_images = False
    actual_images = {}
    expected_images = {}
    
    if has_image_results:
        if 'outputs' in image_results and image_results['outputs']:
            has_actual_images = True
            actual_images = image_results['outputs']
        
        if 'expected_outputs' in image_results and image_results['expected_outputs']:
            has_expected_images = True
            expected_images = image_results['expected_outputs']

    # Ensure outputs are in list form if binary results are provided
    if has_binary_results:
        if not isinstance(reference_outputs, list):
            reference_outputs = [reference_outputs]
        if not isinstance(actual_outputs, list):
            actual_outputs = [actual_outputs]
        num_outputs = len(reference_outputs)
    else:
        num_outputs = 0
    
    # Calculate number of image pairs if we have both expected and actual images
    num_image_pairs = 0
    if has_actual_images and has_expected_images:
        # Find common image keys
        actual_keys = set(actual_images.keys())
        expected_keys = set(expected_images.keys())
        common_keys = actual_keys.intersection(expected_keys)
        num_image_pairs = len(common_keys)
    
    # Calculate grid dimensions for subplots
    if has_binary_results:
        if num_outputs <= 2:
            binary_rows = 1
            binary_cols = max(1, num_outputs)
        else:
            binary_cols = min(3, num_outputs)  # Maximum 3 columns
            binary_rows = (num_outputs + binary_cols - 1) // binary_cols  # Ceiling division
    else:
        binary_rows = 0
        binary_cols = 0
    
    # Calculate rows needed for images
    if has_actual_images:
        if has_expected_images:
            # We'll show pairs of images side by side
            image_cols = 2  # Two columns: expected and actual
            image_rows = num_image_pairs
        else:
            # Just show actual images
            image_cols = min(3, len(actual_images))
            image_rows = (len(actual_images) + image_cols - 1) // image_cols
    else:
        image_rows = 0
        image_cols = 0
    
    # Total grid dimensions
    total_rows = binary_rows + image_rows
    cols = max(binary_cols, image_cols)
    
    # If no data provided, return without creating plot
    if total_rows == 0:
        print("No data provided for plotting")
        return None, None
    
    # Create figure with subplots
    fig, axes = plt.subplots(total_rows, cols, figsize=(6*cols, 6*total_rows), squeeze=False)
    
    # Process binary outputs if provided
    if has_binary_results:
        for i, (reference, actual) in enumerate(zip(reference_outputs, actual_outputs)):
            # Calculate row and column for this subplot
            row = i // binary_cols
            col = i % binary_cols
            ax = axes[row, col]
            
            if not isinstance(reference, np.ndarray):
                reference = np.array(reference)
            if not isinstance(actual, np.ndarray):
                actual = np.array(actual)
            
            # Flatten arrays for scatter plot
            ref_flat = reference.flatten()
            act_flat = actual.flatten()
            
            # If there are too many points, sample a subset
            if len(ref_flat) > max_plot_points:
                indices = np.random.choice(len(ref_flat), max_plot_points, replace=False)
                ref_flat = ref_flat[indices]
                act_flat = act_flat[indices]
            
            # Calculate min and max for this output
            min_val = min(ref_flat.min(), act_flat.min())
            max_val = max(ref_flat.max(), act_flat.max())
            margin = (max_val - min_val) * 0.05
            min_val -= margin
            max_val += margin
            
            # Plot scatter points
            scatter = ax.scatter(ref_flat, act_flat, alpha=1.0, s=10, color='red')
            
            # Plot the 45-degree line (y=x) for reference
            ax.plot([min_val, max_val], [min_val, max_val], color='blue', alpha=0.7, linewidth=2)
            
            # Set equal scales and limits
            ax.set_aspect('equal')
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)
            
            # Add labels and title
            ax.set_xlabel('Reference Output Values')
            ax.set_ylabel('Actual Output Values')
            ax.set_title(f'Output {i}')
            ax.grid(True, alpha=0.3)
            
            # Add metrics to the plot if available
            legend_text = []
            if nmse is not None and i < len(nmse) and nmse[i] is not None:
                legend_text.append(f'NMSE: {nmse[i]:.7f}')
            if mse is not None and i < len(mse) and mse[i] is not None:
                legend_text.append(f'MSE: {mse[i]:.7f}')
            if delta is not None and i < len(delta) and delta[i] is not None:
                legend_text.append(f'Delta: {delta[i]:.7f}')
            
            if legend_text:
                # Add a text box with metrics in the top left corner
                metrics_text = '\n'.join(legend_text)
                ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add image results
    image_start_row = binary_rows  # Start after binary results
    
    if has_actual_images:
        if has_expected_images:
            # Show pairs of expected and actual images side by side
            actual_keys = set(actual_images.keys())
            expected_keys = set(expected_images.keys())
            common_keys = sorted(list(actual_keys.intersection(expected_keys)))
            
            for i, name in enumerate(common_keys):
                if i >= image_rows:  # Limit to available rows
                    break
                
                # Expected image
                ax_expected = axes[image_start_row + i, 0]
                expected_metadata, expected_image = expected_images[name]
                ax_expected.imshow(expected_image)
                ax_expected.set_title(f'Expected: {name}')
                ax_expected.axis('off')
                
                # Actual image
                ax_actual = axes[image_start_row + i, 1]
                actual_metadata, actual_image = actual_images[name]
                ax_actual.imshow(actual_image)
                ax_actual.set_title(f'Actual: {name}')
                ax_actual.axis('off')
                
                # Don't display metadata as text (removed as per user request)
        else:
            # Only show actual images
            image_names = list(actual_images.keys())
            for i, name in enumerate(image_names):
                row = image_start_row + (i // image_cols)
                col = i % image_cols
                
                if row >= total_rows:  # Skip if we run out of rows
                    break
                
                metadata, image = actual_images[name]
                ax = axes[row, col]
                ax.imshow(image)
                ax.set_title(f'Actual: {name}')
                ax.axis('off')
    
    # Hide any unused subplots
    total_plots = total_rows * cols
    used_plots = 0
    
    # Count used plots
    if has_binary_results:
        used_plots += num_outputs
    
    if has_actual_images and has_expected_images:
        used_plots += num_image_pairs * 2  # Each pair uses 2 plots
    elif has_actual_images:
        used_plots += min(len(actual_images), image_rows * image_cols)
    
    for i in range(used_plots, total_plots):
        row = i // cols
        col = i % cols
        if row < total_rows:  # Make sure we don't access out of bounds
            axes[row, col].axis('off')
    
    # Add overall title and adjust layout
    fig.suptitle(f'Reference vs Actual Outputs - {plot_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
    
    # Save the plot
    plot_img_path = os.path.join(output_dir, f"{plot_name}_plot.png")
    plot_base64_path = os.path.join(output_dir, f"{plot_name}_base64.txt")

    if (save_image):
        plt.savefig(plot_img_path, dpi=50)
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=50)
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    with open(plot_base64_path, "w+") as f:
        f.write(f"{plot_base64}")    
    plt.close(fig)
    
    return (plot_img_path, plot_base64_path)
