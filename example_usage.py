# example_usage.py
"""
Example usage of the linear filters without GUI
"""

import cv2
import numpy as np
from linear_filters import apply_linear_filter

def example_filter_usage():
    """Demonstrate how to use the filters programmatically."""
    
    # Load an image (you can replace this with your own image path)
    # For this example, we'll create a simple test image
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    
    # Create a pattern
    cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), -1)  # White square
    cv2.rectangle(img, (180, 50), (280, 150), (255, 0, 0), -1)    # Red square
    cv2.rectangle(img, (50, 180), (150, 280), (0, 255, 0), -1)    # Green square
    cv2.rectangle(img, (180, 180), (280, 280), (0, 0, 255), -1)   # Blue square
    
    # Add some noise
    noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    
    print("Original image created with noise and patterns")
    
    # Example 1: Gaussian smoothing
    result1 = apply_linear_filter(
        img_rgb=img,
        filter_name="Gaussian",
        ksize=15,
        sigma=2.0,
        border_mode_str="reflect",
        grayscale_only=False,
        iterations=1,
        unsharp_alpha=1.0
    )
    cv2.imwrite("example_gaussian.png", cv2.cvtColor(result1, cv2.COLOR_RGB2BGR))
    print("✓ Gaussian filter applied and saved as 'example_gaussian.png'")
    
    # Example 2: Edge detection with Sobel X
    result2 = apply_linear_filter(
        img_rgb=img,
        filter_name="Sobel X",
        ksize=3,
        sigma=1.0,
        border_mode_str="reflect",
        grayscale_only=True,  # Convert to grayscale for edge detection
        iterations=1,
        unsharp_alpha=1.0
    )
    cv2.imwrite("example_sobel_x.png", cv2.cvtColor(result2, cv2.COLOR_RGB2BGR))
    print("✓ Sobel X filter applied and saved as 'example_sobel_x.png'")
    
    # Example 3: Unsharp masking for sharpening
    result3 = apply_linear_filter(
        img_rgb=img,
        filter_name="Unsharp",
        ksize=5,
        sigma=1.0,
        border_mode_str="reflect",
        grayscale_only=False,
        iterations=1,
        unsharp_alpha=2.0  # Strong sharpening
    )
    cv2.imwrite("example_unsharp.png", cv2.cvtColor(result3, cv2.COLOR_RGB2BGR))
    print("✓ Unsharp masking applied and saved as 'example_unsharp.png'")
    
    # Example 4: Multiple iterations of box filter
    result4 = apply_linear_filter(
        img_rgb=img,
        filter_name="Box/Average",
        ksize=7,
        sigma=1.0,
        border_mode_str="reflect",
        grayscale_only=False,
        iterations=3,  # Apply filter 3 times
        unsharp_alpha=1.0
    )
    cv2.imwrite("example_box_multiple.png", cv2.cvtColor(result4, cv2.COLOR_RGB2BGR))
    print("✓ Box filter (3 iterations) applied and saved as 'example_box_multiple.png'")
    
    # Save original for comparison
    cv2.imwrite("example_original.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print("✓ Original image saved as 'example_original.png'")
    
    print("\nExample completed! Check the generated images to see the filter effects.")

if __name__ == "__main__":
    example_filter_usage()
