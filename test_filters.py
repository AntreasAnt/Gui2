#!/usr/bin/env python3
# test_filters.py
"""
Test script for the linear filters implementation.
Creates a simple test image and applies all filters.
"""

import numpy as np
import cv2
from linear_filters import apply_linear_filter

def create_test_image():
    """Create a simple test image with patterns."""
    # Create a 200x200 test image
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # Add some patterns
    # White square
    img[50:100, 50:100] = [255, 255, 255]
    
    # Red square
    img[120:170, 50:100] = [255, 0, 0]
    
    # Green square
    img[50:100, 120:170] = [0, 255, 0]
    
    # Blue square
    img[120:170, 120:170] = [0, 0, 255]
    
    # Add some noise
    noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    
    return img

def test_all_filters():
    """Test all implemented filters."""
    print("Creating test image...")
    test_img = create_test_image()
    
    # Save original test image
    cv2.imwrite("test_original.png", cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))
    print("Saved: test_original.png")
    
    filters_to_test = [
        ("Box/Average", {"ksize": 5}),
        ("Gaussian", {"ksize": 5, "sigma": 1.0}),
        ("Sobel X", {}),
        ("Sobel Y", {}),
        ("Laplacian", {}),
        ("Unsharp", {"ksize": 5, "sigma": 1.0, "unsharp_alpha": 1.5})
    ]
    
    for filter_name, extra_params in filters_to_test:
        print(f"Testing {filter_name} filter...")
        
        # Default parameters
        params = {
            "img_rgb": test_img,
            "filter_name": filter_name,
            "ksize": 5,
            "sigma": 1.0,
            "border_mode_str": "reflect",
            "grayscale_only": False,
            "iterations": 1,
            "unsharp_alpha": 1.0
        }
        
        # Update with filter-specific parameters
        params.update(extra_params)
        
        try:
            result = apply_linear_filter(**params)
            
            # Save result
            filename = f"test_{filter_name.lower().replace('/', '_').replace(' ', '_')}.png"
            cv2.imwrite(filename, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            print(f"  ✓ Saved: {filename}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\nTest completed! Check the generated images.")

if __name__ == "__main__":
    test_all_filters()
