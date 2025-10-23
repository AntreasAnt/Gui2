# linear_filters.py
# -*- coding: utf-8 -*-
"""
Linear Filters Implementation
----------------------------
This module implements various linear spatial filters for image processing.
"""

import cv2
import numpy as np
from typing import Optional

def get_border_type(border_mode_str: str) -> int:
    """Convert border mode string to OpenCV constant."""
    border_map = {
        'reflect': cv2.BORDER_REFLECT,
        'replicate': cv2.BORDER_REPLICATE,
        'constant': cv2.BORDER_CONSTANT
    }
    return border_map.get(border_mode_str, cv2.BORDER_REFLECT)

def apply_box_filter(img: np.ndarray, ksize: int, border_type: int) -> np.ndarray:
    """Apply Box/Average filter."""
    return cv2.blur(img, (ksize, ksize), borderType=border_type)

def apply_gaussian_filter(img: np.ndarray, ksize: int, sigma: float, border_type: int) -> np.ndarray:
    """Apply Gaussian filter."""
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, borderType=border_type)

def apply_sobel_x_filter(img: np.ndarray, border_type: int) -> np.ndarray:
    """Apply Sobel X filter for edge detection."""
    if len(img.shape) == 3:
        # For color images, apply to each channel
        result = np.zeros_like(img, dtype=np.float64)
        for i in range(img.shape[2]):
            sobel = cv2.Sobel(img[:, :, i], cv2.CV_64F, 1, 0, borderType=border_type)
            result[:, :, i] = cv2.convertScaleAbs(sobel)
    else:
        # Grayscale image
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, borderType=border_type)
        result = cv2.convertScaleAbs(sobel)
    
    return result.astype(np.uint8)

def apply_sobel_y_filter(img: np.ndarray, border_type: int) -> np.ndarray:
    """Apply Sobel Y filter for edge detection."""
    if len(img.shape) == 3:
        # For color images, apply to each channel
        result = np.zeros_like(img, dtype=np.float64)
        for i in range(img.shape[2]):
            sobel = cv2.Sobel(img[:, :, i], cv2.CV_64F, 0, 1, borderType=border_type)
            result[:, :, i] = cv2.convertScaleAbs(sobel)
    else:
        # Grayscale image
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, borderType=border_type)
        result = cv2.convertScaleAbs(sobel)
    
    return result.astype(np.uint8)

def apply_laplacian_filter(img: np.ndarray, border_type: int) -> np.ndarray:
    """Apply Laplacian filter for edge detection."""
    if len(img.shape) == 3:
        # For color images, apply to each channel
        result = np.zeros_like(img, dtype=np.float64)
        for i in range(img.shape[2]):
            laplacian = cv2.Laplacian(img[:, :, i], cv2.CV_64F, borderType=border_type)
            result[:, :, i] = cv2.convertScaleAbs(laplacian)
    else:
        # Grayscale image
        laplacian = cv2.Laplacian(img, cv2.CV_64F, borderType=border_type)
        result = cv2.convertScaleAbs(laplacian)
    
    return result.astype(np.uint8)

def apply_unsharp_filter(img: np.ndarray, ksize: int, sigma: float, alpha: float, border_type: int) -> np.ndarray:
    """Apply Unsharp Masking filter for sharpening."""
    # Create blurred version
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, borderType=border_type)
    
    # Calculate sharpened image: sharp = img + α·(img - blur)
    img_float = img.astype(np.float64)
    blurred_float = blurred.astype(np.float64)
    
    # Unsharp mask: img + alpha * (img - blur)
    sharp = img_float + alpha * (img_float - blurred_float)
    
    # Clip values to valid range [0, 255]
    sharp = np.clip(sharp, 0, 255)
    
    return sharp.astype(np.uint8)

def apply_linear_filter(
    img_rgb: np.ndarray,
    filter_name: str,
    ksize: int,
    sigma: float,
    border_mode_str: str,
    grayscale_only: bool,
    iterations: int,
    unsharp_alpha: float
) -> np.ndarray:
    """
    Apply linear spatial filter to an RGB image.
    
    Parameters:
    -----------
    img_rgb : np.ndarray
        Input RGB image (H×W×3, uint8)
    filter_name : str
        Name of the filter to apply
    ksize : int
        Kernel size (must be odd)
    sigma : float
        Standard deviation for Gaussian filter
    border_mode_str : str
        Border handling mode ('reflect', 'replicate', 'constant')
    grayscale_only : bool
        If True, convert to grayscale before processing
    iterations : int
        Number of times to apply the filter
    unsharp_alpha : float
        Alpha parameter for unsharp masking
    
    Returns:
    --------
    np.ndarray
        Filtered image (H×W×3, uint8)
    """
    # Ensure kernel size is odd
    if ksize % 2 == 0:
        ksize += 1
    
    # Get border type
    border_type = get_border_type(border_mode_str)
    
    # Work with a copy
    result = img_rgb.copy()
    
    # Convert to grayscale if requested
    if grayscale_only:
        gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        working_img = gray
    else:
        working_img = result
    
    # Apply filter multiple times if requested
    for _ in range(iterations):
        if filter_name == "Box/Average":
            working_img = apply_box_filter(working_img, ksize, border_type)
        
        elif filter_name == "Gaussian":
            working_img = apply_gaussian_filter(working_img, ksize, sigma, border_type)
        
        elif filter_name == "Sobel X":
            working_img = apply_sobel_x_filter(working_img, border_type)
        
        elif filter_name == "Sobel Y":
            working_img = apply_sobel_y_filter(working_img, border_type)
        
        elif filter_name == "Laplacian":
            working_img = apply_laplacian_filter(working_img, border_type)
        
        elif filter_name == "Unsharp":
            working_img = apply_unsharp_filter(working_img, ksize, sigma, unsharp_alpha, border_type)
        
        else:
            raise ValueError(f"Unknown filter: {filter_name}")
    
    # Convert back to RGB if we were working with grayscale
    if grayscale_only:
        # Convert grayscale back to RGB
        if len(working_img.shape) == 2:
            result = cv2.cvtColor(working_img, cv2.COLOR_GRAY2RGB)
        else:
            result = working_img
    else:
        result = working_img
    
    return result.astype(np.uint8)
