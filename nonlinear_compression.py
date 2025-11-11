# nonlinear_compression.py
# -*- coding: utf-8 -*-
"""
Non-linear Filters and Compression Implementation
TODO: Students should implement these functions
"""
import cv2
import numpy as np
from skimage.metrics import structural_similarity


def apply_nonlinear_filter(img_rgb: np.ndarray, kind: str, **params) -> np.ndarray:
    """
    Apply non-linear filter to RGB image.
    
    Args:
        img_rgb: Input RGB image (numpy array)
        kind: Filter type - "Median", "Bilateral", or "NLMeans"
        **params: ksize, bilateral_d, bilateral_sigma_color, bilateral_sigma_space,
                  nlmeans_h
    
    Returns:
        Filtered RGB image (numpy array, uint8)
    """
    if kind == "Median":
        ksize = params.get('ksize', 5)
        # Ensure odd kernel size
        if ksize % 2 == 0:
            ksize += 1
        result = cv2.medianBlur(img_rgb, ksize)
        return result
    
    elif kind == "Bilateral":
        d = params.get('bilateral_d', 7)
        sigma_color = params.get('bilateral_sigma_color', 75.0)
        sigma_space = params.get('bilateral_sigma_space', 75.0)
        result = cv2.bilateralFilter(img_rgb, d, sigma_color, sigma_space)
        return result
    
    elif kind == "NLMeans":
        h = params.get('nlmeans_h', 10.0)
        result = cv2.fastNlMeansDenoisingColored(img_rgb, None, h, h, 7, 21)
        return result
    
    else:
        raise ValueError(f"Unknown filter type: {kind}")


def compress_image(img_rgb: np.ndarray, codec: str, quality: int) -> tuple:
    """
    Compress image using specified codec.
    
    Args:
        img_rgb: Input RGB image
        codec: "JPEG", "PNG", "WebP", or "TIFF-LZW"
        quality: Quality parameter (1-100)
    
    Returns:
        (compressed_rgb, encoded_bytes): Decompressed RGB image and compressed bytes
    """
    # Convert RGB to BGR for OpenCV
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    if codec == "JPEG":
        success, encoded = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not success:
            raise ValueError("JPEG encoding failed")
        decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        compressed_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
        return compressed_rgb, encoded.tobytes()
    
    elif codec == "PNG":
        # PNG quality is 0-9 (compression level), map 1-100 to 9-0
        compression = int(9 - (quality - 1) * 8 / 99)
        success, encoded = cv2.imencode('.png', bgr, [cv2.IMWRITE_PNG_COMPRESSION, compression])
        if not success:
            raise ValueError("PNG encoding failed")
        decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        compressed_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
        return compressed_rgb, encoded.tobytes()
    
    elif codec == "WebP":
        success, encoded = cv2.imencode('.webp', bgr, [cv2.IMWRITE_WEBP_QUALITY, quality])
        if not success:
            raise ValueError("WebP encoding failed")
        decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        compressed_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
        return compressed_rgb, encoded.tobytes()
    
    elif codec == "TIFF-LZW":
        # TIFF with LZW compression (value 5)
        success, encoded = cv2.imencode('.tiff', bgr, [cv2.IMWRITE_TIFF_COMPRESSION, 5])
        if not success:
            raise ValueError("TIFF-LZW encoding failed")
        decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        compressed_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
        return compressed_rgb, encoded.tobytes()
    
    else:
        raise ValueError(f"Unknown codec: {codec}")


def compute_diff_image(orig_rgb: np.ndarray, comp_rgb: np.ndarray) -> np.ndarray:
    """
    Compute difference heatmap between original and compressed images.
    
    Args:
        orig_rgb: Original RGB image
        comp_rgb: Compressed RGB image
    
    Returns:
        Difference heatmap as RGB image
    """
    # Compute absolute difference
    diff = cv2.absdiff(orig_rgb, comp_rgb)
    
    # Convert to grayscale
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    
    # Apply colormap for visualization
    heatmap = cv2.applyColorMap(gray_diff, cv2.COLORMAP_JET)
    
    # Convert back to RGB
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def compute_metrics(orig_rgb: np.ndarray, comp_rgb: np.ndarray, encoded_bytes: bytes, codec: str) -> str:
    """
    Compute quality metrics and return HTML string.
    
    Metrics to compute:
    - μ Mean, σ Standard Deviation, ℳ Median, ⚙️ Entropy
    - Ὃ Bits per Pixel (BPP)
    - Ὄ Compression Ratio (CR)
    - ὒ PSNR
    - SSIM
    
    Args:
        orig_rgb: Original image
        comp_rgb: Compressed image
        encoded_bytes: Compressed file bytes
        codec: Compression codec used
    
    Returns:
        HTML formatted string with metrics
    """
    # Basic statistics
    mean_val = np.mean(comp_rgb)
    std_val = np.std(comp_rgb)
    median_val = np.median(comp_rgb)
    
    # Entropy
    hist, _ = np.histogram(comp_rgb.ravel(), bins=256, range=(0, 256))
    hist = hist / (hist.sum() + 1e-10)
    hist = hist[hist > 0]  # Remove zeros to avoid log(0)
    entropy = -np.sum(hist * np.log2(hist))
    
    # BPP (Bits Per Pixel)
    total_pixels = orig_rgb.shape[0] * orig_rgb.shape[1]
    bpp = (len(encoded_bytes) * 8) / total_pixels
    
    # Compression Ratio
    orig_size = total_pixels * 3  # RGB = 3 bytes per pixel
    cr = orig_size / len(encoded_bytes)
    
    # PSNR
    psnr = cv2.PSNR(orig_rgb, comp_rgb)
    
    # SSIM
    ssim_val = structural_similarity(orig_rgb, comp_rgb, multichannel=True, channel_axis=2)
    
    # Format HTML
    html = f"""
    <h3 style="color: #f1c57a; margin-top: 0;">Compression Metrics</h3>
    <p><b>Codec:</b> {codec} | <b>File Size:</b> {len(encoded_bytes):,} bytes</p>
    <table style="width:100%; border-spacing: 8px;">
    <tr><td><b>μ Mean:</b></td><td>{mean_val:.2f}</td></tr>
    <tr><td><b>σ Std Dev:</b></td><td>{std_val:.2f}</td></tr>
    <tr><td><b>ℳ Median:</b></td><td>{median_val:.2f}</td></tr>
    <tr><td><b>⚙️ Entropy:</b></td><td>{entropy:.2f} bits</td></tr>
    <tr><td><b>Ὃ BPP:</b></td><td>{bpp:.3f} bits/pixel</td></tr>
    <tr><td><b>Ὄ CR:</b></td><td>{cr:.2f}:1</td></tr>
    <tr><td><b>ὒ PSNR:</b></td><td style="color: {'#4CAF50' if psnr > 30 else '#FFC107' if psnr > 25 else '#F44336'};">{psnr:.2f} dB</td></tr>
    <tr><td><b>SSIM:</b></td><td style="color: {'#4CAF50' if ssim_val > 0.95 else '#FFC107' if ssim_val > 0.90 else '#F44336'};">{ssim_val:.4f}</td></tr>
    </table>
    <p style="font-size: 9pt; color: #888; margin-top: 10px;">
    <i>Original: {orig_size:,} bytes | Compressed: {len(encoded_bytes):,} bytes | Saved: {orig_size - len(encoded_bytes):,} bytes ({100*(1-len(encoded_bytes)/orig_size):.1f}%)</i>
    </p>
    """
    
    return html
