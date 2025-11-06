"""
Helper functions for retinal image processing based on MDPI paper methodology
Reference: Journal of Imaging (jimaging-08-00291)

This implementation combines techniques from the paper including:
- Green channel extraction and preprocessing
- CLAHE enhancement
- Morphological operations
- Frangi vesselness filtering
- Advanced post-processing
"""

import cv2
import numpy as np
import scipy.ndimage
from skimage.filters import frangi
from skimage.util import img_as_float
import matplotlib.pyplot as plt


def extract_green_channel(img_path):
    """
    Extract and preprocess the green channel from retinal fundus image.
    The green channel provides the best vessel contrast.
    
    Args:
        img_path: Path to the retinal image
        
    Returns:
        original_rgb: Original image in RGB
        green_channel: Extracted green channel
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    
    # Convert BGR to RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Extract green channel (index 1 in BGR format)
    green_channel = img[:, :, 1]
    
    return img_rgb, green_channel


def apply_shade_correction(green_channel, kernel_size=51):
    """
    Apply shade correction to compensate for uneven illumination.
    Based on paper's preprocessing methodology using morphological opening.
    
    This removes large-scale variations in illumination by estimating
    the background and normalizing the image.
    
    Args:
        green_channel: Green channel image
        kernel_size: Size of morphological kernel (should be large, e.g., 51)
        
    Returns:
        corrected: Shade-corrected image (0-255 range)
    """

    img = cv2.addWeighted(green_channel, 0.5, ~cv2.GaussianBlur(green_channel, (221, 221), 0), 0.5, 0)
    
    return img


def enhance_contrast_clahe(image, clip_limit=2.0, tile_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        image: Input grayscale image
        clip_limit: Contrast limiting threshold
        tile_size: Size of grid for histogram equalization
        
    Returns:
        enhanced: CLAHE enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    enhanced = clahe.apply(image)
    
    return enhanced


def denoise_image(image, method='bilateral', **kwargs):
    """
    Apply denoising while preserving edges.
    
    Args:
        image: Input image
        method: 'bilateral', 'gaussian', or 'median'
        **kwargs: Additional parameters for the chosen method
        
    Returns:
        denoised: Denoised image
    """
    if method == 'bilateral':
        d = kwargs.get('d', 9)
        sigma_color = kwargs.get('sigma_color', 75)
        sigma_space = kwargs.get('sigma_space', 75)
        denoised = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    elif method == 'gaussian':
        kernel_size = kwargs.get('kernel_size', 5)
        denoised = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif method == 'median':
        kernel_size = kwargs.get('kernel_size', 5)
        denoised = cv2.medianBlur(image, kernel_size)
    else:
        denoised = image
    
    return denoised


def segment_vessels_frangi_paper(enhanced_img, 
                                  sigma_range=(1, 10),
                                  alpha=0.5,
                                  beta=0.5,
                                  gamma=15):
    """
    Vessel segmentation using Frangi vesselness filter.
    Based on paper's multi-scale approach.
    
    Args:
        enhanced_img: Preprocessed and enhanced image
        sigma_range: Range of vessel scales to detect (min, max)
        alpha: Sensitivity to plate-like structures
        beta: Sensitivity to blob-like structures  
        gamma: Hessian normalization factor
        
    Returns:
        vessels_binary: Binary vessel mask
        frangi_response: Raw Frangi filter response
    """
    # Normalize to [0, 1] for Frangi filter
    img_normalized = img_as_float(enhanced_img)
    
    # Apply Frangi filter (multi-scale vessel enhancement)
    frangi_response = frangi(
        img_normalized,
        sigmas=range(sigma_range[0], sigma_range[1]),
        black_ridges=True,  # Vessels are dark on bright background
        alpha=alpha,
        beta=beta,
        gamma=gamma
    )
    
    # Convert back to uint8
    frangi_8bit = (frangi_response * 255).astype(np.uint8)
    
    # Apply Otsu thresholding
    _, vessels_binary = cv2.threshold(
        frangi_8bit, 
        0, 
        255, 
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    return vessels_binary, frangi_8bit


def segment_vessels_morphological(enhanced_img,
                                   kernel_size=7,
                                   threshold_method='otsu'):
    """
    Vessel segmentation using morphological operations.
    Alternative to Frangi filter based on paper's methodology.
    
    Args:
        enhanced_img: Preprocessed image
        kernel_size: Size of morphological kernel
        threshold_method: 'otsu', 'adaptive', or 'manual'
        
    Returns:
        vessels_binary: Binary vessel mask
        blackhat: Black-hat transform result
    """
    # Median blur to reduce noise
    denoised = cv2.medianBlur(enhanced_img, 5)
    
    # Black-hat morphological transform to extract dark vessels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(denoised, cv2.MORPH_BLACKHAT, kernel)
    
    # Enhance with CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    blackhat_enhanced = clahe.apply(blackhat)
    
    # Apply thresholding
    if threshold_method == 'otsu':
        _, vessels_binary = cv2.threshold(
            blackhat_enhanced,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    elif threshold_method == 'adaptive':
        vessels_binary = cv2.adaptiveThreshold(
            blackhat_enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            15,
            -2
        )
    else:
        _, vessels_binary = cv2.threshold(blackhat_enhanced, 15, 255, cv2.THRESH_BINARY)
    
    return vessels_binary, blackhat


def postprocess_vessel_mask(vessel_mask,
                             min_area=30,
                             max_area=10000,
                             morphology_iterations=2):
    """
    Post-process vessel segmentation to remove noise and artifacts.
    Based on paper's refinement strategy.
    
    Args:
        vessel_mask: Binary vessel mask
        min_area: Minimum component area to keep
        max_area: Maximum component area (to remove optic disc)
        morphology_iterations: Number of morphological cleaning iterations
        
    Returns:
        cleaned_mask: Cleaned binary vessel mask
    """
    # Morphological opening to remove small noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(
        vessel_mask,
        cv2.MORPH_OPEN,
        kernel_open,
        iterations=morphology_iterations
    )
    
    # Morphological closing to connect broken vessels
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        kernel_close,
        iterations=morphology_iterations
    )
    
    # Connected component analysis for size filtering
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        cleaned,
        connectivity=8
    )
    
    # Create output mask
    output_mask = np.zeros_like(cleaned)
    
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Keep components within size range
        if min_area < area < max_area:
            output_mask[labels == i] = 255
    
    return output_mask


def create_fov_mask(green_channel, erosion_iterations=10):
    """
    Create field of view (FOV) mask to remove circular border artifacts.
    
    Args:
        green_channel: Green channel image
        erosion_iterations: Number of erosion iterations to shrink mask
        
    Returns:
        fov_mask: Binary FOV mask
    """
    # Threshold to get approximate FOV
    _, fov_mask = cv2.threshold(
        green_channel,
        10,
        255,
        cv2.THRESH_BINARY
    )
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fov_mask = cv2.morphologyEx(fov_mask, cv2.MORPH_CLOSE, kernel, iterations=5)
    fov_mask = cv2.morphologyEx(fov_mask, cv2.MORPH_OPEN, kernel, iterations=3)
    
    # Erode to remove border effects
    fov_mask = cv2.erode(fov_mask, kernel, iterations=erosion_iterations)
    
    return fov_mask


def apply_fov_mask(vessel_mask, fov_mask):
    """
    Apply FOV mask to vessel segmentation.
    
    Args:
        vessel_mask: Binary vessel mask
        fov_mask: Binary FOV mask
        
    Returns:
        masked_vessels: Vessel mask with FOV applied
    """
    masked_vessels = cv2.bitwise_and(vessel_mask, vessel_mask, mask=fov_mask)
    return masked_vessels


def visualize_preprocessing_pipeline(img_path, save_path=None):
    """
    Visualize the complete preprocessing pipeline.
    
    Args:
        img_path: Path to input image
        save_path: Optional path to save visualization
        
    Returns:
        fig: Matplotlib figure object
    """
    # Step 1: Extract green channel
    img_rgb, green = extract_green_channel(img_path)
    
    # Step 2: Shade correction
    shade_corrected = apply_shade_correction(green)
    
    # Step 3: CLAHE enhancement
    clahe_enhanced = enhance_contrast_clahe(shade_corrected)
    
    # Step 4: Denoising
    denoised = denoise_image(clahe_enhanced, method='bilateral')
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Preprocessing Pipeline', fontsize=16, fontweight='bold')
    
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('1. Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(green, cmap='gray')
    axes[0, 1].set_title('2. Green Channel')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(shade_corrected, cmap='gray')
    axes[0, 2].set_title('3. Shade Corrected')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(clahe_enhanced, cmap='gray')
    axes[1, 0].set_title('4. CLAHE Enhanced')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(denoised, cmap='gray')
    axes[1, 1].set_title('5. Denoised')
    axes[1, 1].axis('off')
    
    # Histogram comparison
    axes[1, 2].hist(green.ravel(), 256, [0, 256], alpha=0.5, label='Original')
    axes[1, 2].hist(clahe_enhanced.ravel(), 256, [0, 256], alpha=0.5, label='CLAHE')
    axes[1, 2].set_title('6. Histogram Comparison')
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def complete_vessel_segmentation_pipeline(img_path,
                                           method='frangi',
                                           visualize=False):
    """
    Complete vessel segmentation pipeline from raw image to final mask.
    Implements the paper's full methodology.
    
    Args:
        img_path: Path to retinal image
        method: 'frangi' or 'morphological'
        visualize: Whether to show intermediate results
        
    Returns:
        final_vessels: Final binary vessel mask
        intermediate_results: Dictionary with intermediate steps
    """
    # Preprocessing
    img_rgb, green = extract_green_channel(img_path)
    shade_corrected = apply_shade_correction(green)
    clahe_enhanced = enhance_contrast_clahe(shade_corrected)
    denoised = denoise_image(clahe_enhanced, method='bilateral')
    
    # Vessel segmentation
    if method == 'frangi':
        vessels_raw, frangi_response = segment_vessels_frangi_paper(denoised)
        intermediate = frangi_response
    else:
        vessels_raw, blackhat = segment_vessels_morphological(denoised)
        intermediate = blackhat
    
    # Post-processing
    vessels_cleaned = postprocess_vessel_mask(vessels_raw)
    
    # FOV masking
    fov_mask = create_fov_mask(green)
    final_vessels = apply_fov_mask(vessels_cleaned, fov_mask)
    
    # Store intermediate results
    results = {
        'original': img_rgb,
        'green_channel': green,
        'shade_corrected': shade_corrected,
        'clahe_enhanced': clahe_enhanced,
        'denoised': denoised,
        'intermediate': intermediate,
        'vessels_raw': vessels_raw,
        'vessels_cleaned': vessels_cleaned,
        'fov_mask': fov_mask,
        'final_vessels': final_vessels
    }
    
    if visualize:
        visualize_segmentation_results(results, method)
    
    return final_vessels, results


def visualize_segmentation_results(results, method_name=''):
    """
    Visualize complete segmentation results.
    
    Args:
        results: Dictionary of intermediate results
        method_name: Name of segmentation method used
    """
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f'Vessel Segmentation Pipeline - {method_name}', 
                 fontsize=16, fontweight='bold')
    
    images = [
        ('original', 'Original', None),
        ('green_channel', 'Green Channel', 'gray'),
        ('shade_corrected', 'Shade Corrected', 'gray'),
        ('clahe_enhanced', 'CLAHE Enhanced', 'gray'),
        ('denoised', 'Denoised', 'gray'),
        ('intermediate', f'{method_name} Response', 'gray'),
        ('vessels_raw', 'Raw Vessels', 'gray'),
        ('vessels_cleaned', 'Cleaned Vessels', 'gray'),
        ('fov_mask', 'FOV Mask', 'gray'),
        ('final_vessels', 'Final Result', 'gray'),
    ]
    
    for idx, (key, title, cmap) in enumerate(images):
        if idx >= 12:
            break
        row, col = idx // 4, idx % 4
        if key in results and results[key] is not None:
            axes[row, col].imshow(results[key], cmap=cmap)
            axes[row, col].set_title(title, fontsize=10)
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for idx in range(len(images), 12):
        row, col = idx // 4, idx % 4
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
