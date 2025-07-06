import numpy as np
from matplotlib import pyplot as plt
import scipy.ndimage as ndimage
from scipy import ndimage as ndi
# Molecular Position Detection using Blob Detection

from skimage.filters import gaussian
import os
import json


def gradient_contrast_filter(image, sigma=1.0, alpha=0.5):
    """
    Apply local gradient contrast enhancement to STM images.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input STM image (2D array)
    sigma : float
        Standard deviation for Gaussian smoothing (default: 1.0)
    alpha : float
        Enhancement strength (0-1, default: 0.5)
        
    Returns:
    --------
    enhanced_image : numpy.ndarray
        Enhanced image with improved local contrast
    """
    # Ensure image is float
    img = image.astype(np.float64)
    
    # Remove NaN values by interpolation or setting to mean
    if np.any(np.isnan(img)):
        mask = ~np.isnan(img)
        img[~mask] = np.nanmean(img)
    
    # Calculate gradients
    grad_x = ndimage.sobel(img, axis=1)
    grad_y = ndimage.sobel(img, axis=0)
    
    # Calculate gradient magnitude
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Apply Gaussian smoothing to reduce noise
    smoothed_grad = ndimage.gaussian_filter(grad_magnitude, sigma=sigma)
    
    # Normalize gradient magnitude
    if smoothed_grad.max() > 0:
        normalized_grad = smoothed_grad / smoothed_grad.max()
    else:
        normalized_grad = smoothed_grad
    
    # Enhance contrast based on local gradient
    enhanced = img + alpha * normalized_grad * (img - ndimage.gaussian_filter(img, sigma=sigma))
    
    return enhanced

def adaptive_gradient_filter(image, window_size=5, enhancement_factor=1.5):
    """
    Apply adaptive gradient-based contrast enhancement.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input STM image
    window_size : int
        Size of local window for adaptive enhancement
    enhancement_factor : float
        Factor to control enhancement strength
    """
    img = image.astype(np.float64)
    
    # Handle NaN values
    if np.any(np.isnan(img)):
        mask = ~np.isnan(img)
        img[~mask] = np.nanmean(img)
    
    # Calculate local standard deviation
    kernel = np.ones((window_size, window_size)) / (window_size**2)
    local_mean = ndimage.convolve(img, kernel, mode='reflect')
    local_sq_mean = ndimage.convolve(img**2, kernel, mode='reflect')
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))
    
    # Calculate gradients
    grad_x = ndimage.sobel(img, axis=1)
    grad_y = ndimage.sobel(img, axis=0)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Adaptive enhancement based on local statistics
    enhancement_map = enhancement_factor * grad_magnitude / (local_std + 1e-8)
    enhanced = img + enhancement_map * (img - local_mean)


def line_filter(image, direction='horizontal', filter_type='mean'):
    """
    Apply line-by-line filtering to remove scan artifacts in STM images.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input STM image (2D array)
    direction : str
        Direction to apply filter ('horizontal' or 'vertical')
    filter_type : str
        Type of filter ('mean', 'median', 'polynomial')
        
    Returns:
    --------
    filtered_image : numpy.ndarray
        Image with line artifacts removed
    """
    img = image.astype(np.float64)
    
    # Handle NaN values
    if np.any(np.isnan(img)):
        mask = ~np.isnan(img)
        img[~mask] = np.nanmean(img)
    
    filtered = img.copy()
    
    if direction == 'horizontal':
        # Process each row
        for i in range(img.shape[0]):
            line = img[i, :]
            
            if filter_type == 'mean':
                # Subtract mean of each line
                filtered[i, :] = line - np.nanmean(line)
            elif filter_type == 'median':
                # Subtract median of each line
                filtered[i, :] = line - np.nanmedian(line)
            elif filter_type == 'polynomial':
                # Fit and subtract polynomial trend
                x = np.arange(len(line))
                valid_mask = ~np.isnan(line)
                if np.sum(valid_mask) > 3:  # Need at least 4 points for cubic fit
                    poly_coeffs = np.polyfit(x[valid_mask], line[valid_mask], deg=3)
                    poly_trend = np.polyval(poly_coeffs, x)
                    filtered[i, :] = line - poly_trend
                else:
                    filtered[i, :] = line - np.nanmean(line)
                    
    elif direction == 'vertical':
        # Process each column
        for j in range(img.shape[1]):
            line = img[:, j]
            
            if filter_type == 'mean':
                filtered[:, j] = line - np.nanmean(line)
            elif filter_type == 'median':
                filtered[:, j] = line - np.nanmedian(line)
            elif filter_type == 'polynomial':
                x = np.arange(len(line))
                valid_mask = ~np.isnan(line)
                if np.sum(valid_mask) > 3:
                    poly_coeffs = np.polyfit(x[valid_mask], line[valid_mask], deg=3)
                    poly_trend = np.polyval(poly_coeffs, x)
                    filtered[:, j] = line - poly_trend
                else:
                    filtered[:, j] = line - np.nanmean(line)
    
    return filtered

def polynomial_background_subtraction(image, degree=2, mask_threshold=None):
    """
    Remove polynomial background from STM images.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input STM image (2D array)
    degree : int
        Degree of polynomial surface to fit and subtract
    mask_threshold : float or None
        If provided, exclude pixels with values above this threshold from fitting
        
    Returns:
    --------
    corrected_image : numpy.ndarray
        Image with polynomial background removed
    background : numpy.ndarray
        The fitted polynomial background
    """
    img = image.astype(np.float64)
    
    # Handle NaN values
    if np.any(np.isnan(img)):
        mask = ~np.isnan(img)
        img[~mask] = np.nanmean(img)
    
    # Create coordinate grids
    y, x = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    
    # Flatten arrays for fitting
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = img.flatten()
    
    # Create mask for fitting (exclude NaN and optionally high values)
    fit_mask = ~np.isnan(z_flat)
    if mask_threshold is not None:
        fit_mask &= (z_flat <= mask_threshold)
    
    # Create polynomial terms
    if degree == 1:
        # Linear: z = a + bx + cy
        A = np.column_stack([
            np.ones(np.sum(fit_mask)),
            x_flat[fit_mask],
            y_flat[fit_mask]
        ])
    elif degree == 2:
        # Quadratic: z = a + bx + cy + dx² + exy + fy²
        A = np.column_stack([
            np.ones(np.sum(fit_mask)),
            x_flat[fit_mask],
            y_flat[fit_mask],
            x_flat[fit_mask]**2,
            x_flat[fit_mask] * y_flat[fit_mask],
            y_flat[fit_mask]**2
        ])
    elif degree == 3:
        # Cubic: includes x³, x²y, xy², y³ terms
        A = np.column_stack([
            np.ones(np.sum(fit_mask)),
            x_flat[fit_mask],
            y_flat[fit_mask],
            x_flat[fit_mask]**2,
            x_flat[fit_mask] * y_flat[fit_mask],
            y_flat[fit_mask]**2,
            x_flat[fit_mask]**3,
            x_flat[fit_mask]**2 * y_flat[fit_mask],
            x_flat[fit_mask] * y_flat[fit_mask]**2,
            y_flat[fit_mask]**3
        ])
    else:
        raise ValueError("Degree must be 1, 2, or 3")
    
    # Fit polynomial using least squares
    coeffs, residuals, rank, s = np.linalg.lstsq(A, z_flat[fit_mask], rcond=None)
    
    # Calculate background for all pixels
    if degree == 1:
        background = (coeffs[0] + 
                     coeffs[1] * x_flat + 
                     coeffs[2] * y_flat)
    elif degree == 2:
        background = (coeffs[0] + 
                     coeffs[1] * x_flat + 
                     coeffs[2] * y_flat +
                     coeffs[3] * x_flat**2 + 
                     coeffs[4] * x_flat * y_flat + 
                     coeffs[5] * y_flat**2)
    elif degree == 3:
        background = (coeffs[0] + 
                     coeffs[1] * x_flat + 
                     coeffs[2] * y_flat +
                     coeffs[3] * x_flat**2 + 
                     coeffs[4] * x_flat * y_flat + 
                     coeffs[5] * y_flat**2 +
                     coeffs[6] * x_flat**3 + 
                     coeffs[7] * x_flat**2 * y_flat + 
                     coeffs[8] * x_flat * y_flat**2 + 
                     coeffs[9] * y_flat**3)
    
    # Reshape background to original image shape
    background = background.reshape(img.shape)
    
    # Subtract background
    corrected = img - background
    
    return corrected, background

def plane_subtraction(image):
    """
    Remove linear plane (tilt) from STM images.
    This is a simplified version of polynomial background subtraction with degree=1.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input STM image (2D array)
        
    Returns:
    --------
    corrected_image : numpy.ndarray
        Image with plane removed
    """
    corrected, _ = polynomial_background_subtraction(image, degree=1)
    return corrected

def advanced_line_filter(image, filter_strength=1.0, preserve_features=True):

    """
    Advanced line filter that preserves features while removing scan artifacts.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input STM image
    filter_strength : float
        Strength of the filtering (0-2, default: 1.0)
    preserve_features : bool
        Whether to preserve high-frequency features
        
    Returns:
    --------
    filtered_image : numpy.ndarray
        Filtered image
    """
    img = image.astype(np.float64)
    
    # Handle NaN values
    if np.any(np.isnan(img)):
        mask = ~np.isnan(img)
        img[~mask] = np.nanmean(img)
    
    # Apply horizontal line filter
    h_filtered = line_filter(img, direction='horizontal', filter_type='polynomial')
    
    if preserve_features:
        # Calculate feature map using gradient magnitude
        grad_x = ndimage.sobel(img, axis=1)
        grad_y = ndimage.sobel(img, axis=0)
        feature_map = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize feature map
        feature_map = feature_map / (feature_map.max() + 1e-8)
        
        # Blend original and filtered based on feature strength
        alpha = filter_strength * (1 - feature_map)
        filtered = alpha * h_filtered + (1 - alpha) * img
    else:
        filtered = filter_strength * h_filtered + (1 - filter_strength) * img
    
    return filtered

def get_2std_limits(image, nstd=2 ):

    """
    Get the ±2 standard deviation limits for an image.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
        
    Returns:
    --------
    vmin, vmax : float, float
        The lower and upper limits
    """
    img_mean = np.nanmean(image)
    img_std = np.nanstd(image)
    
    vmin = img_mean - 2 * nstd
    vmax = img_mean + 2 * nstd
    
    return vmin, vmax

def detect_molecules_blob_log(image, min_sigma=1, max_sigma=30, num_sigma=10, threshold=0.1, 
                             overlap=0.5, preprocess=True):
    """
    Detect molecular positions using Laplacian of Gaussian (LoG) blob detection.
    
    Parameters:
    -----------
    image : numpy.ndarray
        STM image
    min_sigma : float
        Minimum standard deviation for Gaussian kernel (smaller molecules)
    max_sigma : float  
        Maximum standard deviation for Gaussian kernel (larger molecules)
    num_sigma : int
        Number of intermediate values of standard deviations to consider
    threshold : float
        Threshold for blob detection (lower = more sensitive)
    overlap : float
        Maximum overlap between blobs (0-1)
    preprocess : bool
        Whether to apply preprocessing before detection
        
    Returns:
    --------
    blobs : numpy.ndarray
        Array of detected blobs with (y, x, radius)
    processed_image : numpy.ndarray
        Preprocessed image used for detection
    """
    # Preprocess image if requested
    if preprocess:
        # Apply standard STM preprocessing pipeline
        processed = image.astype(np.float64)
        
        # Handle NaN values
        if np.any(np.isnan(processed)):
            mask = ~np.isnan(processed)
            processed[~mask] = np.nanmean(processed)
        
        # Apply line filter and background subtraction
        processed = line_filter(processed, direction='horizontal', filter_type='polynomial')
        processed = polynomial_background_subtraction(processed, degree=2)[0]
        
        # Light gradient enhancement
        processed = gradient_contrast_filter(processed, sigma=2.0, alpha=0.5)
        
        # Gaussian blur to smooth noise while preserving molecular features
        processed = gaussian(processed, sigma=1.5)
    else:
        processed = image.copy()
        if np.any(np.isnan(processed)):
            mask = ~np.isnan(processed)
            processed[~mask] = np.nanmean(processed)
    
    # Normalize image for blob detection
    processed = (processed - processed.min()) / (processed.max() - processed.min())
    
    # Detect blobs using LoG
    blobs = blob_log(processed, min_sigma=min_sigma, max_sigma=max_sigma, 
                     num_sigma=num_sigma, threshold=threshold, overlap=overlap)
    
    # Convert sigma to radius (radius = sigma * sqrt(2))
    blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
    
    return blobs, processed

def detect_molecules_multiscale_log(image, scales=None, threshold=0.03, overlap=0.3, preprocess=True):
    """
    Enhanced multi-scale LoG detection with better parameter handling.
    
    Parameters:
    -----------
    image : numpy.ndarray
        STM image
    scales : list or None
        List of scales to search. If None, auto-determines from image statistics
    threshold : float
        Detection threshold
    overlap : float
        Maximum overlap between detections
    preprocess : bool
        Whether to preprocess
        
    Returns:
    --------
    blobs : numpy.ndarray
        Detected blobs (y, x, radius)
    processed : numpy.ndarray
        Processed image
    """
    if preprocess:
        processed = polynomial_background_subtraction(image, degree=2)[0]
        processed = line_filter(processed, direction='horizontal', filter_type='polynomial')
        processed = gradient_contrast_filter(processed, sigma=1.5, alpha=0.4)
        processed = gaussian(processed, sigma=1.0)
    else:
        processed = image.copy()
    
    # Auto-determine scales if not provided
    if scales is None:
        # Estimate molecule size from image statistics
        grad_x = ndimage.sobel(processed, axis=1)
        grad_y = ndimage.sobel(processed, axis=0)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Find characteristic length scale
        fft = np.fft.fft2(processed)
        power_spectrum = np.abs(fft)**2
        
        # Estimate dominant frequency
        freq_y, freq_x = np.unravel_index(np.argmax(power_spectrum[1:, 1:]), power_spectrum[1:, 1:].shape)
        char_length = min(processed.shape) / max(freq_x + 1, freq_y + 1)
        
        # Create scale range around characteristic length
        min_scale = max(1, char_length / 4)
        max_scale = min(30, char_length * 2)
        scales = np.logspace(np.log10(min_scale), np.log10(max_scale), 15)
    
    # Normalize
    processed = (processed - processed.min()) / (processed.max() - processed.min())

    
    # Apply multi-scale LoG
    blobs = blob_log(processed, min_sigma=min(scales), max_sigma=max(scales),
                     num_sigma=len(scales), threshold=threshold, overlap=overlap)
    
    # Convert sigma to radius
    if len(blobs) > 0:
        blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
    
    return blobs, processed


def plot_detected_molecules(image, blobs,figsize=(12, 8), show_original=False):
    """
    Visualize detected molecular positions on STM image.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Original STM image
    blobs : numpy.ndarray
        Detected blobs array with (y, x, radius)
    title : str
        Plot title
    method : str
        Detection method name for display
    figsize : tuple
        Figure size
    show_original : bool
        Whether to show side-by-side with original
    """
    if show_original:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Original image
        ax1.imshow(image, cmap='gray')
        ax1.set_title('Original STM Image')
        ax1.axis('off')
        
        # Image with detected molecules
        ax2.imshow(image, cmap='gray')
        
        # Add dots at molecule centers only
        for blob in blobs[:2]:
            y, x = blob
            ax2.plot(x, y, 'r+', markersize=8, markeredgewidth=2)
        
        ax2.set_title(f'{len(blobs)} molecules detected')
        ax2.axis('off')
        
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(image, cmap='gray')
        
        # Add dots at molecule centers only
        for blob in blobs[:2]:
            y, x = blob
            ax.plot(x, y, 'r+', markersize=8, markeredgewidth=2)
        
        ax.set_title(f'{len(blobs)} molecules detected')
        ax.axis('off')
    
    plt.tight_layout()
    
    return fig


def plot_detected_molecules_c(image, blobs, title="Detected Molecules", method="LoG", 
                           figsize=(12, 8), show_original=True):
    """
    Visualize detected molecular positions on STM image.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Original STM image
    blobs : numpy.ndarray
        Detected blobs array with (y, x, radius)
    title : str
        Plot title
    method : str
        Detection method name for display
    figsize : tuple
        Figure size
    show_original : bool
        Whether to show side-by-side with original
    """
    if show_original:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Original image
        ax1.imshow(image, cmap='gray')
        ax1.set_title('Original STM Image')
        ax1.axis('off')
        
        # Image with detected molecules
        ax2.imshow(image, cmap='gray')
        
        # Draw circles around detected molecules
        for blob in blobs:
            y, x, r = blob
            circle = patches.Circle((x, y), r, color='red', fill=False, linewidth=2, alpha=0.8)
            ax2.add_patch(circle)
            # Add small dot at center
            ax2.plot(x, y, 'r+', markersize=8, markeredgewidth=2)
        
        ax2.set_title(f'{title}\n{method}: {len(blobs)} molecules detected')
        ax2.axis('off')
        
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(image, cmap='gray')
        
        for blob in blobs:
            y, x, r = blob
            circle = patches.Circle((x, y), r, color='red', fill=False, linewidth=2, alpha=0.8)
            ax.add_patch(circle)
            ax.plot(x, y, 'r+', markersize=8, markeredgewidth=2)
        
        ax.set_title(f'{title}\n{method}: {len(blobs)} molecules detected')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def fft_lowpass_filter(image, cutoff_freq=0.1, filter_type='ideal'):
    """
    Apply low-pass filtering using FFT.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    cutoff_freq : float
        Cutoff frequency as fraction of Nyquist frequency (0-1)
    filter_type : str
        Type of filter ('ideal', 'butterworth', 'gaussian')
        
    Returns:
    --------
    filtered_image : numpy.ndarray
        Low-pass filtered image
    """
    # Get image dimensions
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2
    
    # Create frequency grid
    u = np.fft.fftfreq(rows).reshape(-1, 1)
    v = np.fft.fftfreq(cols)
    
    # Calculate distance from center in frequency domain
    D = np.sqrt(u**2 + v**2)
    
    # Normalize cutoff frequency
    D0 = cutoff_freq / 2  # Cutoff frequency
    
    # Create filter mask
    if filter_type == 'ideal':
        # Ideal low-pass filter (sharp cutoff)
        H = (D <= D0).astype(float)
        
    elif filter_type == 'butterworth':
        # Butterworth filter (smooth rolloff)
        n = 2  # Filter order
        H = 1 / (1 + (D / D0)**(2*n))
        
    elif filter_type == 'gaussian':
        # Gaussian filter (very smooth)
        H = np.exp(-(D**2) / (2 * D0**2))
    
    # Apply FFT
    fft_image = np.fft.fft2(image)
    
    # Apply filter in frequency domain
    filtered_fft = fft_image * H
    
    # Convert back to spatial domain
    filtered_image = np.real(np.fft.ifft2(filtered_fft))
    
    return filtered_image

def save_images_dict_for_mathematica(images_dict, filename):
    """
    Save dictionary with image names as keys and arrays as values for Mathematica.
    
    Parameters:
    -----------
    images_dict : dict
        Dictionary with {image_name: numpy_array} pairs
    filename : str
        Output filename (should end with .json)
    """
    # Convert numpy arrays to lists for JSON serialization
    json_dict = {}
    for image_name, array in images_dict.items():
        if isinstance(array, np.ndarray):
            json_dict[image_name] = array.tolist()
        else:
            json_dict[image_name] = array
    
    # Save to JSON
    with open(filename, 'w') as f:
        json.dump(json_dict, f, indent=2)
    
    print(f"Dictionary with {len(json_dict)} images saved to {filename}")
    return filename

# # Example: Create dictionary from your STM images
# stm_images_dict = {}

# # Process a few images and store them
# for f in clean_large_fow_P1:  # Take 3 random images
#     # Get the image name (filename without path)
#     image_name = os.path.basename(f)
    
#     # Load and process the image
#     _z = nio.Scan(f).signals["[P1]_Z"]['forward']/1e-9
    
#     # Apply your processing
#     _z_filtered = line_filter(_z, direction='horizontal')
#     _z_filtered = polynomial_background_subtraction(_z_filtered, degree=2)[0]
#     _z_filtered = gradient_contrast_filter(_z_filtered, sigma=2.0, alpha=2.5)
    
#     # Store in dictionary
#     stm_images_dict[image_name] = _z_filtered
    
#     print(f"Processed and stored: {image_name}, shape: {_z_filtered.shape}")

# # Save to JSON for Mathematica
# json_filename = save_images_dict_for_mathematica(stm_images_dict, 'stm_images.json')


