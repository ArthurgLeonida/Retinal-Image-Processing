import matplotlib.pyplot as plt
import cv2
import scipy

def enhance_contrast(green_channel):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance local contrast."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green_channel)
    
    fig = plt.figure(figsize=(18, 9))
    plt.subplot(121), plt.imshow(green_channel, cmap='gray'), plt.title('Green Channel (Shade Corrected)')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(enhanced, cmap='gray'), plt.title('CLAHE Enhanced')
    plt.xticks([]), plt.yticks([])
    plt.show()
    
    return enhanced

def correct_shade(green_c):
    """Correct the shading in the green channel image."""
    img = cv2.addWeighted(green_c, 0.5, ~cv2.GaussianBlur(green_c, (221, 221), 0), 0.5, 0)
    # img = cv2.addWeighted(green_c, 0.3, ~cv2.medianBlur(green_c, 221), 0.7, 0)
    fig = plt.figure(figsize=(18,9))
    plt.subplot(121),plt.imshow(green_c, cmap='gray'),plt.title('Green channel')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img, cmap='gray'),plt.title('Shade corrected')
    plt.xticks([]), plt.yticks([])
    plt.show()

    return img

def blood_vessel_subtraction(green_shade_correct):

    img2 = scipy.ndimage.filters.gaussian_filter(green_shade_correct, 5)
    
    at = cv2.adaptiveThreshold(
        img2,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        105, # 105
        2
    )
   
    # ====================================================================================================== #
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    th = cv2.morphologyEx(at, cv2.MORPH_CLOSE, clean_kernel, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, clean_kernel, iterations=4)

    fig = plt.figure(figsize=(18,9))
    plt.subplot(121),plt.imshow(green_shade_correct, cmap='gray'),plt.title('Corrected shade)')
    plt.xticks([]), plt.yticks([])   
    plt.subplot(122),plt.imshow(th, cmap='gray'),plt.title('Blood vessels segmented')
    plt.xticks([]), plt.yticks([])
    plt.show()

    return th

import cv2
import numpy as np

def clean_vessel_mask(noisy_mask, green_channel_original):
    """
    Limpa uma máscara de vasos ruidosa ("noisy_mask") usando a ordem correta
    de operações morfológicas e filtros de forma/tamanho.
    """
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) 
    mask_opened = cv2.morphologyEx(noisy_mask, cv2.MORPH_OPEN, kernel_open, iterations=4)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_close = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel_close, iterations=8)    
   
    fig = plt.figure(figsize=(18,9))
    plt.subplot(131),plt.imshow(noisy_mask, cmap='gray'),plt.title('Noisy Vessel Mask')
    plt.xticks([]), plt.yticks([])
    plt.subplot(132),plt.imshow(mask_opened, cmap='gray'),plt.title('Cleaned mask after Opeming')
    plt.xticks([]), plt.yticks([])    
    plt.subplot(133),plt.imshow(mask_close, cmap='gray'),plt.title('Final Vessel Mask after Closing')
    plt.xticks([]), plt.yticks([])
    plt.show()

    return mask_opened