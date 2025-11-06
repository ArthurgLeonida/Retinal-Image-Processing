import matplotlib.pyplot as plt
import cv2
import numpy as np
import scipy
from skimage.filters import frangi
from skimage.util import img_as_float


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
    # img = cv2.addWeighted(green_c, 0.5, ~cv2.bilateralFilter(green_c, (224, 224), 0), 0.5, 0)
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

    # img2 = scipy.ndimage.filters.gaussian_filter(green_shade_correct, 5)
    
    #th = cv2.adaptiveThreshold(
    #    img2,
    #    255,
    #    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #    cv2.THRESH_BINARY_INV,
    #    105, # 105
    #    2
    #)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    # A 'blackhat_img' será uma imagem em tons de cinza onde os vasos
    # (que eram escuros) agora estão BRILHANTES.
    blackhat_img = cv2.morphologyEx(green_shade_correct, cv2.MORPH_BLACKHAT, kernel)
    ret, th = cv2.threshold(blackhat_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # ====================================================================================================== #
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, clean_kernel, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, clean_kernel, iterations=2)
    
    # PROFESSOR SUGERIU TESTAR ESSAS OPERAÇÕES

    # th = cv2.erode(th, kernel, iterations=3)
    # th = cv2.dilate(th, kernel, iterations=3)

    fig = plt.figure(figsize=(18,9))
    plt.subplot(121),plt.imshow(green_shade_correct, cmap='gray'),plt.title('Enchanced image (clahe + shade corrected)')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(th, cmap='gray'),plt.title('Blood vessels segmented')
    plt.xticks([]), plt.yticks([])
    plt.show()

    return th