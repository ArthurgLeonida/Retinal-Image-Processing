import matplotlib.pyplot as plt
import cv2
import numpy as np

def corrige_fundo(green_c):
    img = cv2.addWeighted(green_c, 0.5, ~cv2.medianBlur(green_c, 201), 0.5, 0)
    fig = plt.figure(figsize=(18,9))
    plt.subplot(121),plt.imshow(green_c, cmap='gray'),plt.title('Canal verde')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img, cmap='gray'),plt.title('Correção de fundo')
    plt.xticks([]), plt.yticks([])
    plt.show()

    return img

def correct_shade(green_c):
    img = cv2.addWeighted(green_c, 0.3, ~cv2.medianBlur(green_c, 221), 0.7, 0)
    fig = plt.figure(figsize=(18,9))
    plt.subplot(121),plt.imshow(green_c, cmap='gray'),plt.title('Green channel')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img, cmap='gray'),plt.title('Shade corrected')
    plt.xticks([]), plt.yticks([])
    plt.show()

    return img

def blood_vessel_subtraction(green_shade_correct):

    img2 = scipy.ndimage.filters.gaussian_filter(green_shade_correct, 4)
    th = cv2.adaptiveThreshold(
        green_shade_correct,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        105,
        2
    )
    kernel = np.array(
        ([0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]),
        np.uint8
    )
    th = cv2.erode(th, kernel, iterations=3)
    th = cv2.dilate(th, kernel, iterations=3)

    fig = plt.figure(figsize=(18,7))
    plt.imshow(th, cmap='gray')
    plt.title('Blood vessels subtraction')
    plt.xticks([]),plt.yticks([])
    plt.show()

    return th