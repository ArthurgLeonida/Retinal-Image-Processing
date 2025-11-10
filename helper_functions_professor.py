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
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    # A 'blackhat_img' será uma imagem em tons de cinza onde os vasos
    # (que eram escuros) agora estão BRILHANTES.
    # blackhat_img = cv2.morphologyEx(green_shade_correct, cv2.MORPH_BLACKHAT, kernel)
    # ret, th = cv2.threshold(blackhat_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # ====================================================================================================== #

    """
    image_float = img_as_float(green_shade_correct)
    frangi_img = frangi(
        image_float, 
        sigmas=range(1, 12, 1),  # This range is good
        black_ridges=True,
        beta=0.1,                
        gamma=0.01
    )

    output_max = frangi_img.max()
    print(f"Frangi output max value: {output_max}")
    # frangi_norm = frangi_img / np.max(frangi_img)
    """
    # ====================================================================================================== #
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))


    th = cv2.morphologyEx(at, cv2.MORPH_CLOSE, clean_kernel, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, clean_kernel, iterations=4)
    
    
    #image_float = img_as_float(at)
    #th = frangi(
    #    image_float, 
    #    sigmas=range(1, 10, 1),  # This range is good
    #    black_ridges=False
    #)

    # PROFESSOR SUGERIU TESTAR ESSAS OPERAÇÕES

    # th = cv2.erode(th, kernel, iterations=3)
    # th = cv2.dilate(th, kernel, iterations=3)

    fig = plt.figure(figsize=(18,9))
    plt.subplot(121),plt.imshow(green_shade_correct, cmap='gray'),plt.title('Corrected shade)')
    plt.xticks([]), plt.yticks([])
    #plt.subplot(132),plt.imshow(at, cmap='gray'),plt.title('adaptiveThreshold output')
    #plt.xticks([]), plt.yticks([])    
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
    
    # --- ETAPA 1: Remover a borda (FOV Mask) ---
    # (Usando Otsu para robustez)
    #_, fov_mask = cv2.threshold(green_channel_original, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #fov_mask = cv2.erode(fov_mask, None, iterations=4)
    #mask_no_border = cv2.bitwise_and(noisy_mask, noisy_mask, mask=fov_mask)

    # --- ETAPA 2: LIMPAR O RUÍDO (Abertura) ---

    # ESTA É A NOVA ETAPA CRUCIAL.
    # Usamos ABERTURA (OPEN) para remover todos os pequenos pontos brancos (ruído "sal").
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) 
    mask_opened = cv2.morphologyEx(noisy_mask, cv2.MORPH_OPEN, kernel_open, iterations=4)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_close = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel_close, iterations=8)    
    
    # --- ETAPA 3: CONECTAR OS VASOS (Fechamento) ---
    # Agora que o ruído se foi, podemos conectar com segurança
    # os segmentos de vasos quebrados (pontilhados).
    # Podemos usar iterações mais fortes se necessário.
    # kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # mask_connected = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # --- ETAPA 4: Filtrar por Área e Forma ---
    # Esta etapa agora é opcional, pois a maior parte do ruído se foi.
    # Ela ainda é útil para remover quaisquer blobs restantes (como o disco óptico).
    '''
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_opened, connectivity=8)
    
    vessels_final_mask = np.zeros_like(mask_connected)
    
    
    # Começamos em '1' para pular o fundo (label 0)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        # --- Filtro de Área Mínima ---
        # Descarta qualquer coisa muito pequena (ruído).
        # (Você pode precisar ajustar este valor)
        if area < 50:
            continue
            
        # --- Filtro de Forma (Aspect Ratio) ---
        # Medimos a "largura" e "altura" do componente.
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        
        # Evita divisão por zero
        if width == 0 or height == 0:
            continue
            
        # Vasos são longos, blobs são redondos.
        # Um vaso terá um aspect_ratio alto.
        # Um blob (ruído, lesão) terá aspect_ratio ~1.0.
        aspect_ratio = max(width, height) / min(width, height)
        
        # --- A Lógica de Decisão ---
        # Manter o componente se:
        # 1. Ele for "longo e fino" (parecido com um vaso)
        # 2. OU for "grande" (provavelmente uma junção de vasos, que é redonda)
        # (Você pode precisar ajustar estes valores)
        if aspect_ratio > 1.8 or area > 150:
            # Se for aprovado, desenha na máscara final
            vessels_final_mask[labels == i] = 255
    '''
    fig = plt.figure(figsize=(18,9))
    plt.subplot(131),plt.imshow(noisy_mask, cmap='gray'),plt.title('Noisy Vessel Mask')
    plt.xticks([]), plt.yticks([])
    plt.subplot(132),plt.imshow(mask_opened, cmap='gray'),plt.title('Cleaned mask after Opeming')
    plt.xticks([]), plt.yticks([])    
    plt.subplot(133),plt.imshow(mask_close, cmap='gray'),plt.title('Final Vessel Mask after Closing')
    plt.xticks([]), plt.yticks([])
    plt.show()

    return mask_opened