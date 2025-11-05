import cv2
import numpy as np

def enhance_image(img_path):
    """
    Apply optimized enhancement techniques for fundus images:
    - LAB color space processing (better for medical images)
    - CLAHE on luminance channel
    - Bilateral filtering (preserves edges better)
    - Weighted combination for final enhancement
    """
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        return None, None, None
    
    # Convert to RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Method 1: Green channel with CLAHE (original approach)
    green_channel = img[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    green_enhanced = clahe.apply(green_channel)
    
    # Method 2: LAB color space enhancement (better for medical images)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Apply CLAHE to L-channel
    l_enhanced = clahe.apply(l_channel)
    
    # Merge back and convert to grayscale
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    bgr_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    gray_enhanced = cv2.cvtColor(bgr_enhanced, cv2.COLOR_BGR2GRAY)
    
    # Combine both methods using addWeighted (gives better results)
    # Green channel: 60%, LAB grayscale: 40%
    combined = cv2.addWeighted(green_enhanced, 0.6, gray_enhanced, 0.4, 0)
    
    # Apply bilateral filter instead of denoising (preserves edges better)
    final_enhanced = cv2.bilateralFilter(combined, 9, 75, 75)
    
    return img_rgb, green_enhanced, final_enhanced

def segment_vessels(enhanced_img,
                    median_blur_k=5,
                    blackhat_k=(7, 7),
                    clahe_clip=2.0,
                    post_blur_k=3,
                    min_area=30,
                    max_area=8000,
                    min_aspect_ratio=1.3,
                    min_area_for_aspect=80):
    """
    Segments retinal vessels from an enhanced (e.g., green channel) fundus image.
    
    The pipeline uses a morphology-based approach (black-hat) with
    component-based filtering to isolate vessel-like structures.
    
    Args:
        enhanced_img: 8-bit single-channel image where vessels are dark.
        median_blur_k: Kernel size for the initial median blur (for noise).
        blackhat_k: Kernel size for the black-hat transform.
        clahe_clip: Clip limit for the CLAHE applied to the black-hat.
        post_blur_k: Kernel size for the blur after CLAHE (before thresholding).
        min_area: Minimum pixel area to be considered a vessel.
        max_area: Maximum pixel area to be considered a vessel.
        min_aspect_ratio: Minimum aspect ratio (elongation) to keep a component.
        min_area_for_aspect: Components larger than this area are kept
                             regardless of aspect ratio.

    Returns:
        A binary mask (uint8) of the segmented vessels (Vessels=255, BG=0).
    """
    # PASSO 1: Remover ruído sal-e-pimenta com median blur
    if median_blur_k > 0:
        denoised = cv2.medianBlur(enhanced_img, median_blur_k)
    else:
        denoised = enhanced_img
    
    # PASSO 2: BLACK-HAT transform (correto para vasos ESCUROS)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, blackhat_k)
    blackhat = cv2.morphologyEx(denoised, cv2.MORPH_BLACKHAT, kernel)
    
    # PASSO 3: Enhance black-hat result com CLAHE
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    vessel_enhanced = clahe.apply(blackhat)

    # PASSO 4: Outro median blur leve para suavizar
    if post_blur_k > 0:
        vessel_enhanced = cv2.medianBlur(vessel_enhanced, post_blur_k)
    
    # PASSO 5: Thresholding - Otsu
    _, vessels_binary = cv2.threshold(vessel_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # PASSO 6: Limpeza por componentes conexos
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(vessels_binary, connectivity=8)
    
    vessels_final = np.zeros_like(vessels_binary)

    for i in range(1, num_labels): # Pula o label 0 (background)
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Filtro por tamanho
        if min_area < area < max_area:
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Filtro por forma (aspect ratio)
            if width > 0 and height > 0:
                aspect_ratio = max(width, height) / min(width, height)
                
                # Manter se for elongado (vaso) OU grande o suficiente (junção)
                if aspect_ratio > min_aspect_ratio or area > min_area_for_aspect:
                    vessels_final[labels == i] = 255
    
    # PASSO 8: Inverter para BLACK vessels on WHITE background (melhor visualização)
    vessels_inverted = cv2.bitwise_not(vessels_final)
    
    return vessels_inverted, blackhat


def segment_vessels_frangi(enhanced_img):
    """
    SOLUÇÃO 2: Filtro de Frangi (ABORDAGEM SUPERIOR)
    
    POR QUE FRANGI É MELHOR:
    ========================
    1. ESPECÍFICO PARA VASOS: Detecta estruturas tubulares especificamente
    2. MULTI-ESCALA: Detecta vasos finos E grossos simultaneamente
    3. ROBUSTO A RUÍDO: Análise de Hessian filtra ruído sal-e-pimenta
    4. INVARIANTE: Funciona independente de brilho/contraste local
    5. TEORIA SÓLIDA: Baseado em geometria diferencial, não heurísticas
    
    COMPARAÇÃO COM MORFOLOGIA:
    - Morfologia: Sensível a ruído, requer muitos parâmetros, perde vasos finos
    - Frangi: Robusto, poucos parâmetros, preserva estrutura vascular completa
    
    Requer: pip install scikit-image
    """
    try:
        from skimage.filters import frangi
        
        # PASSO 1: Normalizar imagem para [0, 1] (requerido por Frangi)
        img_normalized = enhanced_img.astype(np.float64) / 255.0
        
        # PASSO 2: Aplicar filtro de Frangi
        # sigmas: range de escalas (larguras de vasos) para detectar
        # black_ridges=True: CRÍTICO! Vasos são ESCUROS no fundus
        vessels_frangi = frangi(
            img_normalized,
            sigmas=range(1, 10, 1),  # Detecta vasos de 1-10 pixels de largura
            black_ridges=True,        # IMPORTANTE: vasos escuros!
            alpha=0.5,                # Sensibilidade a estruturas blob-like
            beta=0.5,                 # Sensibilidade a ruído de fundo
            gamma=15                  # Magnitude do Hessian (contraste)
        )
        
        # PASSO 3: Converter de volta para [0, 255]
        vessels_frangi_8bit = (vessels_frangi * 255).astype(np.uint8)
        
        # PASSO 4: Threshold simples (Frangi já fez o trabalho pesado)
        _, vessels_binary = cv2.threshold(
            vessels_frangi_8bit, 0, 255, 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # PASSO 5: Limpeza mínima (Frangi é muito limpo)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        vessels_cleaned = cv2.morphologyEx(vessels_binary, cv2.MORPH_OPEN, kernel)
        
        # PASSO 6: Remover apenas componentes muito pequenos
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(vessels_cleaned, connectivity=8)
        vessels_final = np.zeros_like(vessels_cleaned)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            # Threshold mais baixo que morfologia (Frangi é mais preciso)
            if 20 < area < 8000:
                vessels_final[labels == i] = 255
        
        # PASSO 7: Inverter para visualização (BLACK on WHITE)
        vessels_inverted = cv2.bitwise_not(vessels_final)
        
        # Retornar Frangi output como "tophat" para compatibilidade
        return vessels_inverted, vessels_frangi_8bit, vessels_frangi_8bit
        
    except ImportError:
        print("\n" + "="*70)
        print("⚠️  AVISO: scikit-image não está instalado!")
        print("="*70)
        print("Frangi Filter NÃO disponível - usando método morfológico básico.")
        print("\nPara usar Frangi (RECOMENDADO), instale:")
        print("  pip install scikit-image")
        print("="*70 + "\n")
        return segment_vessels(enhanced_img)


def segment_optic_disc(img_path):
    """
    Improved optic disc segmentation using:
    - Weighted grayscale conversion (green channel emphasis)
    - Multi-stage morphological filtering
    - Adaptive brightness-based detection
    """
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        return None, None, None, None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Weighted grayscale conversion (emphasize green channel)
    # Green: 70%, Red: 20%, Blue: 10% (optic disc is brightest in green)
    b, g, r = cv2.split(img)
    gray = cv2.addWeighted(
        cv2.addWeighted(g, 0.7, r, 0.2, 0), 
        1.0, 
        b, 
        0.1, 
        0
    )
    
    # Apply bilateral filter (preserves edges while smoothing)
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply Gaussian blur for additional smoothing
    blurred = cv2.GaussianBlur(filtered, (9, 9), 2)
    
    # Multi-stage morphological opening to remove vessels
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel_medium)
    
    # Apply closing to fill small holes
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_large)
    
    # Threshold using Otsu's method
    _, thresh = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return img_rgb, None, None, None
    
    # Get the largest bright region (optic disc)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Fit a circle to the contour
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    center = (int(x), int(y))
    radius = int(radius)
    
    # Create mask for the optic disc
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    
    return img_rgb, mask, center, radius

def measure_optic_disc(mask, radius):
    """
    Calculate area and perimeter of the optic disc.
    """
    if mask is None or radius is None:
        return None, None, None, None
    
    # Count white pixels for actual area
    area_pixels = np.sum(mask == 255)
    
    # Calculate theoretical area from circle
    area_theoretical = np.pi * (radius ** 2)
    
    # Find contours to calculate perimeter
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        perimeter_pixels = cv2.arcLength(contours[0], True)
    else:
        perimeter_pixels = None
    
    # Calculate theoretical perimeter from circle
    perimeter_theoretical = 2 * np.pi * radius
    
    return area_pixels, area_theoretical, perimeter_pixels, perimeter_theoretical