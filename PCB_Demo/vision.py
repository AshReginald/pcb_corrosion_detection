import cv2
import numpy as np

def preprocess_and_crop_pcb(image, max_side=1024):
    H, W = image.shape[:2]
    scale = min(max_side / max(H, W), 1.0)
    
    if scale != 1.0:
        pcb_bgr = cv2.resize(image, (int(W * scale), int(H * scale)),
                            interpolation=cv2.INTER_AREA)
    else:
        pcb_bgr = image.copy()
    
    hsv = cv2.cvtColor(pcb_bgr, cv2.COLOR_BGR2HSV)
    _, Sc, _ = cv2.split(hsv)
    
    _, board_mask0 = cv2.threshold(Sc, 30, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    board_mask = cv2.morphologyEx(board_mask0, cv2.MORPH_CLOSE, kernel, iterations=1)
    board_mask = cv2.morphologyEx(board_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    contours, _ = cv2.findContours(board_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        max_cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_cnt)
        
        pad = int(0.01 * min(pcb_bgr.shape[0], pcb_bgr.shape[1]))
        x0 = max(x - pad, 0)
        y0 = max(y - pad, 0)
        x1 = min(x + w + pad, pcb_bgr.shape[1])
        y1 = min(y + h + pad, pcb_bgr.shape[0])
        
        pcb_bgr = pcb_bgr[y0:y1, x0:x1].copy()
    
    return pcb_bgr

def segment_copper_and_roi(pcb_bgr):
    gray = cv2.cvtColor(pcb_bgr, cv2.COLOR_BGR2GRAY)
    
    hsv = cv2.cvtColor(pcb_bgr, cv2.COLOR_BGR2HSV)
    _, S, _ = cv2.split(hsv)
    
    _, board_mask0 = cv2.threshold(S, 25, 255, cv2.THRESH_BINARY)
    kernel_board = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    board_mask = cv2.morphologyEx(board_mask0, cv2.MORPH_CLOSE, kernel_board, iterations=1)
    board_mask = cv2.morphologyEx(board_mask, cv2.MORPH_OPEN, kernel_board, iterations=1)
    
    lab = cv2.cvtColor(pcb_bgr, cv2.COLOR_BGR2LAB)
    L, _, _ = cv2.split(lab)
    
    L_f = L.astype(np.float32)
    bg = cv2.GaussianBlur(L_f, (201, 201), 0)
    ratio = cv2.divide(L_f, bg + 1)
    ratio_n = cv2.normalize(ratio, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    _, mask_L = cv2.threshold(ratio_n, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if ratio_n[mask_L == 255].mean() < ratio_n[mask_L == 0].mean():
        mask_L = 255 - mask_L
    
    mask_copper0 = cv2.bitwise_and(board_mask, mask_L)
    kernel_copper = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_copper = cv2.morphologyEx(mask_copper0, cv2.MORPH_CLOSE, kernel_copper, iterations=1)
    
    kernel_roi = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    inner_copper = cv2.dilate(mask_copper, kernel_roi, iterations=1)
    
    sigma = 7
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    
    blur = cv2.GaussianBlur(gray, (ksize, ksize), sigmaX=sigma)
    dark = cv2.subtract(blur, gray)
    dark_norm = cv2.normalize(dark, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return mask_copper, inner_copper, dark_norm, gray

def detect_vision_corrosion(mask_copper, inner_copper, dark_norm, gray):
    gray_f = gray.astype(np.float32)
    win = 9
    mean = cv2.blur(gray_f, (win, win))
    mean_sq = cv2.blur(gray_f * gray_f, (win, win))
    var = mean_sq - mean * mean
    var[var < 0] = 0
    std_local = np.sqrt(var)
    
    std_norm = cv2.normalize(std_local, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    score = cv2.addWeighted(dark_norm, 0.5, std_norm, 0.5, 0)
    score_inner = np.zeros_like(score, dtype=np.float32)
    score_inner[inner_copper == 255] = score[inner_copper == 255]
    
    mask_copper_bin = (mask_copper > 0).astype(np.uint8) * 255
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    copper_closed = cv2.morphologyEx(mask_copper_bin, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    gap_raw = cv2.subtract(copper_closed, mask_copper_bin)
    gap_raw = cv2.bitwise_and(gap_raw, inner_copper)
    
    h, w = score_inner.shape
    tile_h = h // 4
    tile_w = w // 4
    
    mask_dark_adaptive = np.zeros_like(score_inner, dtype=np.uint8)
    
    for i in range(4):
        for j in range(4):
            y1 = i * tile_h
            y2 = (i + 1) * tile_h if i < 3 else h
            x1 = j * tile_w
            x2 = (j + 1) * tile_w if j < 3 else w
            
            tile_score = score_inner[y1:y2, x1:x2]
            tile_roi = inner_copper[y1:y2, x1:x2]
            
            vals_tile = tile_score[(tile_roi == 255) & (tile_score > 0)]
            
            if len(vals_tile) > 10:
                sensitivity = 0.15
                thr_tile = np.percentile(vals_tile, 100 * (1.0 - sensitivity))
                
                cond_tile = (tile_score >= thr_tile) & (tile_roi == 255)
                mask_dark_adaptive[y1:y2, x1:x2][cond_tile] = 255
    
    mask_clean = cv2.bitwise_or(gap_raw, mask_dark_adaptive)
    
    return mask_clean, copper_closed

def find_vision_corrosion_regions(mask_clean, border_margin_ratio=0.0):
    h, w = mask_clean.shape
    border_margin = int(min(h, w) * border_margin_ratio)
    core = np.zeros_like(mask_clean)
    core[border_margin:h-border_margin, border_margin:w-border_margin] = 255
    mask_core = cv2.bitwise_and(mask_clean, core)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_core)
    
    components = []
    mask_final = np.zeros_like(mask_core)
    
    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label]
        
        roi_mask = mask_core[y:y+bh, x:x+bw]
        corrosion_pixels = cv2.countNonZero(roi_mask)
        total_pixels = bw * bh
        severity = (corrosion_pixels / total_pixels) * 100
        
        components.append((x, y, bw, bh, severity))
        mask_final[labels == label] = 255
    
    return components, mask_final

def calculate_vision_corrosion_percentage(mask_final, copper_mask):
    copper_pixels = cv2.countNonZero(copper_mask)
    if copper_pixels == 0:
        return 0.0
    
    corrosion_pixels = cv2.countNonZero(mask_final)
    percentage = (corrosion_pixels / copper_pixels) * 100
    return percentage

def get_severity_color(severity):
    if severity < 20:
        return (0, 255, 0)
    elif severity < 40:
        return (0, 255, 255)
    elif severity < 60:
        return (0, 165, 255)
    elif severity < 80:
        return (0, 100, 255)
    else:
        return (0, 0, 255)

def draw_vision_boxes(image, components, thickness=2, show_labels=False):
    result = image.copy()
    for comp_data in components:
        x, y, bw, bh, severity = comp_data
        color = get_severity_color(severity)
        cv2.rectangle(result, (x, y), (x+bw, y+bh), color, thickness)
        
        if show_labels:
            label = f"{severity:.1f}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            font_thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            label_y = y - 5 if y > 20 else y + bh + 15
            cv2.putText(result, label, (x, label_y), font, font_scale, color, font_thickness, cv2.LINE_AA)
    
    return result

def run_vision_pipeline(image, show_labels=False):
    pcb_bgr = preprocess_and_crop_pcb(image)
    mask_copper, inner_copper, dark_norm, gray = segment_copper_and_roi(pcb_bgr)
    mask_clean, copper_closed = detect_vision_corrosion(mask_copper, inner_copper, dark_norm, gray)
    components, mask_final = find_vision_corrosion_regions(mask_clean)
    percentage = calculate_vision_corrosion_percentage(mask_final, mask_copper)
    
    result = draw_vision_boxes(pcb_bgr, components, thickness=2, show_labels=show_labels)
    
    return {
        'result': result,
        'pcb_processed': pcb_bgr,
        'mask_copper': mask_copper,
        'inner_copper': inner_copper,
        'mask_corrosion': mask_final,
        'copper_closed': copper_closed,
        'components': components,
        'percentage': percentage
    }