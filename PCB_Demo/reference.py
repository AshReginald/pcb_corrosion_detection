import cv2
import numpy as np

def crop_pcb_region(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)

    kernel = np.ones((15,15), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < image.shape[0]*image.shape[1]*0.1:
        return image

    x,y,w,h = cv2.boundingRect(largest)
    pad = 5
    x = max(0, x-pad)
    y = max(0, y-pad)
    w = min(image.shape[1]-x, w+2*pad)
    h = min(image.shape[0]-y, h+2*pad)
    return image[y:y+h, x:x+w]

def align_pcb_orientation(ref_img, target_img):
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    target = cv2.resize(target_img, (ref_img.shape[1], ref_img.shape[0]))

    best_mse = np.inf
    best_img = target
    best_scale = 1.0

    for code in [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
        if code is None:
            rot = target
        else:
            rot = cv2.rotate(target, code)
            rot = cv2.resize(rot, (ref_img.shape[1], ref_img.shape[0]))

        for scale in [0.98, 0.99, 1.0, 1.01, 1.02]:
            h, w = rot.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            scaled = cv2.resize(rot, (new_w, new_h))
            
            if scale < 1.0:
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                scaled = cv2.copyMakeBorder(scaled, pad_h, h-new_h-pad_h, 
                                           pad_w, w-new_w-pad_w, 
                                           cv2.BORDER_REFLECT_101)
            elif scale > 1.0:
                crop_h = (new_h - h) // 2
                crop_w = (new_w - w) // 2
                scaled = scaled[crop_h:crop_h+h, crop_w:crop_w+w]
            
            gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
            mse = np.mean((ref_gray.astype(np.float32) - gray.astype(np.float32))**2)

            if mse < best_mse:
                best_mse = mse
                best_img = scaled.copy()
                best_scale = scale

    return best_img

def refine_translation(ref_img, target_img, ref_mask=None, max_shift=10):
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tgt_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = ref_gray.shape

    best_mse = np.inf
    best_shift = (0,0)

    for dy in range(-max_shift, max_shift+1):
        for dx in range(-max_shift, max_shift+1):
            M = np.float32([[1,0,dx],[0,1,dy]])
            shifted = cv2.warpAffine(
                tgt_gray, M, (w,h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101
            )

            if ref_mask is not None:
                diff = (ref_gray - shifted)[ref_mask==255]
                if len(diff) == 0:
                    continue
                mse = np.mean(diff**2)
            else:
                mse = np.mean((ref_gray - shifted)**2)

            if mse < best_mse:
                best_mse = mse
                best_shift = (dx,dy)

    dx,dy = best_shift
    M = np.float32([[1,0,dx],[0,1,dy]])
    return cv2.warpAffine(
        target_img, M, (w,h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

def extract_copper_traces(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, S, _ = cv2.split(hsv)
    _, board = cv2.threshold(S, 25, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    board = cv2.morphologyEx(board, cv2.MORPH_CLOSE, kernel)
    board = cv2.morphologyEx(board, cv2.MORPH_OPEN, kernel)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L,_,_ = cv2.split(lab)

    Lf = L.astype(np.float32)
    bg = cv2.GaussianBlur(Lf, (201,201), 0, borderType=cv2.BORDER_REFLECT_101)
    ratio = cv2.divide(Lf, bg+1)

    ratio = cv2.normalize(ratio, None, 0,255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask_L = cv2.threshold(ratio, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    if ratio[mask_L==255].mean() < ratio[mask_L==0].mean():
        mask_L = cv2.bitwise_not(mask_L)

    copper = cv2.bitwise_and(board, mask_L)
    copper = cv2.morphologyEx(
        copper, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    )
    return copper

def get_copper_contour_mask(copper_mask):
    contours, _ = cv2.findContours(copper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(copper_mask)
    cv2.drawContours(mask, contours, -1, 255, -1)
    return mask

def detect_surface_corrosion(ref_copper, bad_copper, ref_img, bad_img):
    valid = cv2.bitwise_and(ref_copper, bad_copper)
    valid = cv2.erode(valid, np.ones((5,5),np.uint8))

    diff = cv2.absdiff(ref_copper, bad_copper)
    _, missing = cv2.threshold(diff, 50,255, cv2.THRESH_BINARY)
    missing = cv2.bitwise_and(missing, ref_copper)
    missing = cv2.bitwise_and(missing, cv2.bitwise_not(bad_copper))

    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    bad_gray = cv2.cvtColor(bad_img, cv2.COLOR_BGR2GRAY)
    tex = cv2.absdiff(ref_gray, bad_gray)
    _, tex = cv2.threshold(tex, 40,255, cv2.THRESH_BINARY)
    tex = cv2.bitwise_and(tex, valid)

    corrosion = cv2.bitwise_or(missing, tex)
    corrosion = cv2.morphologyEx(
        corrosion, cv2.MORPH_OPEN,
        np.ones((3,3),np.uint8), iterations=2
    )
    corrosion = cv2.dilate(corrosion, np.ones((3,3),np.uint8), iterations=1)
    
    copper_region = get_copper_contour_mask(ref_copper)
    corrosion = cv2.bitwise_and(corrosion, copper_region)
    
    return corrosion, copper_region

def calculate_corrosion_percentage(corrosion_mask, copper_mask):
    copper_pixels = cv2.countNonZero(copper_mask)
    if copper_pixels == 0:
        return 0.0
    
    corrosion_pixels = cv2.countNonZero(corrosion_mask)
    percentage = (corrosion_pixels / copper_pixels) * 100
    return percentage

def find_corrosion_regions(mask, min_area=80):
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes=[]
    for c in contours:
        area = cv2.contourArea(c)
        if area > min_area:
            x,y,w,h = cv2.boundingRect(c)
            if max(w,h)/(min(w,h)+1e-5) < 15:
                roi_mask = mask[y:y+h, x:x+w]
                corrosion_pixels = cv2.countNonZero(roi_mask)
                total_pixels = w * h
                severity = (corrosion_pixels / total_pixels) * 100
                boxes.append((x,y,w,h,severity))
    return boxes

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

def draw_corrosion_boxes(image, boxes, thickness=2, show_labels=False):
    result = image.copy()
    for box_data in boxes:
        x, y, w, h, severity = box_data
        color = get_severity_color(severity)
        cv2.rectangle(result, (x, y), (x+w, y+h), color, thickness)
        
        if show_labels:
            label = f"{severity:.1f}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            font_thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            label_y = y - 5 if y > 20 else y + h + 15
            cv2.putText(result, label, (x, label_y), font, font_scale, color, font_thickness, cv2.LINE_AA)
    
    return result