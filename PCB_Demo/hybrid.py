import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

IMG_SIZE = 224

def pcb_corrosion_mask(img_bgr):
    pcb_bgr = img_bgr.copy()
    gray = cv2.cvtColor(pcb_bgr, cv2.COLOR_BGR2GRAY)

    hsv = cv2.cvtColor(pcb_bgr, cv2.COLOR_BGR2HSV)
    Hc, S, V = cv2.split(hsv)

    _, board_mask0 = cv2.threshold(S, 25, 255, cv2.THRESH_BINARY)
    kernel_board = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    board_mask = cv2.morphologyEx(board_mask0, cv2.MORPH_CLOSE,
                                  kernel_board, iterations=1)
    board_mask = cv2.morphologyEx(board_mask, cv2.MORPH_OPEN,
                                  kernel_board, iterations=1)

    lab = cv2.cvtColor(pcb_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    L_f = L.astype(np.float32)
    bg = cv2.GaussianBlur(L_f, (201, 201), 0)
    ratio = cv2.divide(L_f, bg + 1)
    ratio_n = cv2.normalize(ratio, None, 0, 255,
                            cv2.NORM_MINMAX).astype(np.uint8)

    ret_L, mask_L = cv2.threshold(
        ratio_n, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    if ratio_n[mask_L == 255].mean() < ratio_n[mask_L == 0].mean():
        mask_L = 255 - mask_L

    mask_copper0 = cv2.bitwise_and(board_mask, mask_L)

    kernel_copper = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_copper = cv2.morphologyEx(mask_copper0, cv2.MORPH_CLOSE,
                                   kernel_copper, iterations=1)

    kernel_roi = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    inner_copper = cv2.dilate(mask_copper, kernel_roi, iterations=1)

    sigma = 7
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    blur = cv2.GaussianBlur(gray, (ksize, ksize), sigmaX=sigma)
    dark = cv2.subtract(blur, gray)
    dark_norm = cv2.normalize(dark, None, 0, 255,
                              cv2.NORM_MINMAX).astype(np.uint8)

    gray_f = gray.astype(np.float32)
    win = 9
    mean = cv2.blur(gray_f, (win, win))
    mean_sq = cv2.blur(gray_f * gray_f, (win, win))
    var = mean_sq - mean * mean
    var[var < 0] = 0
    std_local = np.sqrt(var)
    std_norm = cv2.normalize(std_local, None, 0, 255,
                             cv2.NORM_MINMAX).astype(np.uint8)

    score = cv2.addWeighted(dark_norm, 0.5, std_norm, 0.5, 0)
    score_inner = np.zeros_like(score, dtype=np.float32)
    score_inner[inner_copper == 255] = score[inner_copper == 255]

    mask_copper_bin = (mask_copper > 0).astype(np.uint8) * 255
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    copper_closed = cv2.morphologyEx(mask_copper_bin, cv2.MORPH_CLOSE,
                                     kernel_close, iterations=1)
    gap_raw = cv2.subtract(copper_closed, mask_copper_bin)
    gap_raw = cv2.bitwise_and(gap_raw, inner_copper)

    h, w = score_inner.shape
    mid = h // 2
    mask_bottom = np.zeros_like(inner_copper, dtype=bool)
    mask_bottom[mid:h, :] = True

    vals_bottom = score_inner[
        (inner_copper == 255) & mask_bottom & (score_inner > 0)
    ]

    if len(vals_bottom) > 0:
        sensitivity_bottom = 0.15
        thr_bottom = np.percentile(
            vals_bottom, 100 * (1.0 - sensitivity_bottom)
        )
        mask_dark_bottom = np.zeros_like(score_inner, dtype=np.uint8)
        cond_bottom = (
            (score_inner >= thr_bottom) &
            (inner_copper == 255) &
            mask_bottom
        )
        mask_dark_bottom[cond_bottom] = 255
    else:
        mask_dark_bottom = np.zeros_like(score_inner, dtype=np.uint8)

    mask_clean = cv2.bitwise_or(gap_raw, mask_dark_bottom.astype(np.uint8))

    inner_copper01 = (inner_copper > 0).astype(np.uint8)
    mask_clean01 = (mask_clean > 0).astype(np.uint8)

    return mask_clean01, inner_copper01

def crop_pcb_board(full_bgr, s_thresh=25):
    hsv_full = cv2.cvtColor(full_bgr, cv2.COLOR_BGR2HSV)
    S_full = hsv_full[:, :, 1]
    _, board_mask0 = cv2.threshold(S_full, s_thresh, 255, cv2.THRESH_BINARY)

    kernel_board = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    board_mask = cv2.morphologyEx(board_mask0, cv2.MORPH_CLOSE, kernel_board, iterations=1)
    board_mask = cv2.morphologyEx(board_mask, cv2.MORPH_OPEN,  kernel_board, iterations=1)

    contours, _ = cv2.findContours(board_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return full_bgr.copy(), 0, 0

    max_cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_cnt)
    pad = int(0.02 * min(full_bgr.shape[0], full_bgr.shape[1]))
    x0 = max(x - pad, 0)
    y0 = max(y - pad, 0)
    x1 = min(x + w + pad, full_bgr.shape[1])
    y1 = min(y + h + pad, full_bgr.shape[0])
    board_bgr = full_bgr[y0:y1, x0:x1].copy()
    return board_bgr, x0, y0

class ResNet4Channel(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNet4Channel, self).__init__()
        base = models.resnet18(weights=None)
        
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = nn.Linear(base.fc.in_features, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
        
        self.enc1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        self.bottleneck = conv_block(512, 1024)
        
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)
        
        self.out = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        b = self.bottleneck(self.pool4(e4))
        
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.out(d1)

def infer_unet_tiled(board_bgr, model_unet, img_size=224, stride=112, 
                     transform=None, device=None, thr=0.5):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H, W = board_bgr.shape[:2]
    ys = list(range(0, max(H - img_size, 0) + 1, stride))
    xs = list(range(0, max(W - img_size, 0) + 1, stride))
    if len(ys) == 0: ys = [0]
    if len(xs) == 0: xs = [0]

    prob_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    model_unet.eval()
    model_unet.to(device)

    with torch.no_grad():
        for y in ys:
            for x in xs:
                patch = np.zeros((img_size, img_size, 3), dtype=np.uint8)
                sub = board_bgr[y:y+img_size, x:x+img_size]
                patch[:sub.shape[0], :sub.shape[1]] = sub

                patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                patch_pil = Image.fromarray(patch_rgb)
                inp = transform(patch_pil).unsqueeze(0).to(device)

                logits = model_unet(inp)
                probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

                h_sub, w_sub = sub.shape[:2]
                prob_map[y:y+h_sub, x:x+w_sub] += probs[:h_sub, :w_sub]
                count_map[y:y+h_sub, x:x+w_sub] += 1.0

    count_map[count_map == 0] = 1.0
    prob_map /= count_map
    mask_full = (prob_map > thr).astype(np.uint8)
    return prob_map, mask_full

def infer_resnet_heatmap_4ch_board(board_bgr, mask_unet_roi, patch_size=224, 
                                   stride=112, resnet_model=None, transform=None, 
                                   device=None, coverage_thr=0.01):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H, W = board_bgr.shape[:2]
    ys = list(range(0, max(H - patch_size, 0) + 1, stride))
    xs = list(range(0, max(W - patch_size, 0) + 1, stride))
    if len(ys) == 0: ys = [0]
    if len(xs) == 0: xs = [0]

    Gh = len(ys)
    Gw = len(xs)

    severity_grid = np.zeros((Gh, Gw), dtype=np.int32)
    prob_defect_grid = np.zeros((Gh, Gw), dtype=np.float32)

    resnet_model.eval()
    resnet_model.to(device)

    with torch.no_grad():
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                h_sub = min(patch_size, H - y)
                w_sub = min(patch_size, W - x)

                patch_mask = mask_unet_roi[y:y+h_sub, x:x+w_sub]
                coverage = patch_mask.sum() / float(h_sub * w_sub)

                if coverage < coverage_thr:
                    severity_grid[iy, ix] = 0
                    prob_defect_grid[iy, ix] = 0.0
                    continue

                patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                sub = board_bgr[y:y+patch_size, x:x+patch_size]
                patch[:sub.shape[0], :sub.shape[1]] = sub

                mask_clean01, _ = pcb_corrosion_mask(patch)

                patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                patch_pil = Image.fromarray(patch_rgb)
                img_3ch = transform(patch_pil)
                mask_t = torch.from_numpy(mask_clean01).float().unsqueeze(0)
                inp_4ch = torch.cat([img_3ch, mask_t], dim=0).unsqueeze(0).to(device)

                logits = resnet_model(inp_4ch)
                proba = torch.softmax(logits, dim=1).cpu().numpy()[0]

                pred_cls = int(np.argmax(proba))
                p_defect = 1.0 - float(proba[0])

                severity_grid[iy, ix] = pred_cls
                prob_defect_grid[iy, ix] = p_defect

    return severity_grid, prob_defect_grid

def get_severity_color(severity):
    if severity < 1:
        return (0, 255, 0)
    elif severity < 2:
        return (0, 255, 255)
    elif severity < 3:
        return (0, 165, 255)
    else:
        return (0, 0, 255)

def calculate_hybrid_corrosion_percentage(mask_final, copper_mask):
    copper_pixels = cv2.countNonZero(copper_mask)
    if copper_pixels == 0:
        return 0.0
    
    corrosion_pixels = cv2.countNonZero(mask_final)
    percentage = (corrosion_pixels / copper_pixels) * 100
    return percentage

def run_hybrid_pipeline(image, cnn_path=None, unet_path=None, show_labels=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_cnn = ResNet4Channel(num_classes=4)
    model_unet = UNet(in_channels=3, out_channels=1)
    
    if cnn_path and os.path.exists(cnn_path):
        model_cnn.load_state_dict(torch.load(cnn_path, map_location=device))
    
    if unet_path and os.path.exists(unet_path):
        model_unet.load_state_dict(torch.load(unet_path, map_location=device))
    
    model_cnn.to(device).eval()
    model_unet.to(device).eval()
    
    cnn_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])
    
    unet_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                           std=[0.5, 0.5, 0.5]),
    ])
    
    max_side = 1400
    H0, W0 = image.shape[:2]
    scale = min(max_side / max(H0, W0), 1.0)
    if scale < 1.0:
        image = cv2.resize(image, (int(W0*scale), int(H0*scale)),
                          interpolation=cv2.INTER_AREA)
    
    board_bgr, x0, y0 = crop_pcb_board(image)
    H_board, W_board = board_bgr.shape[:2]
    
    board_small = cv2.resize(board_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    _, inner_copper_small = pcb_corrosion_mask(board_small)
    inner_copper_full = cv2.resize(inner_copper_small, (W_board, H_board),
                                   interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    
    prob_unet, mask_unet = infer_unet_tiled(
        board_bgr,
        model_unet=model_unet,
        img_size=IMG_SIZE,
        stride=IMG_SIZE//2,
        transform=unet_transform,
        device=device,
        thr=0.5
    )
    
    mask_unet_roi = (mask_unet & (inner_copper_full > 0).astype(np.uint8)).astype(np.uint8)
    
    severity_grid, prob_defect_grid = infer_resnet_heatmap_4ch_board(
        board_bgr,
        mask_unet_roi=mask_unet_roi,
        patch_size=IMG_SIZE,
        stride=IMG_SIZE//2,
        resnet_model=model_cnn,
        transform=cnn_transform,
        device=device,
        coverage_thr=0.01
    )
    
    severity_map = cv2.resize(severity_grid.astype(np.float32),
                             (W_board, H_board),
                             interpolation=cv2.INTER_NEAREST)
    severity_map[inner_copper_full == 0] = 0
    
    severity_norm = severity_map / 3.0
    heatmap_color = cv2.applyColorMap(
        (severity_norm * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    overlay_resnet = cv2.addWeighted(board_bgr, 0.6, heatmap_color, 0.4, 0)
    
    mask_defect = ((severity_map >= 1) & (mask_unet_roi == 1)).astype(np.uint8)
    contours, _ = cv2.findContours(mask_defect, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    vis_boxes = overlay_resnet.copy()
    components = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        region = severity_map[y:y+h, x:x+w]
        region_level = int(region.max())
        
        roi_mask = mask_unet_roi[y:y+h, x:x+w]
        corrosion_pixels = cv2.countNonZero(roi_mask)
        total_pixels = w * h
        severity = (corrosion_pixels / total_pixels) * 100
        
        components.append((x, y, w, h, severity))
        
        color = get_severity_color(region_level)
        cv2.rectangle(vis_boxes, (x, y), (x+w, y+h), color, 2)
        
        if show_labels:
            label = f"L{region_level}"
            cv2.putText(vis_boxes, label, (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    percentage = calculate_hybrid_corrosion_percentage(mask_unet_roi, inner_copper_full)
    
    overlay_unet = board_bgr.copy()
    overlay_unet[mask_unet_roi == 1] = (0, 0, 255)
    
    return {
        'result': vis_boxes,
        'pcb_processed': board_bgr,
        'mask_copper': (inner_copper_full * 255).astype(np.uint8),
        'inner_copper': (inner_copper_full * 255).astype(np.uint8),
        'mask_corrosion': (mask_unet_roi * 255).astype(np.uint8),
        'severity_heatmap': heatmap_color,
        'unet_overlay': overlay_unet,
        'components': components,
        'percentage': percentage,
        'severity_map': severity_map
    }