import sys
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QFileDialog, QVBoxLayout, QHBoxLayout,
    QTextEdit, QTabWidget, QComboBox, QScrollArea
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, pyqtSignal

from reference import *
from vision import run_vision_pipeline
from hybrid import run_hybrid_pipeline

class ImageLabel(QLabel):
    mouse_moved = pyqtSignal(int, int)
    
    def __init__(self, text=""):
        super().__init__(text)
        self.setMouseTracking(True)
        self.original_size = None
        self.scaled_pixmap = None
        
    def mouseMoveEvent(self, event):
        if self.scaled_pixmap and self.original_size:
            label_w = self.width()
            label_h = self.height()
            pixmap_w = self.scaled_pixmap.width()
            pixmap_h = self.scaled_pixmap.height()
            
            offset_x = (label_w - pixmap_w) // 2
            offset_y = (label_h - pixmap_h) // 2
            
            mouse_x = event.x() - offset_x
            mouse_y = event.y() - offset_y
            
            if 0 <= mouse_x < pixmap_w and 0 <= mouse_y < pixmap_h:
                scale_x = self.original_size[0] / pixmap_w
                scale_y = self.original_size[1] / pixmap_h
                
                orig_x = int(mouse_x * scale_x)
                orig_y = int(mouse_y * scale_y)
                
                self.mouse_moved.emit(orig_x, orig_y)
        
        super().mouseMoveEvent(event)
    
    def setPixmapWithSize(self, pixmap, original_size):
        self.scaled_pixmap = pixmap
        self.original_size = original_size
        self.setPixmap(pixmap)

def cv_to_pixmap(img, max_size=(900,700)):
    if len(img.shape)==2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h,w,_ = img.shape
    qimg = QImage(img.data, w,h, 3*w, QImage.Format_RGB888)
    scaled = QPixmap.fromImage(qimg).scaled(
        max_size[0], max_size[1],
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation
    )
    return scaled, (w, h)

class PCBApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PCB Surface Corrosion Detection - Triple Mode")
        self.resize(1600,900)

        self.good_img = None
        self.bad_img = None
        self.good_crop = None
        self.bad_crop = None
        self.single_img = None
        self.hybrid_img = None

        self.tabs = QTabWidget()
        self.views = {}
        self.coord_labels = {}

        tab_names = [
            "Good PCB",
            "Bad PCB",
            "Result",
            "Ref Copper",
            "Bad Copper",
            "Corrosion Mask",
            "Single PCB",
            "Vision Result",
            "Hybrid PCB",
            "Hybrid Result",
            "Severity Heatmap"
        ]

        for name in tab_names:
            container = QWidget()
            layout = QVBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            
            lbl = ImageLabel(name)
            lbl.setAlignment(Qt.AlignCenter)
            self.views[name] = lbl
            
            coord_label = QLabel("Coordinates: N/A")
            coord_label.setStyleSheet("color: #00ff00; padding: 5px;")
            self.coord_labels[name] = coord_label
            
            lbl.mouse_moved.connect(lambda x, y, n=name: self.update_coordinates(n, x, y))
            
            layout.addWidget(lbl)
            layout.addWidget(coord_label)
            
            self.tabs.addTab(container, name)

        self.log = QTextEdit()
        self.log.setReadOnly(True)

        self.mode_selector = QComboBox()
        self.mode_selector.addItems([
            "Comparison Mode (2 PCBs)",
            "Vision Mode (1 PCB)",
            "Hybrid Mode (Deep Learning)"
        ])
        self.mode_selector.currentIndexChanged.connect(self.on_mode_changed)

        btn_good = QPushButton("Load GOOD PCB")
        btn_bad = QPushButton("Load BAD PCB")
        btn_single = QPushButton("Load Single PCB")
        btn_hybrid = QPushButton("Load PCB (Hybrid)")
        btn_run = QPushButton("Run Detection")

        btn_good.clicked.connect(self.load_good)
        btn_bad.clicked.connect(self.load_bad)
        btn_single.clicked.connect(self.load_single)
        btn_hybrid.clicked.connect(self.load_hybrid)
        btn_run.clicked.connect(self.run_detection)

        left = QVBoxLayout()
        left.addWidget(QLabel("Detection Mode:"))
        left.addWidget(self.mode_selector)
        left.addWidget(btn_good)
        left.addWidget(btn_bad)
        left.addWidget(btn_single)
        left.addWidget(btn_hybrid)
        left.addWidget(btn_run)
        left.addWidget(self.log)

        layout = QHBoxLayout(self)
        layout.addLayout(left, 2)
        layout.addWidget(self.tabs, 6)

        self.setStyleSheet("""
        QWidget {
            background-color: #000000;
            color: #ffffff;
        }

        QPushButton {
            background-color: #222222;
            color: #ffffff;
            padding: 6px;
        }

        QComboBox {
            background-color: #222222;
            color: #ffffff;
            padding: 4px;
        }

        QTextEdit {
            background-color: #000000;
            color: #ffffff;
        }

        QTabWidget::pane {
            border: 1px solid #444444;
        }

        QTabBar::tab {
            background: #ffffff;
            color: #000000;
            padding: 8px;
            margin: 2px;
        }

        QTabBar::tab:selected {
            background: #cccccc;
            color: #000000;
        }
        """)

    def update_coordinates(self, tab_name, x, y):
        self.coord_labels[tab_name].setText(f"Coordinates: X={x}, Y={y}")

    def load_good(self):
        try:
            path, _ = QFileDialog.getOpenFileName(self, "Select GOOD PCB")
            if path:
                self.good_img = cv2.imread(path)
                if self.good_img is None:
                    self.log.append(f"ERROR: Cannot read image {path}")
                    return
                self.good_crop = crop_pcb_region(self.good_img)
                pixmap, size = cv_to_pixmap(self.good_crop)
                self.views["Good PCB"].setPixmapWithSize(pixmap, size)
                self.log.append("Loaded and cropped GOOD PCB")
        except Exception as e:
            self.log.append(f"ERROR loading good PCB: {str(e)}")

    def on_mode_changed(self):
        self.log.clear()
        mode = self.mode_selector.currentIndex()
        if mode == 0:
            self.log.append("Switched to COMPARISON MODE")
            self.log.append("Load 2 PCB images (Good + Bad) and run detection")
        elif mode == 1:
            self.log.append("Switched to VISION MODE")
            self.log.append("Load 1 PCB image for automatic corrosion analysis")
        else:
            self.log.append("Switched to HYBRID MODE")
            self.log.append("Load 1 PCB image for deep learning analysis")
            self.log.append("Requires: cnn_pcb_pipeline_best.pth and unet_pcb_pipeline_best.pth")

    def load_bad(self):
        try:
            path, _ = QFileDialog.getOpenFileName(self, "Select BAD PCB")
            if path:
                self.bad_img = cv2.imread(path)
                if self.bad_img is None:
                    self.log.append(f"ERROR: Cannot read image {path}")
                    return
                self.bad_crop = crop_pcb_region(self.bad_img)
                pixmap, size = cv_to_pixmap(self.bad_crop)
                self.views["Bad PCB"].setPixmapWithSize(pixmap, size)
                self.log.append("Loaded and cropped BAD PCB")
        except Exception as e:
            self.log.append(f"ERROR loading bad PCB: {str(e)}")

    def load_single(self):
        try:
            path, _ = QFileDialog.getOpenFileName(self, "Select PCB for Vision Analysis")
            if path:
                self.single_img = cv2.imread(path)
                if self.single_img is None:
                    self.log.append(f"ERROR: Cannot read image {path}")
                    return
                pixmap, size = cv_to_pixmap(self.single_img)
                self.views["Single PCB"].setPixmapWithSize(pixmap, size)
                self.log.append("Loaded single PCB for vision analysis")
        except Exception as e:
            self.log.append(f"ERROR loading single PCB: {str(e)}")

    def load_hybrid(self):
        try:
            path, _ = QFileDialog.getOpenFileName(self, "Select PCB for Hybrid Analysis")
            if path:
                self.hybrid_img = cv2.imread(path)
                if self.hybrid_img is None:
                    self.log.append(f"ERROR: Cannot read image {path}")
                    return
                pixmap, size = cv_to_pixmap(self.hybrid_img)
                self.views["Hybrid PCB"].setPixmapWithSize(pixmap, size)
                self.log.append("Loaded PCB for hybrid analysis")
        except Exception as e:
            self.log.append(f"ERROR loading hybrid PCB: {str(e)}")

    def run_detection(self):
        mode = self.mode_selector.currentIndex()
        
        if mode == 0:
            self.run_comparison_mode()
        elif mode == 1:
            self.run_vision_mode()
        else:
            self.run_hybrid_mode()

    def run_comparison_mode(self):
        if self.good_crop is None or self.bad_crop is None:
            self.log.append("ERROR: Missing input images for comparison mode")
            return

        try:
            self.log.append("="*50)
            self.log.append("COMPARISON MODE (reference.py)")
            self.log.append("="*50)
            self.log.append("Starting detection pipeline...")
            QApplication.processEvents()

            self.log.append("Step 1: Aligning PCB orientation...")
            QApplication.processEvents()
            bad = align_pcb_orientation(self.good_crop, self.bad_crop)
            
            self.log.append("Step 2: Extracting reference copper traces...")
            QApplication.processEvents()
            ref_copper = extract_copper_traces(self.good_crop)
            
            self.log.append("Step 3: Refining translation alignment...")
            QApplication.processEvents()
            bad = refine_translation(self.good_crop, bad, ref_copper)
            
            self.log.append("Step 4: Extracting bad PCB copper traces...")
            QApplication.processEvents()
            bad_copper = extract_copper_traces(bad)

            self.log.append("Step 5: Detecting surface corrosion...")
            QApplication.processEvents()
            corrosion, copper_region = detect_surface_corrosion(
                ref_copper, bad_copper, self.good_crop, bad
            )
            
            self.log.append("Step 6: Finding corrosion regions...")
            QApplication.processEvents()
            boxes = find_corrosion_regions(corrosion)

            self.log.append("Step 7: Drawing results...")
            QApplication.processEvents()
            result = draw_corrosion_boxes(bad, boxes, thickness=2, show_labels=False)

            ratio = calculate_corrosion_percentage(corrosion, copper_region)

            pixmap, size = cv_to_pixmap(result)
            self.views["Result"].setPixmapWithSize(pixmap, size)
            
            pixmap, size = cv_to_pixmap(ref_copper)
            self.views["Ref Copper"].setPixmapWithSize(pixmap, size)
            
            pixmap, size = cv_to_pixmap(bad_copper)
            self.views["Bad Copper"].setPixmapWithSize(pixmap, size)
            
            pixmap, size = cv_to_pixmap(corrosion)
            self.views["Corrosion Mask"].setPixmapWithSize(pixmap, size)

            self.log.append("="*50)
            self.log.append(f"COMPARISON MODE RESULTS:")
            self.log.append(f"Detected corrosion regions: {len(boxes)}")
            self.log.append(f"Corrosion percentage: {ratio:.2f}%")
            self.log.append(f"Total copper pixels: {cv2.countNonZero(copper_region)}")
            self.log.append(f"Corroded pixels: {cv2.countNonZero(corrosion)}")
            
            if boxes:
                self.log.append("")
                self.log.append("Individual region severity:")
                for idx, (x, y, w, h, sev) in enumerate(boxes, 1):
                    self.log.append(f"  Region {idx}: {sev:.1f}% (at x={x}, y={y}, size={w}x{h})")
            
            self.log.append("="*50)
            
        except Exception as e:
            self.log.append("="*50)
            self.log.append(f"CRITICAL ERROR: {str(e)}")
            self.log.append("="*50)
            import traceback
            self.log.append(traceback.format_exc())

    def run_vision_mode(self):
        if self.single_img is None:
            self.log.append("ERROR: Missing input image for vision mode")
            return

        try:
            self.log.append("="*50)
            self.log.append("VISION MODE (vision.py)")
            self.log.append("="*50)
            self.log.append("Starting vision-based detection...")
            QApplication.processEvents()

            self.log.append("Processing single PCB image...")
            QApplication.processEvents()
            
            results = run_vision_pipeline(self.single_img, show_labels=False)

            pixmap, size = cv_to_pixmap(results['result'])
            self.views["Vision Result"].setPixmapWithSize(pixmap, size)
            
            pixmap, size = cv_to_pixmap(results['pcb_processed'])
            self.views["Single PCB"].setPixmapWithSize(pixmap, size)
            
            pixmap, size = cv_to_pixmap(results['mask_copper'])
            self.views["Ref Copper"].setPixmapWithSize(pixmap, size)
            
            pixmap, size = cv_to_pixmap(results['mask_corrosion'])
            self.views["Corrosion Mask"].setPixmapWithSize(pixmap, size)
            
            overlay_inner = results['pcb_processed'].copy()
            overlay_inner[results['inner_copper'] == 255] = (0, 255, 255)
            pixmap, size = cv_to_pixmap(overlay_inner)
            self.views["Bad Copper"].setPixmapWithSize(pixmap, size)

            self.log.append("="*50)
            self.log.append(f"VISION MODE RESULTS:")
            self.log.append(f"Detected corrosion regions: {len(results['components'])}")
            self.log.append(f"Corrosion percentage: {results['percentage']:.2f}%")
            self.log.append(f"Total copper pixels: {cv2.countNonZero(results['mask_copper'])}")
            self.log.append(f"Corroded pixels: {cv2.countNonZero(results['mask_corrosion'])}")
            
            if results['components']:
                self.log.append("")
                self.log.append("Individual region severity:")
                for idx, (x, y, bw, bh, sev) in enumerate(results['components'], 1):
                    self.log.append(f"  Region {idx}: {sev:.1f}% (at x={x}, y={y}, size={bw}x{bh})")
            
            self.log.append("="*50)
            
        except Exception as e:
            self.log.append("="*50)
            self.log.append(f"CRITICAL ERROR: {str(e)}")
            self.log.append("="*50)
            import traceback
            self.log.append(traceback.format_exc())

    def run_hybrid_mode(self):
        if self.hybrid_img is None:
            self.log.append("ERROR: Missing input image for hybrid mode")
            return

        try:
            self.log.append("="*50)
            self.log.append("HYBRID MODE (hybrid.py)")
            self.log.append("="*50)
            self.log.append("Starting deep learning-based detection...")
            QApplication.processEvents()

            self.log.append("Loading models...")
            QApplication.processEvents()
            
            cnn_path = "cnn_pcb_pipeline_best.pth"
            unet_path = "unet_pcb_pipeline_best.pth"
            
            import os
            if not os.path.exists(cnn_path):
                self.log.append(f"WARNING: {cnn_path} not found, using untrained model")
                cnn_path = None
            if not os.path.exists(unet_path):
                self.log.append(f"WARNING: {unet_path} not found, using untrained model")
                unet_path = None
            
            self.log.append("Processing PCB image with hybrid pipeline...")
            QApplication.processEvents()
            
            results = run_hybrid_pipeline(
                self.hybrid_img,
                cnn_path=cnn_path,
                unet_path=unet_path,
                show_labels=False
            )

            pixmap, size = cv_to_pixmap(results['result'])
            self.views["Hybrid Result"].setPixmapWithSize(pixmap, size)
            
            pixmap, size = cv_to_pixmap(results['pcb_processed'])
            self.views["Hybrid PCB"].setPixmapWithSize(pixmap, size)
            
            pixmap, size = cv_to_pixmap(results['severity_heatmap'])
            self.views["Severity Heatmap"].setPixmapWithSize(pixmap, size)
            
            pixmap, size = cv_to_pixmap(results['mask_copper'])
            self.views["Ref Copper"].setPixmapWithSize(pixmap, size)
            
            pixmap, size = cv_to_pixmap(results['mask_corrosion'])
            self.views["Corrosion Mask"].setPixmapWithSize(pixmap, size)
            
            pixmap, size = cv_to_pixmap(results['unet_overlay'])
            self.views["Bad Copper"].setPixmapWithSize(pixmap, size)

            self.log.append("="*50)
            self.log.append(f"HYBRID MODE RESULTS:")
            self.log.append(f"Detected corrosion regions: {len(results['components'])}")
            self.log.append(f"Corrosion percentage: {results['percentage']:.2f}%")
            self.log.append(f"Total copper pixels: {cv2.countNonZero(results['mask_copper'])}")
            self.log.append(f"Corroded pixels: {cv2.countNonZero(results['mask_corrosion'])}")
            
            if results['components']:
                self.log.append("")
                self.log.append("Individual region severity (ResNet levels):")
                severity_map = results['severity_map']
                for idx, (x, y, bw, bh, cov) in enumerate(results['components'], 1):
                    region = severity_map[y:y+bh, x:x+bw]
                    level = int(region.max())
                    self.log.append(f"  Region {idx}: Level {level}, Coverage {cov:.1f}% (at x={x}, y={y}, size={bw}x{bh})")
            
            self.log.append("="*50)
            
        except Exception as e:
            self.log.append("="*50)
            self.log.append(f"CRITICAL ERROR: {str(e)}")
            self.log.append("="*50)
            import traceback
            self.log.append(traceback.format_exc())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = PCBApp()
    win.show()
    sys.exit(app.exec_())