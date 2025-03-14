import sys
import os
import cv2
import torch
import numpy as np
import datetime
import pandas as pd
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QHBoxLayout, QTextEdit
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer
from ultralytics import YOLO

class FireDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
        # Get the directory of the script (or EXE after compilation)
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_path, "firedetectv2.pt")
        
        self.model = YOLO(model_path)  # Load model dynamically
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.detect_fire)
        self.fire_log = []

    def initUI(self):
        self.setWindowTitle("Fire Detection App")
        self.setGeometry(100, 100, 900, 600)
        
        self.label = QLabel(self)
        self.label.setFixedSize(640, 480)
        
        self.alert_box = QTextEdit(self)
        self.alert_box.setFixedSize(220, 480)
        self.alert_box.setReadOnly(True)
        self.alert_box.setStyleSheet("color: red; font-size: 14px; font-weight: bold;")
        
        self.detect_button = QPushButton("Detect", self)
        self.detect_button.clicked.connect(self.start_detection)
        
        self.save_button = QPushButton("Save Detection", self)
        self.save_button.clicked.connect(self.save_to_excel)
        
        layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.label)
        left_layout.addWidget(self.detect_button)
        left_layout.addWidget(self.save_button)
        layout.addLayout(left_layout)
        layout.addWidget(self.alert_box)
        self.setLayout(layout)
    
    def start_detection(self):
        self.cap = cv2.VideoCapture(0)  
        self.timer.start(30)  

    def detect_fire(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        
        results = self.model(frame)  # Run YOLO model
        fire_detected = False
        
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Fire", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                fire_detected = True
        
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if fire_detected:
            alert_message = f"Fire detected! Time: {current_time}\n"
            self.fire_log.append([current_time, "Fire detected"])
        else:
            alert_message = "No fire detected\n"
        
        self.alert_box.append(alert_message)
        self.display_image(frame)
    
    def save_to_excel(self):
        df = pd.DataFrame(self.fire_log, columns=["Timestamp", "Status"])
        df.to_excel("fire_detection_log.xlsx", index=False)
    
    def display_image(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qimg))
    
    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        self.timer.stop()
        self.save_to_excel()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FireDetectionApp()
    window.show()
    sys.exit(app.exec())
