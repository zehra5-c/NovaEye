import sys
import cv2
import time
import os
from datetime import datetime
from ultralytics import YOLO

from PySide6.QtWidgets import QApplication, QWidget, QLabel, QTextEdit, QPushButton, QVBoxLayout, QHBoxLayout
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer, Qt


class NovaEye(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("NovaEye AI Surveillance System")
        self.setGeometry(100, 100, 1250, 720)

        self.setStyleSheet("""
            QWidget {
                background-color: #0f1117;
                color: white;
                font-family: Segoe UI;
            }

            QLabel {
                border: 2px solid #1f2633;
                border-radius: 12px;
                background-color: #151922;
                color: white;
                font-size: 13pt;
            }

            QTextEdit {
                background-color: #151922;
                border: 2px solid #1f2633;
                border-radius: 12px;
                color: #00ff9d;
                padding: 10px;
                font-size: 11pt;
            }

            QPushButton {
                background-color: #1f2633;
                border: 2px solid #00ff9d;
                border-radius: 12px;
                padding: 10px;
                color: white;
                font-size: 11pt;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #283245;
            }
        """)

        self.camera_label = QLabel("CAMERA WAITING...")
        self.camera_label.setFixedSize(820, 520)
        self.camera_label.setAlignment(Qt.AlignCenter)

        self.status_label = QLabel("AI STATUS: WAITING")
        self.status_label.setFixedHeight(45)
        self.status_label.setAlignment(Qt.AlignCenter)

        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setFixedHeight(45)
        self.fps_label.setAlignment(Qt.AlignCenter)

        self.count_label = QLabel("TARGET COUNT: 0")
        self.count_label.setFixedHeight(45)
        self.count_label.setAlignment(Qt.AlignCenter)

        self.logs = QTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setFixedHeight(330)

        self.start_button = QPushButton("Kamerayı Başlat")
        self.screenshot_button = QPushButton("Ekran Görüntüsü Al")
        self.record_button = QPushButton("Video Kaydı Başlat")

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.status_label)
        right_layout.addWidget(self.fps_label)
        right_layout.addWidget(self.count_label)
        right_layout.addWidget(self.logs)
        right_layout.addWidget(self.start_button)
        right_layout.addWidget(self.screenshot_button)
        right_layout.addWidget(self.record_button)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.camera_label)
        main_layout.addLayout(right_layout)

        self.setLayout(main_layout)

        self.camera = cv2.VideoCapture(0)
        self.model = YOLO("yolo11n.pt")

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.last_log_time = 0
        self.prev_time = time.time()
        self.current_frame = None
        self.recording = False
        self.video_writer = None

        self.label_tr = {
            "person": "İNSAN",
            "bird": "KUŞ",
            "cat": "KEDİ",
            "dog": "KÖPEK",
            "cell phone": "TELEFON",
            "laptop": "LAPTOP",
            "bottle": "ŞİŞE",
            "chair": "SANDALYE",
            "car": "ARABA",
            "backpack": "ÇANTA"
        }

        self.start_button.clicked.connect(self.start_camera)
        self.screenshot_button.clicked.connect(self.take_screenshot)
        self.record_button.clicked.connect(self.toggle_recording)

    def start_camera(self):
        self.logs.append("[SİSTEM] Kamera başlatıldı")
        self.logs.append("[AI] YOLO nesne algılama aktif")
        self.status_label.setText("AI STATUS: ONLINE")
        self.timer.start(30)

    def translate_label(self, label):
        return self.label_tr.get(label, label.upper())

    def take_screenshot(self):
        if self.current_frame is None:
            self.logs.append("[UYARI] Kaydedilecek görüntü yok")
            return
    def toggle_recording(self):
        if not self.recording:
            folder_path = "assets/records"
            os.makedirs(folder_path, exist_ok=True)

            file_name = datetime.now().strftime("novaeye_record_%Y%m%d_%H%M%S.avi")
            file_path = os.path.join(folder_path, file_name)

            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self.video_writer = cv2.VideoWriter(file_path, fourcc, 20.0, (640, 480))

            self.recording = True
            self.record_button.setText("Video Kaydını Durdur")
            self.logs.append(f"[KAYIT] Video kaydı başladı: {file_name}")

        else:
            self.recording = False

            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None

            self.record_button.setText("Video Kaydı Başlat")
            self.logs.append("[KAYIT] Video kaydı durduruldu")

        folder_path = "assets/screenshots"
        os.makedirs(folder_path, exist_ok=True)

        file_name = datetime.now().strftime("novaeye_%Y%m%d_%H%M%S.png")
        file_path = os.path.join(folder_path, file_name)

        cv2.imwrite(file_path, self.current_frame)
        self.logs.append(f"[KAYIT] Ekran görüntüsü kaydedildi: {file_name}")

    def update_frame(self):
        ret, frame = self.camera.read()

        if not ret:
            return

        frame = cv2.flip(frame, 1)

        results = self.model(frame, verbose=False)

        detected_objects = []

        for result in results:
            boxes = result.boxes

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])

                if confidence < 0.5:
                    continue

                class_id = int(box.cls[0])
                label = self.model.names[class_id]
                label_text = self.translate_label(label)

                if label == "person":
                   label_text = "TARGET-01 | İNSAN"

                detected_objects.append(label_text)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 120), 3)

                cv2.putText(
                    frame,
                    f"{label_text} {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 120),
                    2
                )

        target_count = len(detected_objects)
        self.count_label.setText(f"TARGET COUNT: {target_count}")

        if detected_objects:
            self.status_label.setText(
                f"AI STATUS: {detected_objects[0]} DETECTED"
            )

            current_time = time.time()

            if current_time - self.last_log_time > 2:
                detected_text = ", ".join(detected_objects)
                self.logs.append(f"[YOLO] Algılandı: {detected_text}")
                self.last_log_time = current_time
        else:
            self.status_label.setText("AI STATUS: ONLINE")

        current_time = time.time()
        fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time
        self.fps_label.setText(f"FPS: {int(fps)}")

        self.current_frame = frame.copy()

        if self.recording and self.video_writer is not None:
            record_frame = cv2.resize(frame, (640, 480))
            self.video_writer.write(record_frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        height, width, channel = frame.shape
        bytes_per_line = channel * width

        q_image = QImage(
            frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888
        )

        pixmap = QPixmap.fromImage(q_image)

        self.camera_label.setPixmap(
            pixmap.scaled(
                self.camera_label.width(),
                self.camera_label.height(),
                Qt.KeepAspectRatio
            )
        )

    def closeEvent(self, event):
        if self.video_writer is not None:
           self.video_writer.release()
       
        self.camera.release()
        event.accept()


app = QApplication(sys.argv)

window = NovaEye()
window.show()

sys.exit(app.exec())