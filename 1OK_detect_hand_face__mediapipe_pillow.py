# Detect ngón tay, khuôn mặt và thông tin các object trên mặt.

import cv2
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5.QtCore import QTimer, Qt, QSize

# Khởi tạo mô-đun MediaPipe Hands và Face Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

# Tải phông chữ Unicode
font_path = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"  # Đường dẫn đến phông chữ Unicode
font = ImageFont.truetype(font_path, 24)


# Hàm để xác định ngón tay nào đang giơ lên
def fingers_up(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Ngón cái
    if (
        hand_landmarks.landmark[tips_ids[0]].x
        < hand_landmarks.landmark[tips_ids[0] - 1].x
    ):
        fingers.append(1)
    else:
        fingers.append(0)

    # 4 ngón còn lại
    for id in range(1, 5):
        if (
            hand_landmarks.landmark[tips_ids[id]].y
            < hand_landmarks.landmark[tips_ids[id] - 2].y
        ):
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


# Hàm để hiển thị tên các ngón tay
def draw_finger_names(frame, hand_landmarks):
    finger_names = ["Ngón cái", "Ngón trỏ", "Ngón giữa", "Ngón áp út", "Ngón út"]
    tips_ids = [4, 8, 12, 16, 20]
    for i, tip_id in enumerate(tips_ids):
        x = int(hand_landmarks.landmark[tip_id].x * frame.shape[1])
        y = int(hand_landmarks.landmark[tip_id].y * frame.shape[0])
        # Chuyển khung hình sang định dạng Pillow để vẽ văn bản Unicode
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        draw.text((x, y), finger_names[i], font=font, fill=(0, 255, 0, 255))
        # Chuyển khung hình trở lại định dạng OpenCV
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return frame


# Hàm để vẽ dấu cộng trên trán
def draw_forehead_cross(frame, face_landmarks):
    for face in face_landmarks:
        # Lấy tọa độ trung điểm của hộp giới hạn khuôn mặt
        bboxC = face.location_data.relative_bounding_box
        ih, iw, _ = frame.shape
        x_min = int(bboxC.xmin * iw)
        y_min = int(bboxC.ymin * ih)
        box_width = int(bboxC.width * iw)
        box_height = int(bboxC.height * ih)

        x_center = x_min + box_width // 2
        y_forehead = y_min + box_height // 4  # Di chuyển lên để đặt trên trán

        # Vẽ hình vuông màu xanh bao quanh khuôn mặt
        cv2.rectangle(
            frame,
            (x_min, y_min),
            (x_min + box_width, y_min + box_height),
            (0, 255, 0),
            2,
        )

        # Vẽ dấu cộng với chấm tròn ở giữa
        cv2.drawMarker(
            frame,
            (x_center, y_forehead),
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=30,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
        cv2.circle(frame, (x_center, y_forehead), 5, (0, 255, 0), -1)

        # Chuyển khung hình sang định dạng Pillow để vẽ văn bản Unicode
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        text = "Khỏi chạy, head shoot"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.text(
            (x_min + (box_width - text_width) // 2, y_min - text_height - 10),
            text,
            font=font,
            fill=(255, 0, 0, 255),
        )
        # Chuyển khung hình trở lại định dạng OpenCV
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return frame


class VideoApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand and Face Tracking")

        self.lbl_video = QLabel(self)
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setScaledContents(False)

        self.btn_start = QPushButton("Start Tracking", self)
        self.btn_stop = QPushButton("Stop", self)

        layout = QVBoxLayout()
        layout.addWidget(self.lbl_video)
        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_stop)
        self.setLayout(layout)

        self.btn_start.clicked.connect(self.start_tracking)
        self.btn_stop.clicked.connect(self.stop_tracking)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        screen_geometry = QApplication.primaryScreen().geometry()
        self.setMaximumSize(
            int(screen_geometry.width() * 0.8), int(screen_geometry.height() * 0.8)
        )
        self.setMinimumSize(800, 600)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def start_tracking(self):
        if not self.cap:
            self.cap = cv2.VideoCapture(0)
        self.timer.start(10)

    def stop_tracking(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.lbl_video.clear()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Chuyển đổi khung hình sang RGB để phát hiện bàn tay và khuôn mặt
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)
            face_result = face_detection.process(rgb_frame)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
                    fingers = fingers_up(hand_landmarks)
                    num_fingers = sum(fingers)

                    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_image)
                    draw.text(
                        (10, 30),
                        f"Số ngón tay: {num_fingers}",
                        font=font,
                        fill=(255, 0, 0, 255),
                    )
                    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    frame = draw_finger_names(frame, hand_landmarks)

            if face_result.detections:
                frame = draw_forehead_cross(frame, face_result.detections)

            # Chuyển đổi khung hình trở lại định dạng RGB trước khi hiển thị
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = QImage(
                rgb_frame.data,
                rgb_frame.shape[1],
                rgb_frame.shape[0],
                rgb_frame.strides[0],
                QImage.Format_RGB888,
            )
            pixmap = QPixmap.fromImage(img)
            self.lbl_video.setPixmap(self.add_letterbox(pixmap, self.lbl_video.size()))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = QImage(
                    rgb_frame.data,
                    rgb_frame.shape[1],
                    rgb_frame.shape[0],
                    rgb_frame.strides[0],
                    QImage.Format_RGB888,
                )
                pixmap = QPixmap.fromImage(img)
                self.lbl_video.setPixmap(
                    self.add_letterbox(pixmap, self.lbl_video.size())
                )

    def add_letterbox(self, pixmap, size):
        if pixmap.isNull():
            return pixmap

        scaled_pixmap = pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        result_pixmap = QPixmap(size)
        result_pixmap.fill(Qt.black)
        painter = QPainter(result_pixmap)
        painter.drawPixmap(
            (size.width() - scaled_pixmap.width()) // 2,
            (size.height() - scaled_pixmap.height()) // 2,
            scaled_pixmap,
        )
        painter.end()
        return result_pixmap


if __name__ == "__main__":
    app = QApplication([])
    window = VideoApp()
    window.show()
    app.exec()
