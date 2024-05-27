# Detect các bộ phận trên cơ thể và thông tin liên quan.

import cv2
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5.QtCore import QTimer, Qt, QSize

# Khởi tạo các module MediaPipe cho bàn tay, lưới mặt và tư thế
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Tải phông chữ Unicode
font_path = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"  # Đường dẫn đến phông chữ Unicode
font = ImageFont.truetype(font_path, 24)

# Hàm xác định các ngón tay đang giơ lên
def fingers_up(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Xác định xem bàn tay là tay trái hay tay phải
    # Kiểm tra vị trí tương đối của cổ tay (điểm đánh dấu 0) và MCP của ngón giữa (điểm đánh dấu 9)
    if hand_landmarks.landmark[0].x < hand_landmarks.landmark[9].x:
        # Tay phải
        is_right_hand = True
    else:
        # Tay trái
        is_right_hand = False

    # Ngón cái
    if is_right_hand:
        if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)
    else:
        if hand_landmarks.landmark[tips_ids[0]].x > hand_landmarks.landmark[tips_ids[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)

    # Bốn ngón còn lại
    for id in range(1, 5):
        if hand_landmarks.landmark[tips_ids[id]].y < hand_landmarks.landmark[tips_ids[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

# Hàm hiển thị tên của các ngón tay
def draw_finger_names(frame, hand_landmarks):
    finger_names = ["Ngón cái", "Ngón trỏ", "Ngón giữa", "Ngón áp út", "Ngón út"]
    tips_ids = [4, 8, 12, 16, 20]
    for i, tip_id in enumerate(tips_ids):
        x = int(hand_landmarks.landmark[tip_id].x * frame.shape[1])
        y = int(hand_landmarks.landmark[tip_id].y * frame.shape[0])
        # Chuyển đổi frame sang định dạng Pillow để vẽ văn bản Unicode
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        draw.text((x, y), finger_names[i], font=font, fill=(0, 0, 255, 255))
        # Chuyển đổi frame trở lại định dạng OpenCV
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return frame

# Vẽ tên
def draw_name_info(frame, x, y, text, line_height=10, opacity=30):
    # Chuyển đổi frame sang định dạng Pillow
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image, 'RGBA')
    
    words = text.split()
    max_text_width = 0
    total_text_height = 0
    
    # Tính toán kích thước của hộp nền
    text_heights = []
    text_widths = []
    for word in words:
        text_bbox = draw.textbbox((0, 0), word, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        max_text_width = max(max_text_width, text_width)
        total_text_height += text_height + line_height
        text_heights.append(text_height)
        text_widths.append(text_width)
    
    # Loại bỏ chiều cao dòng cuối cùng
    total_text_height -= line_height
    
    # Thêm khoảng cách lề nhãn nội dung
    padding = 10
    box_width = max_text_width + 2 * padding
    box_height = total_text_height + 2 * padding
    
    # Điều chỉnh vị trí y để căn giữa văn bản theo chiều dọc trong khung
    frame_height = frame.shape[0]
    y = (frame_height - box_height) // 2
    
    # Tính toán màu nền với độ mờ
    fill_color = (255, 255, 0, int(255 * (opacity / 100)))
    
    # Vẽ nhãn nội dung
    draw.rectangle([(x, y), (x + box_width, y + box_height)], fill=fill_color, outline=(255, 0, 0), width=2)
    
    # Vẽ từng từ trên khung
    current_y = y + padding
    for i, word in enumerate(words):
        text_x = x + padding + (max_text_width - text_widths[i]) // 2
        draw.text((text_x, current_y), word, font=font, fill=(0, 0, 0, 255))
        current_y += text_heights[i] + line_height
    
    # Chuyển đổi frame trở lại định dạng OpenCV
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return frame

# Hàm vẽ thông tin trên mặt
def draw_face_info(frame, face_landmarks):
    ih, iw, _ = frame.shape

    def draw_line_with_label(frame, start_point, label, angle, length, vertical_offset):
        end_point_1 = (int(start_point[0] - length * np.cos(np.radians(angle))),
                    int(start_point[1] - length * np.sin(np.radians(angle))))
        end_point_2 = (end_point_1[0], end_point_1[1] - vertical_offset)

        cv2.line(frame, start_point, end_point_1, (0, 255, 0), 2)
        cv2.line(frame, end_point_1, end_point_2, (0, 255, 0), 2)

        # Chuyển đổi frame sang định dạng Pillow để vẽ văn bản Unicode
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        # Tính toán kích thước văn bản và vị trí để căn giữa theo chiều ngang
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = end_point_2[0] - (text_width // 2)
        text_y = end_point_2[1] - text_height

        draw.text((text_x, text_y), label, font=font, fill=(255, 0, 0, 255))

        # Chuyển đổi frame trở lại định dạng OpenCV
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return frame
    
    for face_landmark in face_landmarks:
        # Chỉ số các điểm đánh dấu trên mặt
        left_eye_inner_index = 362
        right_eye_outer_index = 133
        left_eye_indices = [362, 385, 387, 263, 373, 380]
        right_eye_indices = [33, 160, 158, 133, 153, 144]
        nose_tip_index = 1
        mouth_left_index = 61
        mouth_right_index = 291
        forehead_index = 10
        chin_index = 152
        left_ear_index = 234
        right_ear_index = 454
        left_cheek_index = 50
        right_cheek_index = 280

        # Lấy tọa độ của các đặc điểm trên mặt
        left_eye_inner_coords = (int(face_landmark.landmark[left_eye_inner_index].x * iw), int(face_landmark.landmark[left_eye_inner_index].y * ih))
        right_eye_outer_coords = (int(face_landmark.landmark[right_eye_outer_index].x * iw), int(face_landmark.landmark[right_eye_outer_index].y * ih))
        right_eye_coords = np.mean([(face_landmark.landmark[i].x * iw, face_landmark.landmark[i].y * ih) for i in left_eye_indices], axis=0).astype(int)
        left_eye_coords = np.mean([(face_landmark.landmark[i].x * iw, face_landmark.landmark[i].y * ih) for i in right_eye_indices], axis=0).astype(int)
        nose_tip_coords = (int(face_landmark.landmark[nose_tip_index].x * iw), int(face_landmark.landmark[nose_tip_index].y * ih))
        mouth_left_coords= (int(face_landmark.landmark[mouth_left_index].x * iw), int(face_landmark.landmark[mouth_left_index].y * ih))
        mouth_right_coords = (int(face_landmark.landmark[mouth_right_index].x * iw), int(face_landmark.landmark[mouth_right_index].y * ih))
        forehead_coords = (int(face_landmark.landmark[forehead_index].x * iw), int(face_landmark.landmark[forehead_index].y * ih))
        chin_coords = (int(face_landmark.landmark[chin_index].x * iw), int(face_landmark.landmark[chin_index].y * ih))
        left_ear_coords = (int(face_landmark.landmark[left_ear_index].x * iw), int(face_landmark.landmark[left_ear_index].y * ih))
        right_ear_coords = (int(face_landmark.landmark[right_ear_index].x * iw), int(face_landmark.landmark[right_ear_index].y * ih))
        right_cheek_coords = (int(face_landmark.landmark[left_cheek_index].x * iw), int(face_landmark.landmark[left_cheek_index].y * ih))
        left_cheek_coords = (int(face_landmark.landmark[right_cheek_index].x * iw), int(face_landmark.landmark[right_cheek_index].y * ih))

        # Vẽ các điểm trên mặt
        cv2.circle(frame, left_eye_inner_coords, 5, (0, 255, 0), -1)
        cv2.circle(frame, right_eye_outer_coords, 5, (0, 255, 0), -1)
        cv2.circle(frame, tuple(left_eye_coords), 5, (0, 255, 0), -1)
        cv2.circle(frame, tuple(right_eye_coords), 5, (0, 255, 0), -1)
        cv2.circle(frame, nose_tip_coords, 5, (0, 255, 0), -1)
        cv2.circle(frame, mouth_left_coords, 5, (0, 255, 0), -1)
        cv2.circle(frame, mouth_right_coords, 5, (0, 255, 0), -1)
        cv2.circle(frame, forehead_coords, 5, (0, 255, 0), -1)
        cv2.circle(frame, chin_coords, 5, (0, 255, 0), -1)
        cv2.circle(frame, left_ear_coords, 5, (0, 255, 0), -1)
        cv2.circle(frame, right_ear_coords, 5, (0, 255, 0), -1)
        cv2.circle(frame, left_cheek_coords, 5, (0, 255, 0), -1)
        cv2.circle(frame, right_cheek_coords, 5, (0, 255, 0), -1)

        # Vẽ các đường và nhãn
        vertical_offset = 40
        eye_coords_vertical_offset = 120
        ear_coords_vertical_offset = 100
        length_offset = 100
        forehead_length_offset = 40  # Đường Trán
        cheek_coords_length_offset = 180
        mouth_length_offset = 220  # Đường Miệng
        chin_length_offset = 100  # Đường Cằm

        eye_vertical_offset = 80
        eye_length_offset = 320

        frame = draw_line_with_label(frame, nose_tip_coords, "Mũi", 205, 100, vertical_offset)
        frame = draw_line_with_label(frame, forehead_coords, "Trán", 105, forehead_length_offset, 40)
        frame = draw_line_with_label(frame, chin_coords, "Cằm", 205, chin_length_offset, 20)

        frame = draw_line_with_label(frame, right_eye_outer_coords, "Mắt phải trong", 25, eye_length_offset, eye_vertical_offset)
        frame = draw_line_with_label(frame, left_eye_inner_coords, "Mắt trái trong", 155, eye_length_offset, eye_vertical_offset)
        
        frame = draw_line_with_label(frame, tuple(left_eye_coords), "Mắt phải", 25, length_offset, eye_coords_vertical_offset)
        frame = draw_line_with_label(frame, tuple(right_eye_coords), "Mắt trái", 155, length_offset, eye_coords_vertical_offset)

        frame = draw_line_with_label(frame, mouth_left_coords, "Miệng phải", 0, mouth_length_offset, vertical_offset)
        frame = draw_line_with_label(frame, mouth_right_coords, "Miệng trái", 180, mouth_length_offset, vertical_offset)
        
        frame = draw_line_with_label(frame, left_ear_coords, "Tai phải", 25, length_offset, ear_coords_vertical_offset)
        frame = draw_line_with_label(frame, right_ear_coords, "Tai trái", 155, length_offset, ear_coords_vertical_offset)

        frame = draw_line_with_label(frame, right_cheek_coords, "Má phải", 25, cheek_coords_length_offset, vertical_offset)
        frame = draw_line_with_label(frame, left_cheek_coords, "Má trái", 155, cheek_coords_length_offset, vertical_offset)

    return frame

# Hàm vẽ thông tin trên cơ thể
def draw_pose_info(frame, pose_landmarks):
    if pose_landmarks:
        ih, iw, _ = frame.shape
        landmarks = pose_landmarks.landmark
        
        # Lấy tọa độ của các phần cơ thể
        left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * iw), int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * ih))
        right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * iw), int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * ih))
        left_elbow = (int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * iw), int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * ih))
        right_elbow = (int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x * iw), int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y * ih))
        left_wrist = (int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * iw), int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * ih))
        right_wrist = (int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * iw), int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * ih))
        left_hip = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * iw), int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * ih))
        right_hip = (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * iw), int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * ih))
        left_knee = (int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x * iw), int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * ih))
        right_knee = (int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x * iw), int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y * ih))
        left_ankle = (int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x * iw), int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * ih))
        right_ankle = (int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x * iw), int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * ih))
        
        # Vẽ các điểm trên cơ thể
        body_parts = {
            "Vai trái": left_shoulder, "Vai phải": right_shoulder, 
            "Khuỷu tay trái": left_elbow, "Khuỷu tay phải": right_elbow, 
            "Cổ tay trái": left_wrist, "Cổ tay phải": right_wrist, 
            "Hông trái": left_hip, "Hông phải": right_hip, 
            "Gối trái": left_knee, "Gối phải": right_knee, 
            "Cổ chân trái": left_ankle, "Cổ chân phải": right_ankle
        }
        
        for part, coords in body_parts.items():
            cv2.circle(frame, coords, 5, (0, 255, 0), -1)
            
            # Chuyển đổi frame sang định dạng Pillow để vẽ văn bản Unicode
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            draw.text(coords, part, font=font, fill=(255, 0, 0, 255))
            # Chuyển đổi frame trở lại định dạng OpenCV
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
        self.setMaximumSize(int(screen_geometry.width() * 0.8), int(screen_geometry.height() * 0.8))
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
            face_result = face_mesh.process(rgb_frame)
            pose_result = pose.process(rgb_frame)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    fingers = fingers_up(hand_landmarks)
                    num_fingers = sum(fingers)
                    
                    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_image)
                    draw.text((10, 30), f'Số ngón tay: {num_fingers}', font=font, fill=(255, 0, 0, 255))
                    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    frame = draw_finger_names(frame, hand_landmarks)

            if face_result.multi_face_landmarks:
                frame = draw_face_info(frame, face_result.multi_face_landmarks)
                frame = draw_name_info(frame, 10, 0, 'CÙ KIM NGỌC', line_height=20, opacity=20)

            if pose_result.pose_landmarks:
                frame = draw_pose_info(frame, pose_result.pose_landmarks)
            
            # Chuyển đổi khung hình trở lại RGB trước khi hiển thị
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = QImage(rgb_frame.data, rgb_frame.shape[1], rgb_frame.shape[0], rgb_frame.strides[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(img)
            self.lbl_video.setPixmap(self.add_letterbox(pixmap, self.lbl_video.size()))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = QImage(rgb_frame.data, rgb_frame.shape[1], rgb_frame.shape[0], rgb_frame.strides[0], QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(img)
                self.lbl_video.setPixmap(self.add_letterbox(pixmap, self.lbl_video.size()))

    def add_letterbox(self, pixmap, size):
        if pixmap.isNull():
            return pixmap

        scaled_pixmap = pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        result_pixmap = QPixmap(size)
        result_pixmap.fill(Qt.black)
        painter = QPainter(result_pixmap)
        painter.drawPixmap((size.width() - scaled_pixmap.width()) // 2,
                           (size.height() - scaled_pixmap.height()) // 2,
                           scaled_pixmap)
        painter.end()
        return result_pixmap

if __name__ == "__main__":
    app = QApplication([])
    window = VideoApp()
    window.show()
    app.exec()
