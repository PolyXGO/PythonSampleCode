#Setup `pip install comtypes`, `pip install pycaw`
# Sử dụng khoảng cách ngón trỏ và ngón cái để tăng giảm âm lượng âm thanh hệ thống. Trên Windows mọi người build sẽ cần bổ sung thư viện python sẽ yêu cầu trên thông báo lỗi nhé!

import cv2
import sys
import mediapipe as mp
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, pyqtSlot, Qt
from PyQt5.QtGui import QImage, QPixmap
from math import hypot
import os

def set_volume_mac(volume):
    """
    Điều chỉnh âm lượng hệ thống trên macOS.
    :param volume: Một số nguyên từ 0 (tắt tiếng) đến 100 (âm lượng tối đa).
    """
    volume = max(0, min(100, volume))  # Đảm bảo âm lượng trong phạm vi hợp lệ
    os.system(f"osascript -e 'set volume output volume {volume}'")

def draw_volume_bar(image, volume):
    """
    Vẽ thanh âm lượng và phần trăm âm lượng trên hình ảnh.
    :param image: Khung hình để vẽ.
    :param volume: Phần trăm âm lượng hiện tại.
    """
    h, w, _ = image.shape
    bar_height = 300  # Đặt độ cao cố định của khung viền và thanh loading
    bar_width = 20
    bar_x = w - bar_width - 20  # Vị trí thanh loading
    bar_y = (h - bar_height) // 2  # Canh giữa theo chiều dọc
    
    # Vẽ khung viền
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
    
    # Tính chiều cao của thanh loading dựa trên volume
    loading_height = int(bar_height * (volume / 100))
    loading_y = bar_y + (bar_height - loading_height)
    
    # Vẽ thanh loading
    cv2.rectangle(image, (bar_x, loading_y), (bar_x + bar_width, bar_y + bar_height), (0, 255, 0), -1)
    
    # Vẽ nhãn phần trăm âm lượng
    cv2.putText(image, f'{int(volume)}%', (bar_x - 70, bar_y + bar_height + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

    def run(self):
        while self._run_flag:
            ret, frame = self.cap.read()
            if ret:
                # Chuyển đổi hình ảnh từ BGR sang RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = self.hands.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(
                            image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        
                        # Đánh dấu đầu ngón tay cái và trỏ bằng chấm đỏ
                        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                        index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        h, w, _ = image.shape
                        thumb_tip_coords = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                        index_finger_tip_coords = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))

                        cv2.circle(image, thumb_tip_coords, 10, (0, 0, 255), -1)
                        cv2.circle(image, index_finger_tip_coords, 10, (0, 0, 255), -1)

                        # Vẽ đường kết nối màu xanh giữa hai đầu ngón tay
                        cv2.line(image, thumb_tip_coords, index_finger_tip_coords, (0, 255, 0), 5)

                        # Tính khoảng cách giữa đầu ngón tay cái và trỏ
                        distance = hypot(index_finger_tip_coords[0] - thumb_tip_coords[0],
                                         index_finger_tip_coords[1] - thumb_tip_coords[1])
                        
                        # Điều chỉnh âm lượng dựa trên khoảng cách
                        volume_level = np.interp(distance, [30, 300], [0, 100])
                        set_volume_mac(volume_level)
                        
                        # Vẽ cột màu và nhãn âm lượng
                        draw_volume_bar(image, volume_level)

                self.change_pixmap_signal.emit(image)

    def stop(self):
        self._run_flag = False
        self.cap.release()
        self.quit()
        self.wait()

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Finger Detection with MediaPipe")
        self.setMinimumSize(800, 600)
        self.disply_width = 640
        self.display_height = 480

        # Tạo nhãn hiển thị video
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)

        # Tạo nút bắt đầu và dừng
        self.start_button = QPushButton('Start')
        self.start_button.clicked.connect(self.start_video)
        self.stop_button = QPushButton('Stop')
        self.stop_button.clicked.connect(self.stop_video)
        
        self.timer = QTimer()

        # Tạo layout và thêm các widget
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.start_button)
        vbox.addWidget(self.stop_button)

        # Đặt widget trung tâm
        widget = QWidget()
        widget.setLayout(vbox)
        self.setCentralWidget(widget)

        # Tạo luồng video capture
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)

    def start_video(self):
        self.thread.start()

    def stop_video(self):
        self.thread.stop()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())
