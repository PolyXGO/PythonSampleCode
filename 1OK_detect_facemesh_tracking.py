import cv2
import mediapipe as mp
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5.QtCore import QTimer, Qt

# Define the color for the face mesh connections
MESH_COLOR = (0, 255, 0)  # Green color

# Initialize MediaPipe face mesh solution
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Initialize MediaPipe hands solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Initialize MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class FaceMeshApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FaceMesh and Hand Tracking")

        # Label to display video frames
        self.lbl_video = QLabel(self)
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setScaledContents(False)

        # Buttons to start and stop tracking
        self.btn_start = QPushButton("Start", self)
        self.btn_stop = QPushButton("Stop", self)

        # Set up the layout for the application
        layout = QVBoxLayout()
        layout.addWidget(self.lbl_video)
        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_stop)
        self.setLayout(layout)

        # Connect button signals to their respective slots
        self.btn_start.clicked.connect(self.start_tracking)
        self.btn_stop.clicked.connect(self.stop_tracking)

        # Initialize video capture and timer
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Configure window size constraints
        screen_geometry = QApplication.primaryScreen().geometry()
        self.setMaximumSize(
            int(screen_geometry.width() * 0.8), int(screen_geometry.height() * 0.8)
        )
        self.setMinimumSize(800, 600)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def start_tracking(self):
        # Start video capture and timer
        if not self.cap:
            self.cap = cv2.VideoCapture(0)
        self.timer.start(10)

    def stop_tracking(self):
        # Stop the timer and release video capture
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.lbl_video.clear()

    def update_frame(self):
        # Capture and process video frames
        ret, frame = self.cap.read()
        if ret:
            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame for face landmarks
            face_result = face_mesh.process(rgb_frame)

            # Process the frame for hand landmarks
            hand_result = hands.process(rgb_frame)

            # Draw face landmarks on the frame
            if face_result.multi_face_landmarks:
                for face_landmarks in face_result.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            color=MESH_COLOR, thickness=1, circle_radius=1
                        ),
                    )

            # Draw hand landmarks on the frame
            if hand_result.multi_hand_landmarks:
                for hand_landmarks in hand_result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=hand_landmarks,
                        connections=mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(
                            color=MESH_COLOR, thickness=2, circle_radius=2
                        ),
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            color=MESH_COLOR, thickness=2, circle_radius=2
                        ),
                    )

            # Convert the processed frame to QImage format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = QImage(
                rgb_frame.data,
                rgb_frame.shape[1],
                rgb_frame.shape[0],
                rgb_frame.strides[0],
                QImage.Format_RGB888,
            )
            pixmap = QPixmap.fromImage(img)

            # Add letterbox effect to the pixmap and display it
            self.lbl_video.setPixmap(self.add_letterbox(pixmap, self.lbl_video.size()))

    def resizeEvent(self, event):
        # Handle resize events to maintain video scaling
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
        # Add letterbox effect to the pixmap for maintaining aspect ratio
        if pixmap.isNull():
            return pixmap

        # Scale the pixmap while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Create a new pixmap filled with black as the background
        result_pixmap = QPixmap(size)
        result_pixmap.fill(Qt.black)

        # Draw the scaled pixmap onto the black background
        painter = QPainter(result_pixmap)
        painter.drawPixmap(
            (size.width() - scaled_pixmap.width()) // 2,
            (size.height() - scaled_pixmap.height()) // 2,
            scaled_pixmap,
        )
        painter.end()
        return result_pixmap


if __name__ == "__main__":
    # Initialize the application and display the window
    app = QApplication([])
    window = FaceMeshApp()
    window.show()
    app.exec()
