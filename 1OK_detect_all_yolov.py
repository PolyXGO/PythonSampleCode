import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QHBoxLayout
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PIL import Image, ImageDraw, ImageFont

# Đường dẫn tới tệp cấu hình và trọng số của YOLOv4-tiny. Detect đơn giản và cần nhẹ mọi người cần nhắc dùng v4 này nhé!
dest_dir = './models/yolov4-tiny/'
cfg_path = os.path.join(dest_dir, 'yolov4-tiny.cfg')
weights_path = os.path.join(dest_dir, 'yolov4-tiny.weights')
names_path = os.path.join(dest_dir, 'coco.names')

# Tải các lớp từ tệp coco.names
with open(names_path, 'r') as f:
    classes = f.read().strip().split('\n')

# Khởi tạo model YOLOv4-tiny
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Chuyển thành GPU nếu có GPU

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        # Mở camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                self.change_pixmap_signal.emit(frame)
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Detection")
        self.disply_width = 640
        self.display_height = 480
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        
        self.initUI()
        
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        
    def initUI(self):
        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        
        hbox.addWidget(self.start_button)
        hbox.addWidget(self.stop_button)
        
        vbox.addLayout(hbox)
        vbox.addWidget(self.image_label)
        
        self.setLayout(vbox)
        
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        
    @pyqtSlot()
    def start_camera(self):
        if not self.thread.isRunning():
            self.thread = VideoThread()
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.start()
        
    @pyqtSlot()
    def stop_camera(self):
        if self.thread.isRunning():
            self.thread.stop()

    def resizeEvent(self, event):
        # Cập nhật lại kích thước của QLabel khi cửa sổ thay đổi kích thước
        self.disply_width = self.image_label.width()
        self.display_height = self.image_label.height()
        super().resizeEvent(event)

    def update_image(self, cv_img):
        processed_img = self.detect_objects(cv_img)
        qt_img = self.convert_cv_qt(processed_img)
        self.image_label.setPixmap(qt_img)

    def detect_objects(self, img):
        height, width = img.shape[:2]
        # Thay đổi kích thước hình ảnh để xử lý nhanh hơn
        resized_img = cv2.resize(img, (320, 320))
        blob = cv2.dnn.blobFromImage(resized_img, 1 / 255.0, (320, 320), swapRB=True, crop=False)
        net.setInput(blob)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
        detections = net.forward(output_layers)
        
        class_ids = []
        confidences = []
        boxes = []
        for output in detections:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        # Chuyển đổi hình ảnh sang định dạng Pillow để xử lý thêm
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.load_default()
        
        if len(indices) > 0:
            for i, index in enumerate(indices.flatten()):
                box = boxes[index]
                x, y, w, h = box
                label = str(classes[class_ids[index]])
                color = (0, 255, 0)
                draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
                draw.text((x, y - 10), f"{label} {i+1}", font=font, fill=color)
        
        # Chuyển đổi hình ảnh từ định dạng Pillow trở lại định dạng OpenCV
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return img

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, aspectRatioMode = Qt.KeepAspectRatio)
        
        # Tạo hình ảnh nền màu đen
        black_background = QImage(self.disply_width, self.display_height, QImage.Format_RGB888)
        black_background.fill(Qt.black)
        
        # Tạo QPainter để vẽ video lên nền đen
        painter = QPainter(black_background)
        x_offset = (self.disply_width - p.width()) // 2
        y_offset = (self.display_height - p.height()) // 2
        painter.drawImage(x_offset, y_offset, p)
        painter.end()
        
        return QPixmap.fromImage(black_background)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())
