######################### Ghi chú #########################
Code detect chỉ mang mục đích học tập, tham khảo nhé!


######################### Thư viện UI Python #########################
Tkinter với ttkthemes: Đơn giản và tích hợp sẵn trong Python, dễ sử dụng.
PyQt với QDarkStyle: Mạnh mẽ, linh hoạt, nhiều tính năng nâng cao.
Kivy: Phát triển ứng dụng đa nền tảng, hiện đại.
CustomTkinter: Hiện đại hơn so với Tkinter thông thường.

Lỗi hiển thị font unicode trên OpenCV chuyển sử dụng thư viện Pillow.
`pip install mediapipe opencv-python pillow`

######################### YOLO #########################
# Yolo version 4 và 7 là bản phát triển ban đầu. Các phiên bản của ultra là khác.
# Hiện tại bản 4 và 7 nhẹ và ổn định để tracking thông thường.

######################### WinOS #########################
1. Cài đặt python. Thiết lập biến môi trường đến thư mục cài đặt Python và Python\Scripts
2. Cài đặt pip nếu chưa:
`python get-pip.py`
Check version: `pip --version`

3. Càid đặt gói cần thiết:
`pip install <tên_gói>`

4. Chạy các ứng dụng Build bình thường.

######################### Trên MacOS #########################
Build Python dùng lệnh: `python3 file_name.py`

1. Kiểm tra cài đặt Python
python3 --version
pip3 --version

2. Cài đặt Homebrew nếu chưa có:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

3. Cài đặt Python + Pip
brew install python

4. Thiết lập biến môi trường
=> Open: `nano ~/.zshrc`
=> Add: export PATH="/usr/local/bin:/usr/local/opt/python/libexec/bin:$PATH"

5. Update thay đổi
`source ~/.zshrc`

6. Kiểm tra Pip
`pip3 --version`

7. Tiến hành cài các gói cần thiết.
Ví dụ: `pip3 install ttkthemes pyqt5 qdarkstyle kivy customtkinter`
`pip3 install pywin32`