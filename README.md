## Ghi chú

Code detect chỉ mang mục đích học tập, tham khảo nhé! Code hay hay mình sẽ cập nhật dần dần và chia theo từng tập tin độc lập.
Phần tính năng từng file code ghi chú ở đầu mỗi tập tin.

## Thư viện UI Python

**Tkinter** với **ttkthemes**: Đơn giản và tích hợp sẵn trong Python, dễ sử dụng.<br/>
**PyQt** với **QDarkStyle**: Mạnh mẽ, linh hoạt, nhiều tính năng nâng cao.<br/>
**Kivy**: Phát triển ứng dụng đa nền tảng, hiện đại.<br/>
**CustomTkinter**: Hiện đại hơn so với Tkinter thông thường.<br/>

## WinOS

1. Cài đặt python. Thiết lập biến môi trường đến thư mục cài đặt Python và Python\Scripts
2. Cài đặt pip hoặc pip3 nếu chưa: `python get-pip.py`

Kiểm tra phiên bản pip version: `pip --version`

3. Cú pháp đặt gói cần thiết (nếu cần):
`pip install <tên_gói>` hoặc `pip3 install <tên_gói>`

4. Chạy các ứng dụng Build bình thường như code ứng dụng Windows form: dùng `Ctrl + F5`

## MacOS

1. Kiểm tra cài đặt Python
`python3 --version`
`pip3 --version`

2. Cài đặt Homebrew nếu chưa có:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

3. Cài đặt Python + Pip
`brew install python`

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

8. Build Python dùng lệnh Terminal tại thư mục dự án: `python3 file_name.py` hoặc `python file_name.py` nếu dùng version cũ hơn.

## NOTE

Để hiển font unicode trên OpenCV sử dụng thư viện Pillow.
`pip install mediapipe opencv-python pillow` hoặc `pip3 install mediapipe opencv-python pillow`

Lỗi No module named 'PyQt5' cần cài đặt PyQt5.
`pip install PyQt5 qdarkstyle` hoặc `pip3 install PyQt5 qdarkstyle`