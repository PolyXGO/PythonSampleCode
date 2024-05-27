# Học nhanh Python với các lưu ý chủ yếu và cơ bản nhất

import os

# ==========================
# Cú pháp và cấu trúc cơ bản
# ==========================
# Python sử dụng thụt lề để xác định các khối mã

if True:
    print("Hello, World!")  # Thụt lề để xác định khối mã

# Kết quả
# Hello, World!

# ==================================

# ==========================
# Biến và kiểu dữ liệu
# ==========================
# Khai báo và sử dụng các biến với các kiểu dữ liệu cơ bản

x = 10       # int
y = 3.14     # float
name = "PolyXGO"  # str
is_active = True  # bool

print(x)
print(y)
print(name)
print(is_active)

# Kết quả
# 10
# 3.14
# PolyXGO
# True

# ==================================

# ==========================
# Các cấu trúc điều khiển
# ==========================
# Sử dụng if-else để thực hiện các điều kiện

if x > 5:
    print("x lớn hơn 5")
else:
    print("x nhỏ hơn hoặc bằng 5")

# Kết quả
# x lớn hơn 5

# Vòng lặp for
for i in range(5):
    print(i)

# Kết quả
# 0
# 1
# 2
# 3
# 4

# Vòng lặp while
count = 0
while count < 5:
    print(count)
    count += 1

# Kết quả
# 0
# 1
# 2
# 3
# 4

# ==================================

# ==========================
# Hàm và module
# ==========================
# Định nghĩa và sử dụng hàm

def greet(name):
    return f"Hello, {name}!"

print(greet("PolyXGO"))

# Kết quả
# Hello, PolyXGO!

# Import module để sử dụng các hàm và lớp được định nghĩa sẵn
import math
print(math.sqrt(16))  # Kết quả là 4.0

# Kết quả
# 4.0

# ==================================

# ==========================
# Danh sách (List) và tuple
# ==========================
# Sử dụng danh sách và tuple

numbers = [1, 2, 3, 4]
numbers.append(5)  # Thêm phần tử vào danh sách
print(numbers)

# Kết quả
# [1, 2, 3, 4, 5]

coordinates = (10.0, 20.0)
print(coordinates)

# Kết quả
# (10.0, 20.0)

# ==================================

# ==========================
# Dictionary và Set
# ==========================
# Sử dụng dictionary và set

person = {
    "name": "PolyXGO",
    "age": 30
}
print(person["name"])  # Truy cập giá trị qua key

# Kết quả
# PolyXGO

unique_numbers = {1, 2, 3, 4}
unique_numbers.add(5)
print(unique_numbers)

# Kết quả
# {1, 2, 3, 4, 5}

# ==================================

# ==========================
# List comprehensions
# ==========================
# Tạo danh sách bằng cách ngắn gọn

squares = [x**2 for x in range(10)]
print(squares)

# Kết quả
# [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# ==================================

# ==========================
# Xử lý ngoại lệ
# ==========================
# Sử dụng try-except để bắt và xử lý ngoại lệ

try:
    result = 10 / 0
except ZeroDivisionError:
    print("Lỗi: Chia cho 0!")

# Kết quả
# Lỗi: Chia cho 0!

# ==================================

# ==========================
# File I/O
# ==========================
# Đọc và ghi file

# Đường dẫn tới tập tin
file_path = "temporary/test_example.txt"

# Kiểm tra và tạo thư mục nếu chưa tồn tại
directory = os.path.dirname(file_path)
if not os.path.exists(directory):
    os.makedirs(directory)

# Kiểm tra và tạo tập tin nếu chưa tồn tại
if not os.path.exists(file_path):
    with open(file_path, "w") as file:
        file.write("")  # Tạo tập tin trống

# Đọc file
with open(file_path, "r") as file:
    content = file.read()
    print(content)

# Kết quả
# (Nội dung ban đầu, nếu có)

# Ghi file
with open(file_path, "w") as file:
    file.write("Hello, World!")

# Đọc lại file để kiểm tra nội dung vừa ghi
with open(file_path, "r") as file:
    content = file.read()
    print(content)

# Kết quả
# Hello, World!

# ==================================

# ==========================
# Lập trình hướng đối tượng (OOP)
# ==========================
# Định nghĩa lớp và tạo đối tượng

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def greet(self):
        return f"Hello, my name is {self.name} and I am {self.age} years old."

person = Person("PolyXGO", 4)
print(person.greet())

# Kết quả
# Hello, my name is PolyXGO and I am 4 years old.

# ==================================

# ==========================
# Đa hình trong OOP
# ==========================
# Đa hình cho phép các lớp con thực hiện các phương thức theo cách khác nhau

class Animal:
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"

animals = [Dog(), Cat()]

for animal in animals:
    print(animal.speak())

# Kết quả
# Woof!
# Meow!

# ==================================
