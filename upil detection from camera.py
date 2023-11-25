import os
import cv2
import qrcode
from PIL import Image
#from ftplib import FTP
import uuid
import numpy as np

local_dir = '/home/pi/ph/photo'
template_path = '/home/pi/ph/sheme/1.png'

def capture_image(cap):
    ret, frame = cap.read()
    
    # Преобразование изображения в черно-белое
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Применение размытия для уменьшения шума перед обнаружением кругов
    gray = cv2.medianBlur(gray, 5)
    
    # Применение преобразования Хафа для обнаружения кругов
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
    
    # Проверка, зрачки обнаружены
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Создание нового черного изображения того же размера, что и оригинал
        black_img = np.zeros(frame.shape, dtype=np.uint8)
        for i in circles[0, :]:
            # Рисование круга (зрачка) на черном изображении
            cv2.circle(black_img, (i[0], i[1]), i[2], (255, 255, 255), -1)
        frame = black_img
    else:
        print("No pupil were found in the image")
    
    file_name = str(uuid.uuid4()) + '.jpg'
    cv2.imwrite(os.path.join(local_dir, file_name), frame)
    return os.path.join(local_dir, file_name)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    cv2.imshow('Press 1 to capture, q to quit', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('1'):  # capture image
        captured_image_path = capture_image(cap)
        file_name = str(uuid.uuid4()) + '.jpg'
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        data = "https://#" + file_name
        qr.add_data(data)
        qr.make(fit=True)

        img_qr = qr.make_image(fill='black', back_color='white').resize((150, 150))
        captured_img = Image.open(captured_image_path).resize((640, 480))

        template_img = Image.open(template_path)
        template_img.paste(captured_img, ((1280-640)//2, (720-480)//2))  
        template_img.paste(img_qr, (1280-150, 720-150)) 

        final_file_path = os.path.join(local_dir, 'final_' + file_name)
        template_img = template_img.convert('RGB')
        template_img.save(final_file_path)

        # Удаление исходного файла после отправки
        os.remove(captured_image_path)

    elif key == ord('q'):  # quit
        break

cap.release()
cv2.destroyAllWindows()
