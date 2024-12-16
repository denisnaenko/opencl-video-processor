import cv2
import os

# Путь к входному видео и папка для кадров
video_path = "input_video/input_video.mp4"
output_folder = "input_images"
os.makedirs(output_folder, exist_ok=True)

# Чтение видео
cap = cv2.VideoCapture(video_path)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Сохранение кадра
    frame_name = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(frame_name, frame)
    print(f"Сохранён кадр: {frame_name}")
    frame_count += 1

cap.release()
print(f"Разделение видео завершено. Кадры сохранены в {output_folder}.")
