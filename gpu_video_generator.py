import pyopencl as cl
import numpy as np
import cv2
import os
import argparse


def apply_opencl_filter_rgb(image, kernel_path, kernel_name):
    """Применяет OpenCL-ядро к многоканальному (RGB) изображению."""
    # Инициализация OpenCL
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    # Загрузка OpenCL-ядра из файла
    with open(kernel_path, 'r') as f:
        kernel_code = f.read()
    program = cl.Program(context, kernel_code).build()
    mf = cl.mem_flags

    # Подготовка данных
    height, width, channels = image.shape
    image_flat = image.reshape((-1, 3))  # Плоский массив, сохраняющий RGB
    output_image = np.empty_like(image_flat)

    # Разделяем каналы
    channel_r = image_flat[:, 0].astype(np.uint8)
    channel_g = image_flat[:, 1].astype(np.uint8)
    channel_b = image_flat[:, 2].astype(np.uint8)

    # Создаём буферы для каждого канала
    input_r = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=channel_r)
    input_g = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=channel_g)
    input_b = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=channel_b)
    output_r = cl.Buffer(context, mf.WRITE_ONLY, channel_r.nbytes)
    output_g = cl.Buffer(context, mf.WRITE_ONLY, channel_g.nbytes)
    output_b = cl.Buffer(context, mf.WRITE_ONLY, channel_b.nbytes)

    # Получаем ядро
    kernel = getattr(program, kernel_name)

    # Запускаем ядро отдельно для каждого канала
    global_size = (channel_r.size,)
    kernel(queue, global_size, None, input_r, output_r)
    kernel(queue, global_size, None, input_g, output_g)
    kernel(queue, global_size, None, input_b, output_b)

    # Копируем результаты обратно в хост
    result_r = np.empty_like(channel_r)
    result_g = np.empty_like(channel_g)
    result_b = np.empty_like(channel_b)

    cl.enqueue_copy(queue, result_r, output_r)
    cl.enqueue_copy(queue, result_g, output_g)
    cl.enqueue_copy(queue, result_b, output_b)

    # Собираем каналы обратно в изображение
    output_image[:, 0] = result_r
    output_image[:, 1] = result_g
    output_image[:, 2] = result_b

    return output_image.reshape((height, width, channels))

def generate_video(input_folder, output_video, kernel_path, kernel_name, frame_size=(1280, 720), fps=30):
    images = sorted([f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))])
    if not images:
        print("Не найдены изображения в указанной папке.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

    for img_file in images:
        image = cv2.imread(os.path.join(input_folder, img_file))
        image_resized = cv2.resize(image, frame_size)

        # Применяем OpenCL-фильтр
        filtered = apply_opencl_filter_rgb(image_resized, kernel_path, kernel_name)

        # Если grayscale, обрабатываем как одноканальное изображение
        if kernel_name == "grayscale":
            filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)

        video_writer.write(filtered)

    video_writer.release()
    print(f"Видео сохранено: {output_video}")

if __name__ == "__main__":
    # Создание CLI-парсера
    parser = argparse.ArgumentParser()

    # Добавление аргументов
    parser.add_argument("--input", required=True, help="Путь к папке с изображениями.")
    parser.add_argument("--output", default="output/video.mp4", help="Имя выходного видеофайла.")
    parser.add_argument("--effect", choices=["blur", "sharpen", "grayscale"], required=True, help="Тип эффекта.")

    # Парсинг аргументов
    args = parser.parse_args()

    # Сопоставление эффектов и файлов ядер
    kernel_map = {
        "blur": "kernels/blur.cl",
        "sharpen": "kernels/sharpen.cl",
        "grayscale": "kernels/grayscale.cl"
    }

    # Вызов функции генерации видео
    generate_video(args.input, args.output, kernel_map[args.effect], args.effect)
