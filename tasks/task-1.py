import pyopencl as cl

# Получаем все платформы
platforms = cl.get_platforms()
print("\nДоступные платформы и устройства:")

# Перечисляем платформы и устройства
for i, platform in enumerate(platforms):
    print(f"\nПлатформа {i}: {platform.name}")
    devices = platform.get_devices()
    for j, device in enumerate(devices):
        print(f"Устройство {j}: {device.name}")

# Выбор платформы и устройства
platform_index = int(input("\nВыберите платформу (индекс): "))
device_index = int(input("Выберите устройство (индекс): "))

platform = platforms[platform_index]
device = platform.get_devices()[device_index]

# Инициализация OpenCL
context = cl.Context([device])
queue = cl.CommandQueue(context)
print(f"\nВыбрано устройство: {device.name}\n")
