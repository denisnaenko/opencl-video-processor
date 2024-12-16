import pyopencl as cl

platform = cl.get_platforms()[0]
device = platform.get_devices()[0]

# Получаем информацию о памяти устройства
device_memory = device.get_info(cl.device_info.GLOBAL_MEM_SIZE)
max_buffer_size = device.get_info(cl.device_info.MAX_MEM_ALLOC_SIZE)

print(f"Доступная глобальная память устройства: {device_memory / 1e6:.2f} MB")
print(f"Максимальный размер буфера: {max_buffer_size / 1e6:.2f} MB")
