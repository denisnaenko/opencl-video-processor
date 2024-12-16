import pyopencl as cl
import numpy as np

# Загрузка и компиляция OpenCL программы
with open("kernels.cl", "r") as f:
    kernel_code = f.read()

# Создание контекста и очереди команд
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

# Компиляция программы
program = cl.Program(context, kernel_code).build()

# Исходные данные
data_size = 10
input_data = np.arange(data_size, dtype=np.float32)
output_data = np.empty_like(input_data)

# Создание буферов
mf = cl.mem_flags
input_buffer = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_data)
output_buffer = cl.Buffer(context, mf.WRITE_ONLY, output_data.nbytes)

# Получение всех ядер из программы
kernels = program.all_kernels()
print("Доступные ядра:")
for kernel in kernels:
    print(f"- {kernel.function_name}")

# Запуск всех ядер по очереди
for kernel in kernels:
    print(f"\nЗапуск ядра: {kernel.function_name}")
    kernel(queue, (data_size,), None, input_buffer, output_buffer, np.float32(2.0))
    cl.enqueue_copy(queue, output_data, output_buffer)
    print(f"Результат: {output_data}")
