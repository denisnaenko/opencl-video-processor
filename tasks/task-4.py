import pyopencl as cl

platform = cl.get_platforms()[0]
device = platform.get_devices()[0]

# Получаем список расширений устройства
extensions = device.get_info(cl.device_info.EXTENSIONS)
print("\nПоддерживаемые расширения устройства:\n")

extensions = str.split(extensions, sep=' ')
extensions.pop()
for i, extension in enumerate(extensions):
    print(f'{i+1}. {extension}')

# Проверка поддержки конкретного расширения
if "cl_khr_fp64" in extensions:
    print("\nУстройство поддерживает double precision (cl_khr_fp64).\n")
else:
    print("\nУстройство НЕ поддерживает double precision (cl_khr_fp64).\n")
