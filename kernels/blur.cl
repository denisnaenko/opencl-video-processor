__kernel void blur(__global const uchar* input, __global uchar* output) {
    int idx = get_global_id(0);
    output[idx] = input[idx] / 4;  // Уменьшение интенсивности как простой пример размытия
}
