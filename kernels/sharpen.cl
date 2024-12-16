__kernel void sharpen(__global const uchar* input, __global uchar* output) {
    int idx = get_global_id(0);
    output[idx] = min(input[idx] * 2, 255);  // Усиление интенсивности для резкости
}
