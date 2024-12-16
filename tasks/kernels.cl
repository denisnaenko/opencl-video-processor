__kernel void add_constant(__global const float* input, __global float* output, float constant_value) {
    int idx = get_global_id(0);
    output[idx] = input[idx] + constant_value;
}

__kernel void multiply_constant(__global const float* input, __global float* output, float constant_value) {
    int idx = get_global_id(0);
    output[idx] = input[idx] * constant_value;
}
