


__kernel build_tree(__global float *x_cords,
                        __global float *y_cords,
                        __global float* z_cords,
                        __global float* childl,
                        __global float* massl,
                        __global float* startl,
                        __global float* global_x_mins,
                        __global float* global_x_maxs,
                        __global float* global_y_mins,
                        __global float* global_y_maxs,
                        __global float* global_z_mins,
                        __global float* global_z_maxs,
                        __global volatile int* blocked,
                        __global volatile int* stepd,
                        __global volatile int* bottomd,
                        __global volatile int* maxdepthd,
                        __global volatile float* radius,
                        const int num_bodies,
                        const int num_nodes)
