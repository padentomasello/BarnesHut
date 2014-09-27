__kernel void bound_box(__global int *x_cords, __global int *y_cords, __global int* xy_max,
    __local int* sminx, __local int* smaxx, __local int* sminy, __local int* smaxy, int n)
{
  size_t tid = get_local_id(0);
  size_t gid = get_group_id(0);
  size_t dim = get_local_size(0);
  size_t idx = get_global_id(0);
  if (idx < (size_t)n) {
    sminx[tid] = x_cords[idx];
    smaxx[tid] = x_cords[idx];
    sminy[tid] = y_cords[idx];
    smaxy[tid] = y_cords[idx];
  } else {
    sminx[tid] = 0;
    smaxx[tid] = 0;
    sminy[tid] = 0;
    smaxy[tid] = 0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for(int step = dim / 2; step > 0; step = step / 2) {
  	if (tid < step) {
		sminx[tid] = min(sminx[tid] , sminx[tid + step]);
    smaxx[tid] = max(smaxx[tid], smaxx[tid + step]);
		sminy[tid] = min(sminy[tid] , sminy[tid + step]);
    smaxy[tid] = max(smaxy[tid], smaxy[tid + step]);
	}
	barrier(CLK_LOCAL_MEM_FENCE);
  }
  //Only one thread needs to outdate the global buffer 
  if (tid == 0) {
  	  xy_max[0] = sminx[0];
      xy_max[1] = smaxx[0];
  	  xy_max[2] = sminy[0];
      xy_max[3] = smaxy[0];
  }
}
