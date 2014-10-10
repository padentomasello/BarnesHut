// TODO include file
#define THREADS1 256  /* must be a power of 2 */
#define THREADS2 1024
#define THREADS3 1024
#define THREADS4 256
#define THREADS5 256
#define THREADS6 512

// block count = factor * #SMs
#define FACTOR1 3
#define FACTOR2 1
#define FACTOR3 1  /* must all be resident at the same time */
#define FACTOR4 1  /* must all be resident at the same time */
#define FACTOR5 5
#define FACTOR6 3

#define WARPSIZE 32
#define MAXDEPTH 32

__kernel void bound_box(__global float *x_cords,
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
                        __global volatile float* radiusd,
                        const int num_bodies,
                        const int num_nodes)
{
  size_t tid = get_local_id(0);
  size_t gid = get_group_id(0);
  size_t dim = get_local_size(0);
  size_t global_dim_size = get_global_size(0);
  size_t idx = get_global_id(0);
  float minx, maxx, miny, maxy, minz, maxz;
  __local float sminx[THREADS1], smaxx[THREADS1], sminy[THREADS1], smaxy[THREADS1], sminz[THREADS1], smaxz[THREADS1];
  minx = maxx = x_cords[0];
  miny = maxy = y_cords[0];
  minz = maxz = z_cords[0];
  float val;
  int inc = global_dim_size;
  for (int j = idx; j < num_bodies; j += inc) {
    val = x_cords[j];
    minx = min(val, minx);
    maxx = max(val, maxx);
    val = y_cords[j];
    miny = min(val, miny);
    maxy = max(val, maxy);
    val = z_cords[j];
    minz = min(val, minz);
    maxz = max(val, maxz);
  }
  sminx[tid] = minx;
  smaxx[tid] = maxx;
  sminy[tid] = miny;
  smaxy[tid] = maxy;
  sminz[tid] = minz;
  smaxz[tid] = maxz;
  barrier(CLK_LOCAL_MEM_FENCE);

  for(int step = dim / 2; step > 0; step = step / 2) {
  	if (tid < step) {
		sminx[tid] = minx = min(sminx[tid] , sminx[tid + step]);
    smaxx[tid] = maxx = max(smaxx[tid], smaxx[tid + step]);
		sminy[tid] = miny = min(sminy[tid] , sminy[tid + step]);
    smaxy[tid] = maxy = max(smaxy[tid], smaxy[tid + step]);
		sminz[tid] = minz = min(sminz[tid] , sminz[tid + step]);
    smaxz[tid] = maxz = max(smaxz[tid], smaxz[tid + step]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Only one thread needs to outdate the global buffer
  inc = (global_dim_size / dim) - 1;
  if (tid == 0) {
    global_x_mins[gid] = minx;
    global_x_maxs[gid] = maxx;
    global_y_mins[gid] = miny;
    global_y_maxs[gid] = maxy;
    global_z_mins[gid] = minz;
    global_z_maxs[gid] = maxz;
    inc = (global_dim_size / dim) - 1;
    if (inc == atomic_inc(blocked)) {
      for(int j = 0; j <= inc; j++) {
        minx = min(minx, global_x_mins[j]);
        maxx = max(maxx, global_x_maxs[j]);
        miny = min(miny, global_y_mins[j]);
        maxy = max(maxy, global_y_maxs[j]);
        minz = min(minz, global_z_mins[j]);
        maxz = max(maxz, global_z_maxs[j]);
      }

      // Compute the radius
      val = max(maxx - minx, maxy - miny);
      *radiusd = (float) (max(val, maxz - minz) * 0.5f);

      int k = num_nodes;
      *bottomd = k;
      // TODO bottomd;

      massl[k] = -1.0f;
      startl[k] = 0;

      x_cords[num_nodes] = (minx + maxx) * 0.5f;
      y_cords[num_nodes] = (miny + maxy) * 0.5f;
      z_cords[num_nodes] = (minz + maxz) * 0.5f;
      k *= 8;
      for (int i = 0; i < 8; i++) childl[k + i] = -1.0;
      (*stepd)++;
    }
  }
}
