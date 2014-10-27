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

#define WARPSIZE 16
#define MAXDEPTH 32

__kernel void bound_box(__global float *x_cords,
                        __global float *y_cords,
                        __global float* z_cords,
                        __global float* velx,
                        __global float* vely,
                        __global float* velz,
                        __global float* accx,
                        __global float* accy,
                        __global float* accz,
                        __global int* childl,
                        __global float* massl,
                        __global int* start,
                        __global int* startl,
                        __global float* global_x_mins,
                        __global float* global_x_maxs,
                        __global float* global_y_mins,
                        __global float* global_y_maxs,
                        __global float* global_z_mins,
                        __global float* global_z_maxs,
                        __global int* count,
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

  for(int step = (dim / 2); step > 0; step = step / 2) {
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
      start[k] = 0;


      x_cords[num_nodes] = (minx + maxx) * 0.5f;
      y_cords[num_nodes] = (miny + maxy) * 0.5f;
      //z_cords[num_nodes] = maxy; // (minz + maxz) * 0.5f;
      z_cords[num_nodes] = (minz + maxz) * 0.5f;
      k *= 8;
      for (int i = 0; i < 8; i++) childl[k + i] = -1.0;
      (*stepd)++;
    }
  }
}

__kernel void build_tree(__global volatile float *x_cords,
                        __global float *y_cords,
                        __global float* z_cords,
                        __global float* velx,
                        __global float* vely,
                        __global float* velz,
                        __global float* accx,
                        __global float* accy,
                        __global float* accz,
                        __global volatile int* child,
                        __global float* mass,
                        __global int* start,
                        __global int* sort,
                        __global float* global_x_mins,
                        __global float* global_x_maxs,
                        __global float* global_y_mins,
                        __global float* global_y_maxs,
                        __global float* global_z_mins,
                        __global float* global_z_maxs,
                        __global int* count,
                        __global volatile int* blocked,
                        __global volatile int* step,
                        __global volatile int* bottom,
                        __global volatile int* maxdepth,
                        __global volatile float* radiusd,
                        const int num_bodies,
                        const int num_nodes) {
  float radius = *radiusd;
  float rootx = x_cords[num_nodes];
  float rooty = y_cords[num_nodes];
  float rootz = z_cords[num_nodes];
  float r;
  int localmaxdepth = 1;
  int skip = 1;
  int inc =  get_global_size(0);
  int i = get_global_id(0);
  float x, y, z;
  int j;
  float px, py, pz;
  int ch, n, cell, locked, patch;
  int depth;
  while (i < num_bodies) {
    if (skip != 0) {
      skip = 0;
      px = x_cords[i];
      py = y_cords[i];
      pz = z_cords[i];
      n = num_nodes;
      depth = 1;
      r = radius;
      j = 0;
      if (rootx < px) j = 1;
      if (rooty < py) j += 2;
      if (rootz < pz) j += 4;
    }
    ch = child[n*8 + j];

    while (ch >= num_bodies) {
      n = ch;
      depth++;
      r *= 0.5f;
      j = 0;
      // determine which child to follow
      if (x_cords[n] < px) j = 1;
      if (y_cords[n] < py) j += 2;
      if (z_cords[n] < pz) j += 4;
      ch = child[n*8+j];
    }

    if (ch != -2 ) {
    locked = n*8+j;
    //int test = child[locked];
    //mem_fence(CLK_GLOBAL_MEM_FENCE);
    /*return;*/
    if (ch == atomic_cmpxchg(&child[locked], ch, -2)) {
      //mem_fence(CLK_GLOBAL_MEM_FENCE);

      if(ch == -1) {
        child[locked] = i;
      } else {
        patch = -1;
        // create new cell(s) and insert the old and new body
        int test = 100;
        do {
          depth++;
          cell = atomic_dec(bottom) - 1;

          if (cell <= num_bodies) {
            *bottom = num_nodes;
             return;
          }
          patch = max(patch, cell);

          x = (j & 1) * r;
          y = ((j >> 1) & 1) * r;
          z = ((j >> 2) & 1) * r;
          r *= 0.5f;

          mass[cell] = -1.0f;
          start[cell] = -1;
          x = x_cords[cell] = x_cords[n] - r + x;
          y = y_cords[cell] = y_cords[n] - r + y;
          z = z_cords[cell] = z_cords[n] - r + z;
          for (int k = 0; k < 8; k++) child[cell*8+k] = -1;

          if (patch != cell) {
            child[n*8+j] = cell;
          }

          j = 0;
          if (x < x_cords[ch]) j = 1;
          if (y < y_cords[ch]) j += 2;
          if (z < z_cords[ch]) j += 4;
          child[cell*8+j] = ch;

          n = cell;
          j = 0;
          if (x < px) j = 1;
          if (y < py) j += 2;
          if (z < pz) j += 4;

          ch = child[n*8+j];
          test--;
        } while (test > 0 && ch >= 0);
        child[n*8+j] = i;
        mem_fence(CLK_GLOBAL_MEM_FENCE);
        child[locked] = patch;
       }
        localmaxdepth = max(depth, localmaxdepth);
        i += inc;  // move on to next body
        skip = 1;
      }
      }
  }
  atomic_max(maxdepth, localmaxdepth);
}

__kernel void compute_sums(__global volatile float *x_cords,
                        __global float *y_cords,
                        __global float* z_cords,
                        __global float* velx,
                        __global float* vely,
                        __global float* velz,
                        __global float* accx,
                        __global float* accy,
                        __global float* accz,
                        __global volatile int* children,
                        __global float* mass,
                        __global int* start,
                        __global int* sort,
                        __global float* global_x_mins,
                        __global float* global_x_maxs,
                        __global float* global_y_mins,
                        __global float* global_y_maxs,
                        __global float* global_z_mins,
                        __global float* global_z_maxs,
                        __global int* count,
                        __global volatile int* blocked,
                        __global volatile int* step,
                        __global volatile int* bottom,
                        __global volatile int* maxdepth,
                        __global volatile float* radiusd,
                        const int num_bodies,
                        const int num_nodes) {
  int i, j, k, inc, num_children_missing, cnt, bottom_value, child;
  float m, cm, px, py, pz;
  // TODO change this to THREAD3 Why?
  volatile int missing_children[THREADS1 * 8];
  // TODO chache kernel information

  bottom_value = *bottom;
  inc = get_global_size(0);
  // Align work to WARP SIZE
  k = (bottom_value & (-WARPSIZE)) + get_global_id(0);
  if (k < bottom_value) k += inc;

  num_children_missing = 0;

  while (k <= num_nodes) {
    if (num_children_missing == 0) { // Must be new cell
      cm = 0.0f;
      px = 0.0f;
      py = 0.0f;
      pz = 0.0f;
      cnt = 0;
      j = 0;
      for (i = 0; i < 8; i++) {
        child = children[k*8+i];
        if (child >= 0) {
          if (i != j) {
            // Moving children to front. Apparently needed later
            // TODO figure out why this is
            children[k*8+i] = -1;
            children[k*8+j] = child;
          }
          missing_children[num_children_missing*THREADS1+get_local_id(0)] = child;
          m = mass[child];
          num_children_missing++;
          if (m >= 0.0f) {
            // Child has already been touched
            num_children_missing--;
            if (child >= num_bodies) { // Count the bodies. TODO Why?
              cnt += count[child] - 1;
            }
            // Sum mass and positions
            cm += m;
            px += x_cords[child] * m;
            py += y_cords[child] * m;
            pz += z_cords[child] * m;
          }
          j++;
        }
      }
      cnt += j;
    }

    if (num_children_missing != 0) {
      do {
        child = missing_children[(num_children_missing - 1)*THREADS1+get_local_id(0)];
        m = mass[child];
        if (m >= 0.0f) {
          // Child has been touched
          num_children_missing--;
          if (child >= num_bodies) { // Count the bodies. TODO Why?
            cnt += count[child] - 1;
          }
          // Sum mass and positions
          cm += m;
          px += x_cords[child] * m;
          py += y_cords[child] * m;
          pz += z_cords[child] * m;
        }
      } while ((m >= 0.0f) && (num_children_missing != 0));
        // Repeat until we are done or child is not ready TODO question: is this for thread divergence?
    }

    if (num_children_missing == 0) {
      //We're done! finish the sum
      count[k] = cnt;
      m = 1.0f / cm;
      x_cords[k] = px * m;
      y_cords[k] = py * m;
      z_cords[k] = pz * m;
      mem_fence(CLK_GLOBAL_MEM_FENCE);
      mass[k] = cm;
      k += inc;
    }
  }
}
__kernel void sort(__global volatile float *x_cords,
                        __global float *y_cords,
                        __global float* z_cords,
                        __global float* velx,
                        __global float* vely,
                        __global float* velz,
                        __global float* accx,
                        __global float* accy,
                        __global float* accz,
                        __global volatile int* children,
                        __global float* mass,
                        __global volatile int* start,
                        __global int* sort,
                        __global float* global_x_mins,
                        __global float* global_x_maxs,
                        __global float* global_y_mins,
                        __global float* global_y_maxs,
                        __global float* global_z_mins,
                        __global float* global_z_maxs,
                        __global int* count,
                        __global volatile int* blocked,
                        __global volatile int* step,
                        __global volatile int* bottom,
                        __global volatile int* maxdepth,
                        __global volatile float* radiusd,
                        const int num_bodies,
                        const int num_nodes) {
  int i, k, child, decrement, start_index, bottom_node;
  bottom_node = *bottom;
  decrement = get_global_size(0);
  k = num_nodes + 1 - decrement + get_global_id(0);
  while (k >= bottom_node) {
    start_index = start[k];
    if (start_index >= 0) {
      for (i = 0; i < 8; i++) {
        child = children[k*8+i];
        if (child >= num_bodies) {
          printf("Child1 : %d \n", child);
          start[child] = start_index;
          start_index += count[child];
        } else if (child >= 0) {
          printf("Child : %d Index: %d, \n", child, start_index);
          sort[start_index] = child;
          start_index++;
        }
      }
      k -= decrement;
    }
    mem_fence(CLK_GLOBAL_MEM_FENCE);
   //barrier(CLK_GLOBAL_MEM_FENCE); //TODO how to add throttle?
  }
}
inline int thread_vote(__local int* allBlock, int warpId, int cond)
{
     /*Relies on underlying wavefronts (not whole workgroup)*/
       /*executing in lockstep to not require barrier */
    int old = allBlock[warpId];

    // Increment if true, or leave unchanged
    (void) atomic_add(&allBlock[warpId], cond);

    int ret = (allBlock[warpId] == WARPSIZE);
    printf("allBlock[warp]: %d warp %d \n", allBlock[warpId], warpId);
    allBlock[warpId] = 0;

    printf("Return : %d \n", ret);
    return ret;
}

__kernel void calculate_forces(__global volatile float *x_cords,
                        __global float *y_cords,
                        __global float* z_cords,
                        __global float* velx,
                        __global float* vely,
                        __global float* velz,
                        __global float* accx,
                        __global float* accy,
                        __global float* accz,
                        __global volatile int* children,
                        __global float* mass,
                        __global int* start,
                        __global int* sort,
                        __global float* global_x_mins,
                        __global float* global_x_maxs,
                        __global float* global_y_mins,
                        __global float* global_y_maxs,
                        __global float* global_z_mins,
                        __global float* global_z_maxs,
                        __global int* count,
                        __global volatile int* blocked,
                        __global volatile int* step,
                        __global volatile int* bottom,
                        __global volatile int* maxdepth,
                        __global volatile float* radiusd,
                        const int num_bodies,
                        const int num_nodes) {
  int warp_id, starting_warp_thread_id, shared_mem_offset, difference, depth, child;
  __local volatile int child_index[MAXDEPTH * THREADS1/WARPSIZE], parent_index[MAXDEPTH * THREADS1/WARPSIZE];
 __local volatile int allBlock[THREADS1 / WARPSIZE];
  __local volatile float dq[MAXDEPTH * THREADS1/WARPSIZE];
  __local volatile int shared_step, shared_maxdepth;
  __local volatile int allBlocks[THREADS1/WARPSIZE];
  float px, py, pz, ax, ay, az, dx, dy, dz, temp;
  int idx = get_global_id(0);
  int global_size = get_global_size(0);

  if (idx == 0) {
    int itolsqd = 1.0f / (0.5f*0.5f);
    shared_step = *step;
    shared_maxdepth = *maxdepth;
    temp = *radiusd;
    dq[0] = temp * temp * itolsqd;
    for (int i = 1; i < shared_maxdepth; i++) {
      dq[i] = dq[i - 1] * 0.25f;
    }

    if (shared_maxdepth > MAXDEPTH) {
      temp =  1/0;
    }
    for (int i = 0; i < THREADS1/WARPSIZE; i++) {
      allBlocks[i] = 0;
    }
  }
  barrier(CLK_GLOBAL_MEM_FENCE);

  if (shared_maxdepth <= MAXDEPTH) {
    // Warp and memory ids
    warp_id = idx / WARPSIZE;
    starting_warp_thread_id = warp_id * WARPSIZE;
    shared_mem_offset = warp_id * MAXDEPTH;
    difference = idx - starting_warp_thread_id;
    if (difference < MAXDEPTH) {
      dq[difference + shared_mem_offset] = dq[difference];
    }
  barrier(CLK_GLOBAL_MEM_FENCE);
  for (int k = idx; k < num_bodies; k+=global_size) {
    atomic_add(&allBlock[warp_id], 1);
    int index = sort[k];
    px = x_cords[index];
    py = y_cords[index];
    pz = z_cords[index];
    ax = 0.0f;
    ay = 0.0f;
    az = 0.0f;
    depth = shared_mem_offset;
    if (starting_warp_thread_id == idx) {
      parent_index[shared_mem_offset] = num_nodes;
      child_index[shared_mem_offset] = 0;
    }
    mem_fence(CLK_GLOBAL_MEM_FENCE);
    while (depth >= shared_mem_offset) {
      // Stack has elements
      while(child_index[depth] < 8) {
        child = children[parent_index[depth]*8+child_index[depth]];
        if (idx == starting_warp_thread_id) {
          child_index[depth]++;
        }
        mem_fence(CLK_GLOBAL_MEM_FENCE);
        if (child >= 0) {
          dx = x_cords[child] - px;
          dy = y_cords[child] - py;
          dz = z_cords[child] - pz;
          temp = dx*dx + (dy*dy + (dz*dz + 0.0001f));
          //if ((child <= num_bodies || thread_vote(allBlocks, warp_id, temp >= dq[depth]))) {
          int thread_vote_num = thread_vote(allBlocks, warp_id, temp >= dq[depth]);
          if ((child < num_bodies)  ||  thread_vote_num )  {
            if (thread_vote_num == 1) {
            printf("Test: vote: %d child: %d \n", thread_vote_num, child);
            }
            temp = native_rsqrt(temp);
            temp = mass[child] * temp * temp *temp;
            ax += dx * temp;
            ay += dy * temp;
            az += dz * temp;

          } else {
            depth++;
            if (starting_warp_thread_id == idx) {
              parent_index[depth] = child;
              child_index[depth] = 0;
            }
            mem_fence(CLK_GLOBAL_MEM_FENCE);
          }
        } else {
          depth = max(shared_mem_offset, depth - 1);
        }
      }
      depth--;
    }

    if (shared_step > 0) {
      velx[index] += (ax - accx[index]);
      vely[index] += (ay - accy[index]);
      velz[index] += (az - accz[index]);
    }

    accx[index] = ax;
    accy[index] = ay;
    accz[index] = az;

    }
  }
}

