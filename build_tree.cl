


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
                        __global volatile float* radiusd,
                        const int num_bodies,
                        const int num_nodes) {
  int radius = *radiusd;
  int rootx = x_cords[num_nodes];
  int rooty = y_cords[num_nodes];
  int rootz = z_cords[num_nodes];
  int localmaxdepth = 1;
  int skip = 1;
  int inc =  get_global_size(0);
  int i = get_globaal_id(0);
  int j, r;
  int x, y, z;
  int ch, n, cell, locked, patch;
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

    ch = childd[n*8 + j]
    while (ch >= num_bodies) {
      n = ch;
      depth++;
      r *= 0.5f;
      j = 0; 
      if (rootx < px) j = 1;
      if (rooty < py) j += 2;
      if (rootz < pz) j += 4;
    }
    if (ch != -2) {
      locked = n*8+j;
      if (ch == atomic_cmpxchg(&childd[locked], ch, -2)) {
        if(ch == -1) {
          // if null, just insert the new body
          child[locked] = i;
         } else {
          patch = -1;
          // create new cell(s) and insert the old and new body
          do {
            depth++;
            cell = atomic_dec(&bottomd) - 1;
            if (cell <= num_bodies) {
              depth = 1/0;
              bottomd = num_nodes;
            }
            patch = max(patch, cell);
            
            x = (j & 1) * r;
            y = ((j >> 1) & 1) * r;
            z = ((j >> 2) & 1) * r;
            r *= 0.5f;

            massd[cell] = -1.0f;
            startd[cell] = -1;
            x = x_cords[cell] = x_cords[n] - r + x;
            y = y_cords[cell] = y_cords[n] - r + y;
            z = z_cords[cell] = z_cords[n] - r + z;
            for (k = 0; k < 8; k++) childd[cell*8+k] = -1;

            if (patch != cell) { 
              childd[n*8+j] = cell;
            }

            j = 0;
            if (x < posxd[ch]) j = 1;
            if (y < posyd[ch]) j += 2;
            if (z < poszd[ch]) j += 4;
            childd[cell*8+j] = ch;

            n = cell;
            j = 0;
            if (x < px) j = 1;
            if (y < py) j += 2;
            if (z < pz) j += 4;

            ch = childd[n*8+j];
          } while (ch >= 0);
            childd[n*8+j] = i;
            __threadfence();  // push out subtree
            childd[locked] = patch;
        }

        localmaxdepth = max(depth, localmaxdepth);
        i += inc;  // move on to next body
        skip = 1;
      }
    }
    __syncthreads();  // throttle
  }
  // record maximum tree depth
  atomicMax((int *)&maxdepthd, localmaxdepth);
}
        

      


}


