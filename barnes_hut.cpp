#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cmath>
#include <iostream>
#include <unistd.h>

#include "clhelp.h"

using namespace std;

// TODO this used to be 512.
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

struct KernelArgs{
  cl_mem posx, posy, posz, child, mass, start, minx, maxx, miny, maxy, minz, maxz, blocked, step, bottom, max_depth, radius, count;
  cl_mem velx, vely, velz, accx, accy, accz, sort;
  int num_bodies, num_nodes;
  int num_args;
};

struct HostMemory {
  float *mass, *posx, *posy, *posz, *velx, *vely, *velz;
  int* start, *child, *count, *sort;
  int step, max_depth, bottom, blocked;
};

void CreateMemBuffer (cl_vars_t* cv, KernelArgs* args, HostMemory* host_memory) {
  cl_int err;
  int num_nodes = args->num_nodes;
  int num_bodies = args->num_bodies;
  // TODO This shouldn't be hardcodes
  int num_work_groups = 32;
  args->minx = clCreateBuffer(cv->context, CL_MEM_READ_WRITE,
      sizeof(float) * num_work_groups, NULL, &err);
  args->maxx = clCreateBuffer(cv->context, CL_MEM_READ_WRITE,
      sizeof(float) * num_work_groups, NULL, &err);
  args->miny = clCreateBuffer(cv->context, CL_MEM_READ_WRITE,
      sizeof(float) * num_work_groups, NULL, &err);
  args->maxy = clCreateBuffer(cv->context, CL_MEM_READ_WRITE,
      sizeof(float) * num_work_groups, NULL, &err);
  args->minz = clCreateBuffer(cv->context, CL_MEM_READ_WRITE,
      sizeof(float) * num_work_groups, NULL, &err);
  args->maxz = clCreateBuffer(cv->context, CL_MEM_READ_WRITE,
      sizeof(float) * num_work_groups, NULL, &err);
  args->step = clCreateBuffer(cv->context, CL_MEM_READ_WRITE,
      sizeof(int) * 1, NULL, &err);
  args->bottom = clCreateBuffer(cv->context, CL_MEM_READ_WRITE,
      sizeof(int) * 1, NULL, &err);
  args->max_depth = clCreateBuffer(cv->context, CL_MEM_READ_WRITE,
      sizeof(int) * 1, NULL, &err);
  args->radius = clCreateBuffer(cv->context, CL_MEM_READ_WRITE,
      sizeof(float) * 1, NULL, &err);

  // Create Buffers  NOTE* These do need to be (num_nodes + 1)
  args->posx = clCreateBuffer(cv->context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE,
      sizeof(float) * (num_nodes + 1), host_memory->posx, &err);

  CHK_ERR(err);
  args->posz = clCreateBuffer(cv->context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE,
      sizeof(float)*(num_nodes + 1), host_memory->posz, &err);
  args->posy = clCreateBuffer(cv->context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE,
      sizeof(float)*(num_nodes + 1), host_memory->posy, &err);
  CHK_ERR(err);
  args->mass = clCreateBuffer(cv->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      sizeof(float)*(num_nodes + 1), host_memory->mass, &err);
  CHK_ERR(err);
  args->count = clCreateBuffer(cv->context, CL_MEM_READ_WRITE,
      sizeof(int)*(num_nodes + 1), NULL, &err);
  CHK_ERR(err);
  args->start = clCreateBuffer(cv->context, CL_MEM_READ_WRITE,
      sizeof(int)*(num_nodes + 1), NULL, &err);
  CHK_ERR(err);
  args->child = clCreateBuffer(cv->context, CL_MEM_READ_WRITE,
      sizeof(int)*8*(num_nodes + 1), NULL, &err);

  //Set the following alligned on WARP boundaries
  int inc = (num_bodies + WARPSIZE -1) & (-WARPSIZE);
    cl_buffer_region velxl_region = {0, 1*inc};
    cl_buffer_region velyl_region = {1*inc, 2*inc};
    cl_buffer_region velzl_region = {2*inc, 3*inc};
    cl_buffer_region accxl_region = {3*inc, 4*inc};
    cl_buffer_region accyl_region = {4*inc, 5*inc};
    cl_buffer_region acczl_region = {5*inc, 6*inc};
    cl_buffer_region sortl_region = {6*inc, 7*inc};
    args->velx = clCreateSubBuffer(args->child, CL_MEM_READ_WRITE,
        CL_BUFFER_CREATE_TYPE_REGION, &velxl_region, &err);
    args->vely = clCreateSubBuffer(args->child, CL_MEM_READ_WRITE,
        CL_BUFFER_CREATE_TYPE_REGION, &velyl_region, &err);
    args->velz = clCreateSubBuffer(args->child, CL_MEM_READ_WRITE,
        CL_BUFFER_CREATE_TYPE_REGION, &velzl_region, &err);
    args->accx = clCreateSubBuffer(args->child, CL_MEM_READ_WRITE,
        CL_BUFFER_CREATE_TYPE_REGION, &accxl_region, &err);
    args->accy = clCreateSubBuffer(args->child, CL_MEM_READ_WRITE,
        CL_BUFFER_CREATE_TYPE_REGION, &accyl_region, &err);
    args->accz = clCreateSubBuffer(args->child, CL_MEM_READ_WRITE,
        CL_BUFFER_CREATE_TYPE_REGION, &acczl_region, &err);
    args->sort = clCreateSubBuffer(args->child, CL_MEM_READ_WRITE,
        CL_BUFFER_CREATE_TYPE_REGION, &sortl_region, &err);

  // Global scalars //TODO is there a better way to do this?
  // TODO Would it be more efficient to use an InitializationKernel? See Cuda implementation around line 82
  args->blocked = clCreateBuffer(cv->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      sizeof(int), &host_memory->blocked, &err);
  args->step = clCreateBuffer(cv->context, CL_MEM_READ_WRITE |CL_MEM_COPY_HOST_PTR,
      sizeof(int), &host_memory->step, &err);
  args->max_depth = clCreateBuffer(cv->context, CL_MEM_READ_WRITE |CL_MEM_COPY_HOST_PTR,
      sizeof(int), &host_memory->max_depth, &err);

}


void SetArgs(cl_kernel *kernel, KernelArgs* args){
  // Allocate device memory // TODO Can change all host arrays to size num_cells. Using num_boides for debugging purpuses.
  cl_int err;
  err = clSetKernelArg(*kernel, 0, sizeof(cl_mem), &args->posx);
  CHK_ERR(err);
  err = clSetKernelArg(*kernel, 1, sizeof(cl_mem), &args->posy);
  CHK_ERR(err);
  err = clSetKernelArg(*kernel, 2, sizeof(cl_mem), &args->posz);
  CHK_ERR(err);
  err = clSetKernelArg(*kernel, 3, sizeof(cl_mem), &args->velx);
  CHK_ERR(err);
  err = clSetKernelArg(*kernel, 4, sizeof(cl_mem), &args->vely);
  CHK_ERR(err);
  err = clSetKernelArg(*kernel, 5, sizeof(cl_mem), &args->velz);
  CHK_ERR(err);
  err = clSetKernelArg(*kernel, 6, sizeof(cl_mem), &args->accx);
  CHK_ERR(err);
  err = clSetKernelArg(*kernel, 7, sizeof(cl_mem), &args->accy);
  CHK_ERR(err);
  err = clSetKernelArg(*kernel, 8, sizeof(cl_mem), &args->accz);
  CHK_ERR(err);
  err = clSetKernelArg(*kernel, 9, sizeof(cl_mem), &args->child);
  CHK_ERR(err);
  err = clSetKernelArg(*kernel, 10, sizeof(cl_mem), &args->mass);
  CHK_ERR(err);
  err = clSetKernelArg(*kernel, 11, sizeof(cl_mem), &args->start);
  CHK_ERR(err);
  err = clSetKernelArg(*kernel, 12, sizeof(cl_mem), &args->count);
  CHK_ERR(err);
  err = clSetKernelArg(*kernel, 13, sizeof(cl_mem), &args->minx);
  CHK_ERR(err);
  err = clSetKernelArg(*kernel, 14, sizeof(cl_mem), &args->maxx);
  CHK_ERR(err);
  err = clSetKernelArg(*kernel, 15, sizeof(cl_mem), &args->miny);
  CHK_ERR(err);
  err = clSetKernelArg(*kernel, 16, sizeof(cl_mem), &args->maxy);
  CHK_ERR(err);
  err = clSetKernelArg(*kernel, 17, sizeof(cl_mem), &args->minz);
  CHK_ERR(err);
  err = clSetKernelArg(*kernel, 18, sizeof(cl_mem), &args->maxz);
  CHK_ERR(err);
  err = clSetKernelArg(*kernel, 19, sizeof(cl_mem), &args->count);
  CHK_ERR(err);
  err = clSetKernelArg(*kernel, 20, sizeof(cl_mem), &args->blocked);
  CHK_ERR(err);
  err = clSetKernelArg(*kernel, 21, sizeof(cl_mem), &args->step);
  CHK_ERR(err);
  err = clSetKernelArg(*kernel, 22, sizeof(cl_mem), &args->bottom);
  CHK_ERR(err);
  err = clSetKernelArg(*kernel, 23, sizeof(cl_mem), &args->max_depth);
  CHK_ERR(err);
  err = clSetKernelArg(*kernel, 24, sizeof(cl_mem), &args->radius);
  CHK_ERR(err);
  err = clSetKernelArg(*kernel, 25, sizeof(int), &args->num_bodies);
  CHK_ERR(err);
  err = clSetKernelArg(*kernel, 26, sizeof(int), &args->num_nodes);
  CHK_ERR(err);

}

void AllocateHostMemory(HostMemory* host, int num_nodes, int num_bodies) {
  host->step = -1;
  host->max_depth = -1;
  host->blocked = 0;
  host->mass = (float *)malloc(sizeof(float) * (num_nodes + 1));
  if (host->mass == NULL) {fprintf(stderr, "cannot allocate mass\n");  exit(-1);}
  host->start = (int *)malloc(sizeof(float) * (num_nodes + 1));
  if (host->start == NULL) {fprintf(stderr, "cannot allocate mass\n");  exit(-1);}
  host->posx = (float *)malloc(sizeof(float) * (num_nodes + 1)); // TODO Can change to number of bodies
  if (host->posx == NULL) {fprintf(stderr, "cannot allocate posx\n");  exit(-1);}
  host->posy = (float *)malloc(sizeof(float) * (num_nodes + 1)); // TODO Can change to number of bodies
  if (host->posy == NULL) {fprintf(stderr, "cannot allocate posy\n");  exit(-1);}
  host->posz = (float *)malloc(sizeof(float) * (num_nodes + 1));
  if (host->posz == NULL) {fprintf(stderr, "cannot allocate posz\n");  exit(-1);}
  host->velx = (float *)malloc(sizeof(float) * num_bodies);
  if (host->velx == NULL) {fprintf(stderr, "cannot allocate velx\n");  exit(-1);}
  host->vely = (float *)malloc(sizeof(float) * num_bodies);
  if (host->vely == NULL) {fprintf(stderr, "cannot allocate vely\n");  exit(-1);}
  host->velz = (float *)malloc(sizeof(float) * num_bodies);
  if (host->velz == NULL) {fprintf(stderr, "cannot allocate velz\n");  exit(-1);}
  host->child = (int *)malloc(sizeof(int) * 8*(num_nodes + 1));
  if (host->child == NULL) {fprintf(stderr, "cannot allocate velz\n");  exit(-1);}
  // TODO This can be removed after debugging. Count is only used on GPU
  host->count = (int *)malloc(sizeof(int) *(num_nodes + 1));
  if (host->count == NULL) {fprintf(stderr, "cannot allocate velz\n");  exit(-1);}
  host->sort = (int *)malloc(sizeof(int) *(num_nodes + 1));
  if (host->count == NULL) {fprintf(stderr, "cannot allocate velz\n");  exit(-1);}
}

void CalculateSummation(cl_vars_t* cv, KernelArgs* args, HostMemory* host_memory) {
  int bottom = host_memory->bottom;
  int num_nodes = args->num_nodes;
  int num_bodies = args->num_bodies;
  int child, cnt, j;

  for (int parent = bottom; parent <= num_nodes; parent++) {
    float px = 0.0f;
    float py = 0.0f;
    float pz = 0.0f;
    int cnt = 0;
    float m = 0.0;
    int j = 0;
    float cm = 0.0;
    for (int i = 0; i < 8; i++) {
      child = host_memory->child[8*parent+i];
      if (child >= 0) {
        if (i != j) {
          // Moving children to front. Apparently needed later
          // TODO figure out why this is
          host_memory->child[parent*8+i] = -1;
          host_memory->child[parent*8+j] = child;
        }
        m = host_memory->mass[child];
        if (child >= num_bodies) { // Count the bodies. TODO Why?
          cnt += host_memory->count[child] - 1;
        }
        // Sum mass and positions
        cm += m;
        px += host_memory->posx[child] * m;
        py += host_memory->posy[child] * m;
        pz += host_memory->posz[child] * m;
        j++;
      }
    }
    cnt += j;
    host_memory->count[parent] = cnt;
    m = 1.0f / cm;
    host_memory->posx[parent] = px * m;
    host_memory->posy[parent] = py * m;
    host_memory->posz[parent] = pz * m;
    host_memory->mass[parent] = cm;
  }
}


// TODO maybe count the nodes in order to make sure this reaches all of the.
void CheckTree(int index, HostMemory *host_memory,int  num_bodies) {
  if (index < num_bodies) return;
  float parent_x_position = host_memory->posx[index];
  float parent_y_position = host_memory->posy[index];
  float parent_z_position = host_memory->posz[index];
  float child_x_position, child_y_position, child_z_position;
  int j, child_index;
  for (int i = 0; i < 8; i++) {
    child_index = host_memory->child[8*index + i];
    if (child_index >= 0) {
    child_x_position = host_memory->posx[child_index];
    child_y_position = host_memory->posy[child_index];
    child_z_position = host_memory->posz[child_index];
    j = 0;
    if (child_x_position > parent_x_position) j+=1;
    if (child_y_position > parent_y_position) j+=2;
    if (child_z_position > parent_z_position) j+=4;
    if (j != i) {
      std::cout << "ERROR" << std::endl;
    }
    CheckTree(child_index, host_memory, num_bodies);
    }
  }
}

void CheckSummation(HostMemory* gpu_host, HostMemory* cpu_host, int num_nodes) {
  const float epsilon = 0.000001;
  for(int i = cpu_host->bottom; i <= num_nodes; i++) {
    if (gpu_host->posx[i] - cpu_host->posx[i] > (epsilon*cpu_host->posx[i])) {
      cout << "Summation x ERROR at i: " << i << " cout gpu: " << gpu_host->posx[i] << " cout cpu: " << cpu_host->posx[i] << endl;
    }
    if (gpu_host->posy[i] - cpu_host->posy[i] > (epsilon*cpu_host->posy[i])) {
      cout << "Summation y ERROR at i: " << i << " cout gpu: " << gpu_host->posy[i] << " cout cpu: " << cpu_host->posy[i] << endl;
    }
    if (gpu_host->posz[i] - cpu_host->posz[i] > (epsilon*cpu_host->posz[i])) {
      cout << "Summation z ERROR at i: " << i << " cout gpu: " << gpu_host->posz[i] << " cout cpu: " << cpu_host->posz[i] << endl;
    }
    if (abs(gpu_host->mass[i] - cpu_host->mass[i]) > (epsilon*cpu_host->mass[i])) {
      cout << "Summation mass ERROR at i: " << i << " cout gpu: " << gpu_host->mass[i] << " cout cpu: " << cpu_host->mass[i] << endl;
    }
    if (gpu_host->count[i] - cpu_host->count[i] > 0) {
      cout << "Summation count ERROR at point: " << i << " cout gpu: " << gpu_host->count[i] << " cout cpu: " << cpu_host->count[i] << endl;
    }
  }
}

void CalculateSorted(int index, HostMemory* host_memory, int start, int num_nodes) {
  int bottom_node = host_memory->bottom;
  for (int i = 0; i < 8; i++) {
    int child_index = host_memory->child[index*8+i];
    if (child_index >= num_nodes) {
      host_memory->start[child_index] = start;
      start += host_memory->count[child_index];
      CalculateSorted(child_index, host_memory, start, num_nodes);
    } else if (child_index >= 0) {
      host_memory->sort[index] = child_index;
      start++;
    }
  }
}

void CheckSorted(HostMemory* gpu_host, HostMemory* cpu_host, int num_nodes, int num_bodies) {
  for (int i = 0; i < num_bodies; i++) {
    if (gpu_host->sort[i] != cpu_host->sort[i]) {
      cout << "Error for sorting at index: " << i << "sorted gpu : " << gpu_host->sort[i] << "sorted cpu: " << cpu_host->sort[i];
    }
    if (gpu_host->start[i] != cpu_host->start[i]) {
      cout << "Error for start at index: " << i << "sorted gpu : " << gpu_host->start[i] << "sorted cpu: " << cpu_host->start[i];
    }
  }
}

void CalculateForce(HostMemory *host_memory, num_bodies) {
  float dq[MAXDEPTH];
  float parent_index[MAXDEPTH];
  float child_index[MAXDEPTH];
  float px, py, pz;
  dq[0] = 1 / (0.5 * 0.5);
  for (int i = 1; i < host_memory->max_depth; i++) {
    dq[i] = dq[i - 1] * 0.25f;
  }
  if (max_depth > MAXDEPTH) {
    return 1/0;
  }
  for (int k = 0; k < num_bodies; k+=WARPSIZE) {
    //int index = host_memory->sort[k];
    ax = 0.0f;
    ay = 0.0f;
    az = 0.0f;
    parent_index[0] = num_nodes;
    child_index[0] = 0;
    while (depth >= 0) {
      while (child_index[depth] < 8) {
        child = host_memory->child[parent_index[depth]*8+child_index[depth]];
        child_index[depth]++;
        if (child >= 0) {
          bool go_deeper;
          for (int j = 0; j < WARPSIZE; j++) {
            int index = host_memory->sort[k+j];
            px = host_memory->posx[index];
            py = host_memory->posy[index];
            pz = host_memory->posz[index];
            dx = host_memory->posx[child] - px;
            dy = host_memory->posy[child] - py;
            dz = host_memory->posz[child] - pz;
            tmp = dx*dx + (dy*dy + (dz*dz + 0.00001));
            if (! (child <= num_bodies || tmp >= dq[depth]) )  {
              go_deeper = true;
            }
          }
          if (!go_deeper) {
            temp = rsqrtf(tmp);
            temp = host-memory->mass[child] * temp * temp * temp;
            for (int j = 0; i < WARPSIZE; j++) {
              px = host_memory->posx[index];
              py = host_memory->posy[index];
              pz = host_memory->posz[index];
              dx = host_memory->posx[child] - px;
              dy = host_memory->posy[child] - py;
              dz = host_memory->posz[child] - pz;
              tmp = dx*dx + (dy*dy + (dz*dz + 0.00001));
              ax += dx * temp;
              ay += dy * temp;
              az += dz * temp;
            }
          } else {
            depth++;
            parent_index[depth] = child;
            child_index[depth] = 0;
          }
        } else {
          depth = max(0, depth - 1);
        }
      }
      depth--;
    }
    if (step > 0) {
      host_memory->velx[




    
}


void ReadFromGpu(cl_vars_t* cv, KernelArgs* args, HostMemory* host_memory) {
  int num_nodes = args->num_nodes;
  int num_bodies = args->num_bodies;
  cl_int err;
  err = clEnqueueReadBuffer(cv->commands, args->posx, true, 0,
      sizeof(float)*(num_nodes + 1), host_memory->posx, 0, NULL, NULL);
  err = clEnqueueReadBuffer(cv->commands, args->posy, true, 0,
      sizeof(float)*(num_nodes + 1), host_memory->posy, 0, NULL, NULL);
  err = clEnqueueReadBuffer(cv->commands, args->count, true, 0, sizeof(int)*(num_nodes+1), host_memory->count, 0,
    NULL, NULL);
  err = clEnqueueReadBuffer(cv->commands, args->posz, true, 0, sizeof(float)*(num_nodes + 1), host_memory->posz, 0,
    NULL, NULL);
  err = clEnqueueReadBuffer(cv->commands, args->child, true, 0, sizeof(int)*8*(num_nodes + 1), host_memory->child, 0,
    NULL, NULL);
  err = clEnqueueReadBuffer(cv->commands, args->mass, true, 0, sizeof(float)*(num_nodes + 1), host_memory->mass, 0,
    NULL, NULL);
  err = clEnqueueReadBuffer(cv->commands, args->start, true, 0, sizeof(int)*(num_nodes + 1), host_memory->start, 0,
    NULL, NULL);
  err = clEnqueueReadBuffer(cv->commands, args->step, true, 0, sizeof(int), &host_memory->step, 0,
    NULL, NULL);
  err = clEnqueueReadBuffer(cv->commands, args->bottom, true, 0, sizeof(int), &host_memory->bottom, 0,
    NULL, NULL);
  float radius;
  err = clEnqueueReadBuffer(cv->commands, args->radius, true, 0, sizeof(float), &radius, 0,
    NULL, NULL);
  CHK_ERR(err);
}



void DebuggingPrintValue(cl_vars_t* cv, KernelArgs* args, HostMemory *host_memory, bool check_tree_position){
  cl_int err;
  int num_nodes = args->num_nodes;
  int num_bodies = args->num_bodies;

  printf("x: %.20f\n",host_memory->posx[num_nodes]);
  printf("y: %.20f \n", host_memory->posy[num_nodes]);
  printf("z: %.20f \n", host_memory->posz[num_nodes]);
  printf("mass: %f \n", host_memory->mass[num_nodes]);
  printf("startd: %d \n", host_memory->start[num_nodes]);
  printf("stepd_num: %d \n", host_memory->step);
  printf("bottom_num: %d \n", host_memory->bottom);
  float radius;
  err = clEnqueueReadBuffer(cv->commands, args->radius, true, 0, sizeof(float), &radius, 0, NULL, NULL);
  printf("radius: %f \n", radius);
  int k = args->num_nodes * 8;
  for(int i = 0; i < 8; i++) {
    int index = host_memory->child[k + i];
    printf("child: %d \n", index);
    if (index != -1) {
    printf("child x: %.12f \n", host_memory->posx[index]);
    printf("child y: %.12f \n", host_memory->posy[index]);
    printf("child z: %.12f \n", host_memory->posz[index]);
    }
  }
  if (check_tree_position) {
    CheckTree(num_nodes, host_memory, num_bodies);
  }
}


int main (int argc, char *argv[])
{
  int split = 100;
  int num_bodies = pow(split, 3);
  printf("Number Bodies: %d \n", num_bodies);
  int blocks = 4; // TODO Supposed to be set to multiprocecsor count

  int num_nodes = num_bodies * 2;
  if (num_nodes < 1024*blocks) num_nodes = 1024*blocks;
  while ((num_nodes & (WARPSIZE - 1)) != 0) num_nodes++;
  num_nodes--;

  KernelArgs args;
  args.num_nodes = num_nodes;
  args.num_bodies = num_bodies;

  HostMemory host_memory;
  AllocateHostMemory(&host_memory, num_nodes, num_bodies);

  // Sample Values
  for (int i = 0; i < split; i++) {
    for (int j = 0; j < split; j++) {
      for (int k = 0; k < split; k++) {
        host_memory.posx[i*(split*split)+j*(split)+k] = i;
        host_memory.posy[i*(split*split)+j*(split)+k] = j;
        host_memory.posz[i*(split*split)+j*(split)+k] = k;
        host_memory.mass[i*(split*split)+j*(split)+k] = i+j+k;
      }
    }
  }

  std::string kernel_source_str;

  std::list<std::string> kernel_names;

  string bounding_box_name_str = std::string("bound_box");
  string build_tree_name_str = std::string("build_tree");
  string compute_sums_name_str = string("compute_sums");
  string sort_name_str = string("sort");
  string calculate_forces_name_str = string("calculate_forces");

  kernel_names.push_back(bounding_box_name_str);
  kernel_names.push_back(build_tree_name_str);
  kernel_names.push_back(compute_sums_name_str);
  kernel_names.push_back(sort_name_str);
  kernel_names.push_back(calculate_forces_name_str);

  std::string kernel_file = std::string("bound_box.cl");

  std::map<std::string, cl_kernel>kernel_map;

  readFile(kernel_file, kernel_source_str);

  cl_vars_t cv;
  initialize_ocl(cv);
  compile_ocl_program(kernel_map, cv, kernel_source_str.c_str(),
      kernel_names);

  cl_int err = CL_SUCCESS;

  CreateMemBuffer(&cv, &args, &host_memory);


  // Set local work size and global work sizes <]
  // TODO CAN BE optimized.
  size_t local_work_size[1] = {THREADS1};
  size_t global_work_size[1] = {THREADS1};
  size_t num_work_groups = 2;

  // Set the Kernel Arguements for bounding box
  SetArgs(&kernel_map[bounding_box_name_str], &args);
  SetArgs(&kernel_map[build_tree_name_str], &args);
  SetArgs(&kernel_map[compute_sums_name_str], &args);
  SetArgs(&kernel_map[sort_name_str], &args);
  SetArgs(&kernel_map[calculate_forces_name_str], &args);

  err = clEnqueueNDRangeKernel(cv.commands, kernel_map[bounding_box_name_str], 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
  err = clEnqueueNDRangeKernel(cv.commands, kernel_map[build_tree_name_str], 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);

 //Read memory and calculate summation tree on CPU
 //
  HostMemory host_memory_test;
  AllocateHostMemory(&host_memory_test, num_nodes, num_bodies);
  err = clFinish(cv.commands);
  ReadFromGpu(&cv, &args, &host_memory_test);
  CalculateSummation(&cv, &args, &host_memory_test);
  // Run summation Kernel
  err = clEnqueueNDRangeKernel(cv.commands, kernel_map[compute_sums_name_str], 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
  CHK_ERR(err);
  err = clFinish(cv.commands);
  ReadFromGpu(&cv, &args, &host_memory);
  CheckSummation(&host_memory, &host_memory_test, num_nodes);
  err = clFinish(cv.commands);

  // TODO These tests can be condenses
  HostMemory host_memory_before_sorted;
  AllocateHostMemory(&host_memory_before_sorted, num_nodes, num_bodies);
  ReadFromGpu(&cv, &args, &host_memory_before_sorted);
  CalculateSorted(num_nodes, &host_memory_before_sorted, 0, num_nodes);



  //// Run Sorted Kernel
  err = clEnqueueNDRangeKernel(cv.commands, kernel_map[sort_name_str], 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
  ReadFromGpu(&cv, &args, &host_memory);
  CheckSorted(&host_memory_before_sorted, &host_memory, num_nodes, num_bodies);

  err = clEnqueueNDRangeKernel(cv.commands, kernel_map[calculate_forces_name_str], 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);

  //DebuggingPrintValue(&cv, &args, &host_memory, false);

}

