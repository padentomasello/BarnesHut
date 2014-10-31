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

#define WARPSIZE 16
#define MAXDEPTH 32
//
// random number generator

#define MULT 1103515245
#define ADD 12345
#define MASK 0x7FFFFFFF
#define TWOTO31 2147483648.0

static int A = 1;
static int B = 0;
static int randx = 1;
static int lastrand;


static void drndset(int seed)
{
   A = 1;
   B = 0;
   randx = (A * seed + B) & MASK;
   A = (MULT * A) & MASK;
   B = (MULT * B + ADD) & MASK;
}


static double drnd()
{
   lastrand = randx;
   randx = (A * randx + B) & MASK;
   return (double)lastrand / TWOTO31;
}

struct KernelArgs{
  cl_mem posx, posy, posz, child, mass, start, minx, maxx, miny, maxy, minz, maxz, blocked, step, bottom, max_depth, radius, count;
  cl_mem velx, vely, velz, accx, accy, accz, sort;
  int num_bodies, num_nodes;
  int num_args;
};

struct HostMemory {
  float *mass, *posx, *posy, *posz, *velx, *vely, *velz, *accx, *accy, *accz;
  int* start, *child, *count, *sort;
  int step, max_depth, bottom, blocked, num_nodes, num_bodies;
  float radius;
};

void CreateMemBuffer (cl_vars_t* cv, KernelArgs* args, HostMemory* host_memory) {
  cl_int err;
  int num_nodes = args->num_nodes;
  int num_bodies = args->num_bodies;
  // TODO This shouldn't be hardcodes
  int num_work_groups = 40;
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
  cout << "INC" << inc << endl;
  inc = inc * sizeof(float);
    cl_buffer_region velxl_region, velyl_region, velzl_region, accxl_region, accyl_region, acczl_region, sortl_region;
    velxl_region.origin = 0;
    velxl_region.size = inc;
    velyl_region.origin = inc;
    velyl_region.size = inc;
    velzl_region.origin = 2*inc;
    velzl_region.size = inc;
    accxl_region.origin = 3*inc;
    accxl_region.size = inc;
    accyl_region.origin = 4*inc;
    accyl_region.size = inc;
    acczl_region.origin = 5*inc;
    acczl_region.size = inc;
    sortl_region.origin = 6*inc;
    sortl_region.size = inc;
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
  err = clSetKernelArg(*kernel, 12, sizeof(cl_mem), &args->sort);
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
  host->num_nodes = num_nodes;
  host->num_bodies = num_bodies;
  int inc = (num_bodies + WARPSIZE -1) & (-WARPSIZE);
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
  host->velx = (float *)malloc(sizeof(float) * inc);
  if (host->velx == NULL) {fprintf(stderr, "cannot allocate velx\n");  exit(-1);}
  host->vely = (float *)malloc(sizeof(float) * inc);
  if (host->vely == NULL) {fprintf(stderr, "cannot allocate vely\n");  exit(-1);}
  host->velz = (float *)malloc(sizeof(float) * inc);
  if (host->velz == NULL) {fprintf(stderr, "cannot allocate velz\n");  exit(-1);}
  host->accx = (float *)malloc(sizeof(float) * inc);
  if (host->accx == NULL) {fprintf(stderr, "cannot allocate velx\n");  exit(-1);}
  host->accy = (float *)malloc(sizeof(float) * inc);
  if (host->accy == NULL) {fprintf(stderr, "cannot allocate vely\n");  exit(-1);}
  host->accz = (float *)malloc(sizeof(float) * inc);
  if (host->accz == NULL) {fprintf(stderr, "cannot allocate velz\n");  exit(-1);}
  host->child = (int *)malloc(sizeof(int) * 8*(num_nodes + 1));
  if (host->child == NULL) {fprintf(stderr, "cannot allocate velz\n");  exit(-1);}
  // TODO This can be removed after debugging. Count is only used on GPU
  host->count = (int *)malloc(sizeof(int) *(num_nodes + 1));
  if (host->count == NULL) {fprintf(stderr, "cannot allocate velz\n");  exit(-1);}
  host->sort = (int *)malloc(sizeof(int) *(inc));
  if (host->sort == NULL) {fprintf(stderr, "cannot allocate velz\n");  exit(-1);}
}

void CalculateSummation(cl_vars_t* cv, KernelArgs* args, HostMemory* host_memory) {
  int bottom = host_memory->bottom;
  int num_nodes = args->num_nodes;
  int num_bodies = args->num_bodies;
  int child, cnt, j;

  cout << "bottom: " << bottom << endl;
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
  const float epsilon = 0.00001;
  for(int i = cpu_host->bottom; i <= num_nodes; i++) {
    if (abs((gpu_host->posx[i] - cpu_host->posx[i])/cpu_host->posx[i]) > epsilon) {
      cout << "Summation x ERROR at i: " << i << " cout gpu: " << gpu_host->posx[i] << " cout cpu: " << cpu_host->posx[i] << endl;
    }
    if (abs((gpu_host->posy[i] - cpu_host->posy[i])/cpu_host->posy[i]) > epsilon) {
      cout << "Summation y ERROR at i: " << i << " cout gpu: " << gpu_host->posy[i] << " cout cpu: " << cpu_host->posy[i] << endl;
    }
    if (abs((gpu_host->posz[i] - cpu_host->posz[i])/cpu_host->posz[i]) > epsilon) {
      cout << "Summation z ERROR at i: " << i << " cout gpu: " << gpu_host->posz[i] << " cout cpu: " << cpu_host->posz[i] << endl;
    }
    if (abs((gpu_host->mass[i] - cpu_host->mass[i])/cpu_host->mass[i]) > epsilon) {
      cout << "Summation mass ERROR at i: " << i << " cout gpu: " << gpu_host->mass[i] << " cout cpu: " << cpu_host->mass[i] << endl;
    }
    if (gpu_host->count[i] - cpu_host->count[i] > 0) {
      cout << "Summation count ERROR at point: " << i << " cout gpu: " << gpu_host->count[i] << " cout cpu: " << cpu_host->count[i] << endl;
    }
  }
}

void CalculateSorted(int index, HostMemory* host_memory, int start, int num_nodes) {
  int bottom_node = host_memory->bottom;
  int num_bodies = host_memory->num_bodies;
  for (int i = 0; i < 8; i++) {
    int child_index = host_memory->child[index*8+i];
    if (child_index >= host_memory->num_bodies) {
      host_memory->start[child_index] = start;
      CalculateSorted(child_index, host_memory, start, num_nodes);
      start += host_memory->count[child_index];
    } else if (child_index >= 0) {
      host_memory->sort[start] = child_index;
      start++;
    }
  }
}

void CheckSorted(HostMemory* gpu_host, HostMemory* cpu_host, int num_nodes, int num_bodies) {
  for (int i = 0; i < num_bodies; i++) {
    //cout << "sorted " << gpu_host->sort[i] << endl;
    if (gpu_host->sort[i] != cpu_host->sort[i]) {
      cout << "Error for sorting at index: " << i << " sorted gpu : " << gpu_host->sort[i] << " sorted cpu: " << cpu_host->sort[i] << endl;
    }
    if (gpu_host->start[i] != cpu_host->start[i]) {
      cout << "Error for start at index: " << i << " started gpu : " << gpu_host->start[i] << " started cpu: " << cpu_host->start[i] << endl;
    }
  }
}

void CalculateForce(HostMemory *host_memory, int num_bodies) {
  float dq[MAXDEPTH];
  int parent_index[MAXDEPTH];
  int child_index[MAXDEPTH];
  int child;
  float px[WARPSIZE], py[WARPSIZE], pz[WARPSIZE], dx[WARPSIZE], dy[WARPSIZE], dz[WARPSIZE], temp[WARPSIZE],
        ax[WARPSIZE], ay[WARPSIZE], az[WARPSIZE];
  int itolsqd = 1 / (0.5 * 0.5);
  int max_depth = host_memory->max_depth;
  float temp1 = host_memory->radius;
  cout << "Radius" << temp1 << endl;
  cout << host_memory->max_depth;
  dq[0] = temp1 * temp1 * itolsqd;
  for (int i = 1; i < host_memory->max_depth; i++) {
    cout << "TEST";
    dq[i] = dq[i - 1] * 0.25f;
    cout << "dq[i] " << dq[i] << " i" << endl;
  }
  if (max_depth > MAXDEPTH) {
    dq[0] = 1/0;
  }
  int k;
  for (k = 0; k + WARPSIZE < num_bodies; k+=WARPSIZE) {
    for (int i = 0; i < WARPSIZE; i++) {
      ax[i] = 0.0f;
      ay[i] = 0.0f;
      az[i] = 0.0f;
    }
    parent_index[0] = host_memory->num_nodes;
    child_index[0] = 0;
    int depth = 0;
    while (depth >= 0) {
      while (child_index[depth] < 8) {
        child = host_memory->child[parent_index[depth]*8+child_index[depth]];
        child_index[depth]++;
        if (child >= 0) {
          bool go_deeper = false;
          for (int j = 0; j < WARPSIZE; j++) {
            //cout << k + j << endl;
            int index = host_memory->sort[k+j];
            //cout << index << endl;
            px[j] = host_memory->posx[index];
            py[j] = host_memory->posy[index];
            pz[j] = host_memory->posz[index];
            dx[j] = host_memory->posx[child] - px[j];
            dy[j] = host_memory->posy[child] - py[j];
            dz[j] = host_memory->posz[child] - pz[j];
            temp[j] = dx[j]*dx[j] + (dy[j]*dy[j] + (dz[j]*dz[j] + 0.0001f));
            if (k == 0 && child == 241) {
              cout << "cond: "<< (temp[j] >= dq[depth]) << endl;
            }
            if (k == 0 && child == 241 ) {
              cout << "temp: "<< (temp[j]) << endl;
            }
            if (k == 0 && child == 241) {
              cout << "dq: "<< dq[depth] << endl;
            }
            if (k == 0 && child == 241) {
              cout << "depth: "<< depth << endl;
            }
            if (! (child <= num_bodies || temp[j] >= dq[depth]) )  {
              go_deeper = true; 
            }
          }
          //if (k == 0) {
            //cout << "Go deaper: " << go_deeper << " child: " << child << endl;
          //}
          if (!go_deeper) {
            for (int j = 0; j < WARPSIZE; j++) {
              temp[j] = 1 / sqrt(temp[j]);
              temp[j] = host_memory->mass[child] * temp[j] * temp[j] * temp[j];
              ax[j] += dx[j] * temp[j];
              ay[j] += dy[j] * temp[j];
              az[j] += dz[j] * temp[j];
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
    if (host_memory->step > 0) {
      for (int j = 0; j < WARPSIZE; j++) {
        int index = host_memory->sort[k+j];
        host_memory->velx[index] += (ax[j] - host_memory->accx[index]);
        host_memory->vely[index] += (ay[j] - host_memory->accy[index]);
        host_memory->velz[index] += (az[j] - host_memory->accz[index]);
      }
    }

    for (int j = 0; j < WARPSIZE; j++) {
      int index = host_memory->sort[k+j];
      host_memory->accx[index] = ax[j];
      host_memory->accy[index] = ay[j];
      host_memory->accz[index] = az[j];
    }
  }
  for(; k < num_bodies; k++) {
    int index = host_memory->sort[k];
    cout << "index: " << index << endl;
    float px1 = host_memory->posx[index];
    float py1 = host_memory->posy[index];
    float pz1 = host_memory->posz[index];
    //if (index == 0) {
      cout << "child:  "<< child << " dx: " << dx << " dy: " << dy << " dz: " << dz << endl;
      cout << "child:  "<< child << " px: " << px1 << " py: " << py1 << " pz: " << pz1 << endl;
    //}
    float ax = 0.0f;
    float ay = 0.0f;
    float az = 0.0f;
    int depth = 0;
    parent_index[depth] = host_memory->num_nodes;
    child_index[depth] = 0;
    while (depth >= 0) {
      while (child_index[depth] < 8) {
        child = host_memory->child[parent_index[depth]*8 + child_index[depth]];
        child_index[depth]++;
        if (child >= 0) {
          float dx = host_memory->posx[child] - px1;
          float dy = host_memory->posy[child] - py1;
          float dz = host_memory->posz[child] - pz1;
          //if (index == 0) {
            //cout << "child:  "<< child << " dx: " << dx << " dy: " << dy << " dz: " << dz << endl;
            //cout << "child:  "<< child << " px: " << px << " py: " << py << " pz: " << pz << endl;
          //}
          float tmp = dx*dx + (dy*dy + (dz*dz + 0.0001f));
          if (child < num_bodies) {
            tmp = 1 / sqrt(tmp);
            tmp = host_memory->mass[child] * tmp * tmp * tmp;
            ax += dx * tmp;
            ay += dy * tmp;
            az += dz * tmp;
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
    if (host_memory->step > 0) {
        host_memory->velx[index] += (ax - host_memory->accx[index]);
        host_memory->vely[index] += (ay - host_memory->accy[index]);
        host_memory->velz[index] += (az - host_memory->accz[index]);
    }
    host_memory->accx[index] = ax;
    host_memory->accy[index] = ay;
    host_memory->accz[index] = az;
  }

}

void CheckForces(HostMemory* gpu_host, HostMemory* cpu_host) {
  const float epsilon = 0.01f;
  for (int i = 0; i < cpu_host->num_bodies; i++) {
    if (abs((gpu_host->velx[i] - cpu_host->velx[i])/cpu_host->velx[i]) > epsilon) {
     cout << "Error at index: " << i << " for velx, cpu : " << cpu_host->velx[i] << " gpu : " <<  gpu_host->velx[i] << endl;
    }
    if (abs((gpu_host->vely[i] - cpu_host->vely[i])/cpu_host->vely[i]) > epsilon) {
     cout << "Error at index: " << i << " for vely, cpu : " << cpu_host->vely[i] << " gpu : " <<  gpu_host->vely[i] << endl;
    }
    if (abs((gpu_host->velz[i] - cpu_host->velz[i])/cpu_host->velz[i]) > epsilon) {
     cout << "Error at index: " << i << " for velz, cpu : " << cpu_host->velz[i] << " gpu : " <<  gpu_host->velz[i] << endl;
    }
    //cout << "velz: " << cpu_host->velz[i] << endl;
    if (abs((gpu_host->accx[i] - cpu_host->accx[i])/cpu_host->accx[i]) > epsilon) {
     cout << "Error at index: " << i << " for accx, cpu : " << cpu_host->accx[i] << " gpu : " <<  gpu_host->accx[i] << endl;
    }
    if (abs((gpu_host->accy[i] - cpu_host->accy[i])/cpu_host->accy[i]) > epsilon) {
     cout << "Error at index: " << i << " for accy, cpu : " << cpu_host->accy[i] << " gpu : " <<  gpu_host->accy[i] << endl;
    }
    if (abs((gpu_host->accz[i] - cpu_host->accz[i])/cpu_host->accz[i]) > epsilon) {
     cout << "Error at index: " << i << " for accz, cpu : " << cpu_host->accz[i] << " gpu : " <<  gpu_host->accz[i] << endl;
    }
  }
}


void ReadFromGpu(cl_vars_t* cv, KernelArgs* args, HostMemory* host_memory) {
  int num_nodes = args->num_nodes;
  int num_bodies = args->num_bodies;
  int inc = (num_bodies + WARPSIZE -1) & (-WARPSIZE);
  cout << "inc : " << inc << endl;
  cl_int err;
  err = clEnqueueReadBuffer(cv->commands, args->posx, false, 0,
      sizeof(float)*(num_nodes + 1), host_memory->posx, 0, NULL, NULL);
  CHK_ERR(err);
  err = clEnqueueReadBuffer(cv->commands, args->posy, false, 0,
      sizeof(float)*(num_nodes + 1), host_memory->posy, 0, NULL, NULL);
  CHK_ERR(err);
  err = clEnqueueReadBuffer(cv->commands, args->count, false, 0, sizeof(int)*(num_nodes+1), host_memory->count, 0,
    NULL, NULL);
  CHK_ERR(err);
  err = clEnqueueReadBuffer(cv->commands, args->posz, false, 0, sizeof(float)*(num_nodes + 1), host_memory->posz, 0,
    NULL, NULL);
  CHK_ERR(err);
  err = clEnqueueReadBuffer(cv->commands, args->velx, false, 0, sizeof(float)*inc, host_memory->velx, 0,
    NULL, NULL);
  CHK_ERR(err);
  err = clEnqueueReadBuffer(cv->commands, args->vely, false, 0, sizeof(float)*inc, host_memory->vely, 0,
    NULL, NULL);
  CHK_ERR(err);
  err = clEnqueueReadBuffer(cv->commands, args->velz, false, 0, sizeof(float)*inc, host_memory->velz, 0,
    NULL, NULL);
  CHK_ERR(err);
  err = clEnqueueReadBuffer(cv->commands, args->accx, false, 0, sizeof(float)*inc, host_memory->accx, 0,
    NULL, NULL);
  CHK_ERR(err);
  err = clEnqueueReadBuffer(cv->commands, args->accy, false, 0, sizeof(float)*inc, host_memory->accy, 0,
    NULL, NULL);
  CHK_ERR(err);
  err = clEnqueueReadBuffer(cv->commands, args->accz, false, 0, sizeof(float)*inc, host_memory->accz, 0,
    NULL, NULL);
  CHK_ERR(err);
  err = clEnqueueReadBuffer(cv->commands, args->child, false, 0, sizeof(int)*8*(num_nodes + 1), host_memory->child, 0,
    NULL, NULL);
  CHK_ERR(err);
  err = clEnqueueReadBuffer(cv->commands, args->sort, false, 0, sizeof(int)*inc, host_memory->sort, 0,
    NULL, NULL);
  CHK_ERR(err);
  err = clEnqueueReadBuffer(cv->commands, args->mass, false, 0, sizeof(float)*(num_nodes + 1), host_memory->mass, 0,
    NULL, NULL);
  CHK_ERR(err);
  err = clEnqueueReadBuffer(cv->commands, args->start, false, 0, sizeof(int)*(num_nodes + 1), host_memory->start, 0,
    NULL, NULL);
  CHK_ERR(err);
  err = clEnqueueReadBuffer(cv->commands, args->step, false, 0, sizeof(int), &host_memory->step, 0,
    NULL, NULL);
  CHK_ERR(err);
  err = clEnqueueReadBuffer(cv->commands, args->bottom, false, 0, sizeof(int), &host_memory->bottom, 0,
    NULL, NULL);
  CHK_ERR(err);
  err = clEnqueueReadBuffer(cv->commands, args->max_depth, false, 0, sizeof(int), &host_memory->max_depth, 0,
    NULL, NULL);
  CHK_ERR(err);
  err = clEnqueueReadBuffer(cv->commands, args->radius, false, 0, sizeof(float), &host_memory->radius, 0,
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
  int num_bodies = atoi(argv[1]);
  if (num_bodies < 1) {
    fprintf(stderr, "nbodies is too small: %d\n", num_bodies);
    exit(-1);
  }
  if (num_bodies > (1 << 30)) {
    fprintf(stderr, "nbodies is too large: %d\n", num_bodies);
    exit(-1);
  }
  int num_nodes = num_bodies * 2;
  int blocks = 32; // TODO Supposed to be set to multiprocecsor count
  if (num_nodes < 1024*blocks) num_nodes = 1024*blocks;
  while ((num_nodes & (WARPSIZE - 1)) != 0) num_nodes++;
  num_nodes--;

  printf("Number Bodies: %d \n", num_bodies);
  cout << "work group size: " << CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE << endl;
  cout << "work group size: " << CL_KERNEL_WORK_GROUP_SIZE << endl;


  KernelArgs args;
  args.num_nodes = num_nodes;
  args.num_bodies = num_bodies;

  HostMemory host_memory;
  AllocateHostMemory(&host_memory, num_nodes, num_bodies);

  float rsc, vsc, r, x, y, z, sq, scale, v;
  int i;
  drndset(7);
  rsc = (3 * 3.1415926535897932384626433832795) / 16;
  vsc = sqrt(1.0 / rsc);
  for (i = 0; i < num_bodies; i++) {
    host_memory.mass[i] = 1.0 / num_bodies;
    r = 1.0 / sqrt(pow(drnd()*0.999, -2.0/3.0) - 1);
    do {
      x = drnd()*2.0 - 1.0;
      y = drnd()*2.0 - 1.0;
      z = drnd()*2.0 - 1.0;
      sq = x*x + y*y + z*z;
    } while (sq > 1.0);
    scale = rsc * r / sqrt(sq);
    host_memory.posx[i] = x * scale;
    host_memory.posy[i] = y * scale;
    host_memory.posz[i] = z * scale;

    do {
      x = drnd();
      y = drnd() * 0.1;
    } while (y > x*x * pow(1 - x*x, 3.5));
    v = x * sqrt(2.0 / sqrt(1 + r*r));
    do {
      x = drnd()*2.0 - 1.0;
      y = drnd()*2.0 - 1.0;
      z = drnd()*2.0 - 1.0;
      sq = x*x + y*y + z*z;
    } while (sq > 1.0);
    scale = vsc * v / sqrt(sq);
    host_memory.velx[i] = x * scale;
    host_memory.vely[i] = y * scale;
    host_memory.velz[i] = z * scale;
  }

  // Sample Values
  //for (int i = 0; i < split; i++) {
    //for (int j = 0; j < split; j++) {
      //for (int k = 0; k < split; k++) {
        //host_memory.posx[i*(split*split)+j*(split)+k] = i;
        //host_memory.posy[i*(split*split)+j*(split)+k] = j;
        //host_memory.posz[i*(split*split)+j*(split)+k] = k;
        ////host_memory.velx[i*(split*split)+j*(split)+k] = 1;
        ////host_memory.vely[i*(split*split)+j*(split)+k] = 1;
        ////host_memory.velz[i*(split*split)+j*(split)+k] = 1;
        //host_memory.mass[i*(split*split)+j*(split)+k] = i+j+k;
      //}
    //}
  //}

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
  size_t cb;
  initialize_ocl(cv);
  compile_ocl_program(kernel_map, cv, kernel_source_str.c_str(),
      kernel_names);
  //cl_context context = clCreateContextFromType(0,
    //CL_DEVICE_TYPE_GPU, NULL, NULL, NULL);
  //clGetContextInfo(context, CL_CONTEXT_DEVICES, 0,
    //NULL, &cb);
    //devices = malloc(cb);
  //clGetContextInfo(context, CL_CONTEXT_DEVICES, cb,
  //devices, NULL);
  cl_int err = CL_SUCCESS;

  CreateMemBuffer(&cv, &args, &host_memory);


  // Set local work size and global work sizes <]
  // TODO CAN BE optimized.
  size_t local_work_size[1] = {THREADS1};
  size_t global_work_size[1] = {THREADS1};

  //cout << clGetKernelWorkGroupInfo ( kernel_map[bounding_box_name_str], cl_device_id device, 
      //cl_kernel_work_group_info param_name, size_t param_value_size, void *param_value,
      //size_t *param_value_size_ret) << endl;

  // Set the Kernel Arguements for bounding box
  SetArgs(&kernel_map[bounding_box_name_str], &args);
  SetArgs(&kernel_map[build_tree_name_str], &args);
  SetArgs(&kernel_map[compute_sums_name_str], &args);
  SetArgs(&kernel_map[sort_name_str], &args);
  SetArgs(&kernel_map[calculate_forces_name_str], &args);

  err = clEnqueueNDRangeKernel(cv.commands, kernel_map[bounding_box_name_str], 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
  CHK_ERR(err);
  err = clEnqueueNDRangeKernel(cv.commands, kernel_map[build_tree_name_str], 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
  CHK_ERR(err);

 //Read memory and calculate summation tree on CPU
 //
  HostMemory host_memory_test;
  AllocateHostMemory(&host_memory_test, num_nodes, num_bodies);
  err = clFinish(cv.commands);
  ReadFromGpu(&cv, &args, &host_memory_test);
  err = clFinish(cv.commands);
  CalculateSummation(&cv, &args, &host_memory_test);
  // Run summation Kernel
  err = clEnqueueNDRangeKernel(cv.commands, kernel_map[compute_sums_name_str], 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
  CHK_ERR(err);
  err = clFinish(cv.commands);
  ReadFromGpu(&cv, &args, &host_memory);
  clFinish(cv.commands);
  CheckSummation(&host_memory, &host_memory_test, num_nodes);
  err = clFinish(cv.commands);

  //// TODO These tests can be condenses
  //HostMemory host_memory_before_sorted;
  //AllocateHostMemory(&host_memory_before_sorted, num_nodes, num_bodies);
  //ReadFromGpu(&cv, &args, &host_memory_before_sorted);
  //DebuggingPrintValue(&cv, &args, &host_memory_before_sorted, false);
  //CalculateSorted(num_nodes, &host_memory_before_sorted, 0, num_nodes);



  //// Run Sorted Kernel
  err = clEnqueueNDRangeKernel(cv.commands, kernel_map[sort_name_str], 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
  //CHK_ERR(err);
  //clFlush(cv.commands);
  //ReadFromGpu(&cv, &args, &host_memory);
  //CheckSorted(&host_memory, &host_memory_before_sorted, num_nodes, num_bodies);


  HostMemory host_memory_cpu_force_calc;
  AllocateHostMemory(&host_memory_cpu_force_calc, num_nodes, num_bodies);
  ReadFromGpu(&cv, &args, &host_memory_cpu_force_calc);
  clFinish(cv.commands);
  CalculateForce(&host_memory_cpu_force_calc, num_bodies);
  err = clEnqueueNDRangeKernel(cv.commands, kernel_map[calculate_forces_name_str], 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
  ReadFromGpu(&cv, &args, &host_memory);
  clFinish(cv.commands);
  CheckForces(&host_memory, &host_memory_cpu_force_calc);

  DebuggingPrintValue(&cv, &args, &host_memory, false);

}

