#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cmath>
#include <unistd.h>

#include "clhelp.h"

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
  float *mass, *posx, *posy, *posz, *velx, *vely, *velz, *child;
  int* start;
};

void CreateMemBuffer (cl_vars_t* cv, KernelArgs* args, HostMemory* host_memory) {
  cl_int err;
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
      sizeof(float)*(num_nodes + 1), host_memory.posy, &err);
  args->posy = clCreateBuffer(cv.context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE,
      sizeof(float)*(num_nodes + 1), host_memory->posz, &err);
  CHK_ERR(err);
  args->mass = clCreateBuffer(cv->context, CL_MEM_READ_WRITE,
      sizeof(float)*(num_nodes + 1), NULL, &err);
  CHK_ERR(err);
  args->count = clCreateBuffer(cv->context, CL_MEM_READ_WRITE,
      sizeof(float)*(num_nodes + 1), NULL, &err);
  CHK_ERR(err);
  args->start = clCreateBuffer(cv->context, CL_MEM_READ_WRITE,
      sizeof(int)*(num_nodes + 1), NULL, &err);
  CHK_ERR(err);
  args->child = clCreateBuffer(cv->context, CL_MEM_READ_WRITE,
      sizeof(float)*8*(num_nodes + 1), NULL, &err);
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
  err = clSetKernelArg(*kernel, 3, sizeof(cl_mem), &args->child);
  CHK_ERR(err);
  err = clSetKernelArg(*kernel, 4, sizeof(cl_mem), &args->mass);
  CHK_ERR(err);
  err = clSetKernelArg(*kernel, 5, sizeof(cl_mem), &args->start);
  CHK_ERR(err);
  err = clSetKernelArg(*kernel, 6, sizeof(cl_mem), &args->minx);
  CHK_ERR(err);
  err = clSetKernelArg(*kernel, 7, sizeof(cl_mem), &args->maxx);
  CHK_ERR(err);
  err = clSetKernelArg(*kernel, 8, sizeof(cl_mem), &args->miny);
  CHK_ERR(err);
  err = clSetKernelArg(*kernel, 9, sizeof(cl_mem), &args->maxy);
  CHK_ERR(err);
  err = clSetKernelArg(*kernel, 10, sizeof(cl_mem), &args->minz);
  CHK_ERR(err);
  err = clSetKernelArg(*kernel, 11, sizeof(cl_mem), &args->maxz);
  CHK_ERR(err);
  err = clSetKernelArg(*kernel, 12, sizeof(cl_mem), &args->blocked);
  CHK_ERR(err);
  err = clSetKernelArg(*kernel, 13, sizeof(cl_mem), &args->step);
  CHK_ERR(err);
  err = clSetKernelArg(*kernel, 14, sizeof(cl_mem), &args->bottom);
  CHK_ERR(err);
  err = clSetKernelArg(*kernel, 15, sizeof(cl_mem), &args->max_depth);
  CHK_ERR(err);
  err = clSetKernelArg(*kernel, 16, sizeof(cl_mem), &args->radius);
  CHK_ERR(err);
  err = clSetKernelArg(*kernel, 17, sizeof(int), &args->num_bodies);
  CHK_ERR(err);
  err = clSetKernelArg(*kernel, 18, sizeof(int), &args->num_nodes);
  CHK_ERR(err);

}

void AllocateHostMemory(HostMemory* host, int num_nodes, int num_bodies) {
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
  host->child = (float *)malloc(sizeof(float) * 8*(num_nodes + 1));
  if (host->child == NULL) {fprintf(stderr, "cannot allocate velz\n");  exit(-1);}
}


int main (int argc, char *argv[])
{
  //register int i, run;
  //register int step, timesteps;
  //register int runtime, mintime;
  //int error;
  //register float dtime, dthf, epssq, itolsq;
  //float time, timing[7];
  //clock_t starttime, endtime;
  // TODO child_h not neccessary in actual computation
  int *start_h;
  float *maxxl, *maxyl, *maxzl;
  float *minxl, *minyl, *minzl;

  KernelArgs args;

  //int *errl, *sortl, *childl, *countl, *startl;
  //float *massl;
  //float *velxl, *velyl, *velzl;
  //float *accxl, *accyl, *acczl;
  register double rsc, vsc, r, v, x, y, z, sq, scale;
  args.num_bodies = 10;
  int num_bodies =10;
  int blocks = 4; // TODO Supposed to be set to multiprocecsor count

  int num_nodes = num_bodies * 2;
  if (num_nodes < 1024*blocks) num_nodes = 1024*blocks;
  while ((num_nodes & (WARPSIZE - 1)) != 0) num_nodes++;
  num_nodes--;
  args.num_nodes = num_nodes;

  HostMemory host_memory;
  AllocateHostMemory(&host_memory, num_nodes, num_bodies);

  for (int i = 0; i < num_bodies; i ++) {
    host_memory.posx[i] = i;
    host_memory.posy[i] = i;
    host_memory.posz[i] = i;
  }


  std::string bounding_box_kernel_str;
  std::string bounding_box_name_str = std::string("bound_box");
  std::string bounding_box_kernel_file = std::string("bound_box.cl");
  cl_kernel bound_box;

  readFile(bounding_box_kernel_file, bounding_box_kernel_str);

  cl_vars_t cv;
  initialize_ocl(cv);
  compile_ocl_program(bound_box, cv, bounding_box_kernel_str.c_str(),
      bounding_box_name_str.c_str());

  cl_mem posxl, posyl, poszl, minx_d, maxx_d, miny_d, maxy_d, minz_d, maxz_d, blocked, childl,
         velxl, velyl, velzl, accxl, accyl, acczl, sortl, massl, countl, startl;
  //cl_mem step_d, bottom_d, max_depth_d, radius_d;

  // Create Scalar buffers?
  cl_int err = CL_SUCCESS;



  /* Set local work size and global work sizes */
  // TODO CAN BE optimized.
  size_t local_work_size[1] = {THREADS1};
  size_t global_work_size[1] = {32*THREADS1};
  size_t num_work_groups = 2;
  //printf("WORK GORUP SIZE: %d", CL_DEVICE_MAX_WORK_GROUP_SIZE);

  // Used for global reduction in finding min and max of bounding box
  args.minx_d = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
      sizeof(float)*num_work_groups, NULL, &err);
  args.maxx_d = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
      sizeof(float)*num_work_groups, NULL, &err);
  args.miny_d = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
      sizeof(float)*num_work_groups, NULL, &err);
  args.maxy_d = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
      sizeof(float)*num_work_groups, NULL, &err);
  args.minz_d = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
      sizeof(float)*num_work_groups, NULL, &err);
  args.maxz_d = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
      sizeof(float)*num_work_groups, NULL, &err);
  args.maxz_d = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
      sizeof(float)*num_work_groups, NULL, &err);
  args.blocked = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
      1*sizeof(int), NULL, &err);
  CHK_ERR(err);
  // Global scalars //TODO is there a better way to do this?
  // TODO Would it be more efficient to use an InitializationKernel? See Cuda implementation around line 82
  int stepd_num = -1;
  int max_depth_num = -1;
  int bottom_num;
  int blocked_num = 0;
  err = clEnqueueWriteBuffer(cv.commands, args.blocked, true, 0, sizeof(int),
           &blocked_num, 0, NULL, NULL);
  err = clEnqueueWriteBuffer(cv.commands, args.step_d, true, 0, sizeof(int),
           &stepd_num, 0, NULL, NULL);
  err = clEnqueueWriteBuffer(cv.commands, args.max_depth_d, true, 0, sizeof(int),
           &max_depth_num, 0, NULL, NULL);

  // Set the Kernel Arguements for bounding box
  SetArgs(&bound_box, &args);

  err = clEnqueueNDRangeKernel(cv.commands, bound_box, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
  CHK_ERR(err);

  err = clEnqueueReadBuffer(cv.commands, args.posxl, true, 0, sizeof(float)*(num_nodes + 1), host_memory.posx, 0,
    NULL, NULL);
  err = clEnqueueReadBuffer(cv.commands, args.posyl, true, 0, sizeof(float)*(num_nodes + 1), host_memory.posy, 0,
    NULL, NULL);
  err = clEnqueueReadBuffer(cv.commands, args.poszl, true, 0, sizeof(float)*(num_nodes + 1), host_memory.posz, 0,
    NULL, NULL);
  err = clEnqueueReadBuffer(cv.commands, args.childl, true, 0, sizeof(float)*8*(num_nodes + 1), host_memory.child, 0,
    NULL, NULL);
  err = clEnqueueReadBuffer(cv.commands, args.massl, true, 0, sizeof(float)*(num_nodes + 1), host_memory.mass, 0,
    NULL, NULL);
  err = clEnqueueReadBuffer(cv.commands, args.startl, true, 0, sizeof(int)*(num_nodes + 1), host_memory.start, 0,
    NULL, NULL);
  err = clEnqueueReadBuffer(cv.commands, args.step_d, true, 0, sizeof(int), &stepd_num, 0,
    NULL, NULL);
  err = clEnqueueReadBuffer(cv.commands, args.bottom_d, true, 0, sizeof(int), &bottom_num, 0,
    NULL, NULL);
  float radius;
  err = clEnqueueReadBuffer(cv.commands, args.radius_d, true, 0, sizeof(float), &radius, 0,
    NULL, NULL);
  CHK_ERR(err);
  printf("x: %f", host_memory.posx[num_nodes]);
  printf("y: %f \n", host_memory.posy[num_nodes]);
  printf("z: %f \n", host_memory.posz[num_nodes]);
  printf("mass: %f \n", host_memory.mass[num_nodes]);
  printf("startd: %d \n", host_memory.start[num_nodes]);
  printf("stepd_num: %d \n", stepd_num);
  printf("bottom_num: %d \n", bottom_num);
  printf("radius: %f \n", radius);
  int k = num_nodes * 8;
  for(int i = 0; i < 8; i++) printf("child: %f \n", host_memory.child[k + i]);


}

