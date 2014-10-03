
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cmath>
#include <unistd.h>

#include "clhelp.h"

#define WARPSIZE 32

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
  float *mass_h, *posx_h, *posy_h, *posz_h, *velx_h, *vely_h, *velz_h, *child_h;
  int *start_h;
  float *maxxl, *maxyl, *maxzl;
  float *minxl, *minyl, *minzl;

  //int *errl, *sortl, *childl, *countl, *startl;
  //float *massl;
  //float *velxl, *velyl, *velzl;
  //float *accxl, *accyl, *acczl;
  register double rsc, vsc, r, v, x, y, z, sq, scale;
  int num_bodies = 10;
  int blocks = 4; // TODO Supposed to be set to multiprocecsor count

  int num_nodes = num_bodies * 2;
  if (num_nodes < 1024*blocks) num_nodes = 1024*blocks;
  while ((num_nodes & (WARPSIZE - 1)) != 0) num_nodes++;
  num_nodes--;
  // Allocate device memory // TODO Can change all host arrays to size num_cells. Using num_boides for debugging purpuses.
  {
  mass_h = (float *)malloc(sizeof(float) * (num_nodes + 1));
  if (mass_h == NULL) {fprintf(stderr, "cannot allocate mass\n");  exit(-1);}
  start_h = (int *)malloc(sizeof(float) * (num_nodes + 1));
  if (start_h == NULL) {fprintf(stderr, "cannot allocate mass\n");  exit(-1);}
  posx_h = (float *)malloc(sizeof(float) * (num_nodes + 1)); // TODO Can change to number of bodies
  if (posx_h == NULL) {fprintf(stderr, "cannot allocate posx\n");  exit(-1);}
  posy_h = (float *)malloc(sizeof(float) * (num_nodes + 1)); // TODO Can change to number of bodies
  if (posy_h == NULL) {fprintf(stderr, "cannot allocate posy\n");  exit(-1);}
  posz_h = (float *)malloc(sizeof(float) * (num_nodes + 1));
  if (posz_h == NULL) {fprintf(stderr, "cannot allocate posz\n");  exit(-1);}
  velx_h = (float *)malloc(sizeof(float) * num_bodies);
  if (velx_h == NULL) {fprintf(stderr, "cannot allocate velx\n");  exit(-1);}
  vely_h = (float *)malloc(sizeof(float) * num_bodies);
  if (vely_h == NULL) {fprintf(stderr, "cannot allocate vely\n");  exit(-1);}
  velz_h = (float *)malloc(sizeof(float) * num_bodies);
  if (velz_h == NULL) {fprintf(stderr, "cannot allocate velz\n");  exit(-1);}
  child_h = (float *)malloc(sizeof(float) * 8*(num_nodes + 1));
  if (velz_h == NULL) {fprintf(stderr, "cannot allocate velz\n");  exit(-1);}
  }

  for (int i = 0; i < num_bodies; i ++) {
    posx_h[i] = i;
    posy_h[i] = i;
    posz_h[i] = i;
  }

  std::string bounding_box_kernel_str;
  std::string bounding_box_name_str = std::string("bound_box");
  std::string bounding_box_kernel_file = std::string("bound_box.cl");
  cl_vars_t cv;
  cl_kernel bound_box;

  readFile(bounding_box_kernel_file, bounding_box_kernel_str);

  initialize_ocl(cv);
  compile_ocl_program(bound_box, cv, bounding_box_kernel_str.c_str(),
      bounding_box_name_str.c_str());

  cl_mem posxl, posyl, poszl, minx_d, maxx_d, miny_d, maxy_d, minz_d, maxz_d, blocked, childl,
         velxl, velyl, velzl, accxl, accyl, acczl, sortl, massl, countl, startl;
  cl_int d_num_nodes;

  cl_int err = CL_SUCCESS;

  // Create Buffers  NOTE* These do need to be (num_nodes + 1)
  {
  posxl = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
      sizeof(float) * (num_nodes + 1), NULL, &err);
  CHK_ERR(err);
  posyl = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
      sizeof(float)*(num_nodes + 1), NULL, &err);
  CHK_ERR(err);
  poszl = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
      sizeof(float)*(num_nodes + 1), NULL, &err);
  posxl = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
      sizeof(float) * (num_nodes + 1), NULL, &err);
  CHK_ERR(err);
  posyl = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
      sizeof(float)*(num_nodes + 1), NULL, &err);
  CHK_ERR(err);
  poszl = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
      sizeof(float)*(num_nodes + 1), NULL, &err);
  CHK_ERR(err);
  massl = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
      sizeof(float)*(num_nodes + 1), NULL, &err);
  CHK_ERR(err);
  countl = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
      sizeof(float)*(num_nodes + 1), NULL, &err);
  CHK_ERR(err);
  startl = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
      sizeof(int)*(num_nodes + 1), NULL, &err);
  CHK_ERR(err);
  childl = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
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
    velxl = clCreateSubBuffer(childl, CL_MEM_READ_WRITE, 
        CL_BUFFER_CREATE_TYPE_REGION, &velxl_region, &err);
    velyl = clCreateSubBuffer(childl, CL_MEM_READ_WRITE, 
        CL_BUFFER_CREATE_TYPE_REGION, &velyl_region, &err);
    velzl = clCreateSubBuffer(childl, CL_MEM_READ_WRITE, 
        CL_BUFFER_CREATE_TYPE_REGION, &velzl_region, &err);
    accxl = clCreateSubBuffer(childl, CL_MEM_READ_WRITE, 
        CL_BUFFER_CREATE_TYPE_REGION, &accxl_region, &err);
    accyl = clCreateSubBuffer(childl, CL_MEM_READ_WRITE, 
        CL_BUFFER_CREATE_TYPE_REGION, &accyl_region, &err);
    acczl = clCreateSubBuffer(childl, CL_MEM_READ_WRITE, 
        CL_BUFFER_CREATE_TYPE_REGION, &acczl_region, &err);
    sortl = clCreateSubBuffer(childl, CL_MEM_READ_WRITE, 
        CL_BUFFER_CREATE_TYPE_REGION, &sortl_region, &err);
  }

  // Write to buffers 
  {
  err = clEnqueueWriteBuffer(cv.commands, posxl, true, 0, sizeof(float)*(num_nodes + 1),
			     posx_h, 0, NULL, NULL);
  CHK_ERR(err);
  err = clEnqueueWriteBuffer(cv.commands, posyl, true, 0, sizeof(float)*(num_nodes + 1),
			     posy_h, 0, NULL, NULL);
  CHK_ERR(err);

  err = clEnqueueWriteBuffer(cv.commands, poszl, true, 0, sizeof(float)*(num_nodes + 1),
           posz_h, 0, NULL, NULL);
  CHK_ERR(err);
  }

  /* Set local work size and global work sizes */
  size_t local_work_size[1] = {256};
  size_t global_work_size[1] = {1024};
  size_t num_work_groups = 4;

  // Used for global reduction in finding min and max of bounding box
  {
  minx_d = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
      sizeof(float)*num_work_groups, NULL, &err);
  maxx_d = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
      sizeof(float)*num_work_groups, NULL, &err);
  miny_d = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
      sizeof(float)*num_work_groups, NULL, &err);
  maxy_d = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
      sizeof(float)*num_work_groups, NULL, &err);
  minz_d = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
      sizeof(float)*num_work_groups, NULL, &err);
  maxz_d = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
      sizeof(float)*num_work_groups, NULL, &err);
  maxz_d = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
      sizeof(float)*num_work_groups, NULL, &err);
  blocked = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
      1*sizeof(int), NULL, &err);
  }
  CHK_ERR(err);
  int blocked_num[1]; // TODO Would it be more efficient to use an InitializationKernel? See Cuda implementation around line 82
  blocked_num[0] = 0;
  err = clEnqueueWriteBuffer(cv.commands, blocked, true, 0, sizeof(int),
			     blocked_num, 0, NULL, NULL);

  // Set the Kernel Arguements for bounding box
  {
  err = clSetKernelArg(bound_box, 0, sizeof(cl_mem), &posxl);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 1, sizeof(cl_mem), &posyl);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 2, sizeof(cl_mem), &poszl);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 3, sizeof(cl_mem), &childl);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 4, sizeof(cl_mem), &massl);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 5, sizeof(cl_mem), &startl);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 6, local_work_size[0]*sizeof(float), NULL);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 7, local_work_size[0]*sizeof(float), NULL);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 8, local_work_size[0]*sizeof(float), NULL);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 9, local_work_size[0]*sizeof(float), NULL);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 10, local_work_size[0]*sizeof(float), NULL);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 11, local_work_size[0]*sizeof(float), NULL);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 12, sizeof(cl_mem), &minx_d);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 13, sizeof(cl_mem), &maxx_d);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 14, sizeof(cl_mem), &miny_d);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 15, sizeof(cl_mem), &maxy_d);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 16, sizeof(cl_mem), &minz_d);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 17, sizeof(cl_mem), &maxz_d);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 18, sizeof(int*), &blocked);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 19, sizeof(int), &num_bodies);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 20, sizeof(int), &num_nodes);
  CHK_ERR(err);
  }

  err = clEnqueueNDRangeKernel(cv.commands, bound_box, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
  CHK_ERR(err);

  err = clEnqueueReadBuffer(cv.commands, posxl, true, 0, sizeof(float)*(num_nodes + 1), posx_h, 0,
    NULL, NULL);
  err = clEnqueueReadBuffer(cv.commands, posyl, true, 0, sizeof(float)*(num_nodes + 1), posy_h, 0,
    NULL, NULL);
  err = clEnqueueReadBuffer(cv.commands, poszl, true, 0, sizeof(float)*(num_nodes + 1), posz_h, 0,
    NULL, NULL);
  err = clEnqueueReadBuffer(cv.commands, childl, true, 0, sizeof(float)*8*(num_nodes + 1), child_h, 0,
    NULL, NULL);
  err = clEnqueueReadBuffer(cv.commands, massl, true, 0, sizeof(float)*(num_nodes + 1), mass_h, 0,
    NULL, NULL);
  err = clEnqueueReadBuffer(cv.commands, startl, true, 0, sizeof(int)*(num_nodes + 1), start_h, 0,
    NULL, NULL);
  CHK_ERR(err);
  printf("x: %f", posx_h[num_nodes]);
  printf("y: %f \n", posy_h[num_nodes]);
  printf("z: %f \n", posz_h[num_nodes]);
  printf("mass: %f \n", mass_h[num_nodes]);
  printf("startd: %d \n", start_h[num_nodes]);
  int k = num_nodes * 8;
  for(int i = 0; i < 8; i++) printf("child: %f \n", child_h[k + i]);
  clReleaseMemObject(posxl);
  clReleaseMemObject(posyl);
  clReleaseMemObject(poszl);


}

