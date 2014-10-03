
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
  float *mass_h, *posx_h, *posy_h, *posz_h, *velx_h, *vely_h, *velz_h;
  float *maxxl, *maxyl, *maxzl;
  float *minxl, *minyl, *minzl;

  //int *errl, *sortl, *childl, *countl, *startl;
  //float *massl;
  //float *velxl, *velyl, *velzl;
  //float *accxl, *accyl, *acczl;
  register double rsc, vsc, r, v, x, y, z, sq, scale;
  int num_bodies = 100000;
  int blocks = 4; // TODO Supposed to be set to multiprocecsor count

  int num_nodes = num_bodies * 2;
  if (num_nodes < 1024*blocks) num_nodes = 1024*blocks;
  while ((num_nodes & (WARPSIZE - 1)) != 0) num_nodes++;
  num_nodes--;
  // Allocate device memory // TODO Can change all host arrays to size num_cells. Using num_boides for debugging purpuses.
  mass_h = (float *)malloc(sizeof(float) * (num_nodes + 1));
  if (mass_h == NULL) {fprintf(stderr, "cannot allocate mass\n");  exit(-1);}
  posx_h = (float *)malloc(sizeof(float) * (num_nodes + 1)); // TODO Can change to number of bodies
  if (posx_h == NULL) {fprintf(stderr, "cannot allocate posx\n");  exit(-1);}
  posy_h = (float *)malloc(sizeof(float) * (num_nodes + 1)); // TODO Can change to number of bodies
  if (posy_h == NULL) {fprintf(stderr, "cannot allocate posy\n");  exit(-1);}
  posz_h = (float *)malloc(sizeof(float) * (num_nodes + 1));
  if (posz_h == NULL) {fprintf(stderr, "cannot allocate posz\n");  exit(-1);}
  //velx_h = (float *)malloc(sizeof(float) * num_bodies);
  //if (velx_h == NULL) {fprintf(stderr, "cannot allocate velx\n");  exit(-1);}
  //vely_h = (float *)malloc(sizeof(float) * num_bodies);
  //if (vely_h == NULL) {fprintf(stderr, "cannot allocate vely\n");  exit(-1);}
  //velz_h = (float *)malloc(sizeof(float) * num_bodies);
  //if (velz_h == NULL) {fprintf(stderr, "cannot allocate velz\n");  exit(-1);}

  for (int i = 0; i < num_bodies; i ++) {
    posx_h[i] = i;
    posy_h[i] = i;
  }
  for (int i = num_bodies; i < num_nodes; i++) {
    posx_h[i] = 0;
    posy_h[i] = 0;
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

  cl_mem posxl, posyl, minx_d, maxx_d, miny_d, maxy_d, minz_d, maxz_d, blocked;
  cl_int d_num_nodes;

  cl_int err = CL_SUCCESS;

  posxl = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
      sizeof(float) * (num_nodes + 1), NULL, &err);
  CHK_ERR(err);
  posyl = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
      sizeof(float)*(num_nodes + 1), NULL, &err);
  CHK_ERR(err);

  err = clEnqueueWriteBuffer(cv.commands, posxl, true, 0, sizeof(float)*(num_nodes + 1),
			     posx_h, 0, NULL, NULL);
  CHK_ERR(err);
  err = clEnqueueWriteBuffer(cv.commands, posyl, true, 0, sizeof(float)*(num_nodes + 1),
			     posy_h, 0, NULL, NULL);
  CHK_ERR(err);

  /* Set local work size and global work sizes */
  size_t local_work_size[1] = {256};
  size_t global_work_size[1] = {1024};
  size_t num_work_groups = 4;
  
  // Used for global reduction in finding min and max of bounding box
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
  blocked = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
      1*sizeof(int), NULL, &err);
  CHK_ERR(err);
  int blocked_num[1];
  blocked_num[0] = 0;
  err = clEnqueueWriteBuffer(cv.commands, blocked, true, 0, sizeof(int),
			     blocked_num, 0, NULL, NULL);

  /* Set the Kernel Arguements */
  err = clSetKernelArg(bound_box, 0, sizeof(cl_mem), &posxl);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 1, sizeof(cl_mem), &posyl);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 2, local_work_size[0]*sizeof(float), NULL);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 3, local_work_size[0]*sizeof(float), NULL);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 4, local_work_size[0]*sizeof(float), NULL);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 5, local_work_size[0]*sizeof(float), NULL);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 6, sizeof(cl_mem), &minx_d);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 7, sizeof(cl_mem), &maxx_d);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 8, sizeof(cl_mem), &miny_d);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 9, sizeof(cl_mem), &maxx_d);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 10, sizeof(int*), &blocked);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 11, sizeof(int), &num_bodies);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 12, sizeof(int), &num_nodes);
  CHK_ERR(err);

  err = clEnqueueNDRangeKernel(cv.commands, bound_box, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
  CHK_ERR(err);

  err = clEnqueueReadBuffer(cv.commands, posxl, true, 0, sizeof(float)*(num_nodes + 1), posx_h, 0,
    NULL, NULL);
  err = clEnqueueReadBuffer(cv.commands, posyl, true, 0, sizeof(float)*(num_nodes + 1), posy_h, 0,
    NULL, NULL);
  CHK_ERR(err);
  printf("x: %f", posx_h[num_nodes]);
  printf("x: %f \n", posy_h[num_nodes]);
  clReleaseMemObject(posxl);
  clReleaseMemObject(posyl);

  
}

