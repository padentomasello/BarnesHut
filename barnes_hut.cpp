
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cmath>
#include <unistd.h>

#include "clhelp.h"

int main (int argc, char *argv[])
{
  int num_bodies = 10;
  int num_cells = 19;
  int* h_xcords = new int[num_bodies];
  int* h_ycords = new int[num_bodies];
  int* h_xy_max = new int[4];
  for (int i = 0; i < num_bodies; i ++) {
    h_xcords[i] = i;
    h_ycords[i] = i;
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

  cl_mem d_xcords, d_ycords, d_xy_max, global_minx, global_maxx, global_miny, global_maxy, blocked;
  cl_int d_num_cells;

  cl_int err = CL_SUCCESS;

  d_xcords = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
      sizeof(int) * (num_bodies), NULL, &err);
  CHK_ERR(err);
  d_ycords = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
      sizeof(int)*(num_bodies), NULL, &err);
  d_xy_max = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
      sizeof(int)*4, NULL, &err);
  CHK_ERR(err);

  err = clEnqueueWriteBuffer(cv.commands, d_xcords, true, 0, sizeof(int)*num_bodies,
			     h_xcords, 0, NULL, NULL);
  CHK_ERR(err);
  err = clEnqueueWriteBuffer(cv.commands, d_ycords, true, 0, sizeof(int)*num_bodies,
			     h_ycords, 0, NULL, NULL);
  CHK_ERR(err);

  /* Set local work size and global work sizes */
  size_t local_work_size[1] = {256};
  size_t global_work_size[1] = {512};
  size_t num_work_groups = 2;

  global_minx = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
      sizeof(int)*num_work_groups, NULL, &err);
  global_maxx = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
      sizeof(int)*num_work_groups, NULL, &err);
  global_miny = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
      sizeof(int)*num_work_groups, NULL, &err);
  global_maxy = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
      sizeof(int)*num_work_groups, NULL, &err);
  blocked = clCreateBuffer(cv.context, CL_MEM_READ_WRITE,
      1*sizeof(int), NULL, &err);
  CHK_ERR(err);


  /* Set the Kernel Arguements */
  err = clSetKernelArg(bound_box, 0, sizeof(cl_mem), &d_xcords);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 1, sizeof(cl_mem), &d_ycords);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 2, sizeof(cl_mem), &d_xy_max);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 3, local_work_size[0]*sizeof(int), NULL);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 4, local_work_size[0]*sizeof(int), NULL);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 5, local_work_size[0]*sizeof(int), NULL);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 6, local_work_size[0]*sizeof(int), NULL);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 7, num_work_groups*sizeof(int), &global_minx);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 8, num_work_groups*sizeof(int), &global_maxx);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 9, num_work_groups*sizeof(int), &global_minx);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 10, num_work_groups*sizeof(int), &global_maxy);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 11, 1*sizeof(int*), &blocked);
  CHK_ERR(err);
  err = clSetKernelArg(bound_box, 12, sizeof(int), &num_bodies);
  CHK_ERR(err);

  err = clEnqueueNDRangeKernel(cv.commands, bound_box, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
  CHK_ERR(err);

  err = clEnqueueReadBuffer(cv.commands, d_xy_max, true, 0, sizeof(int)*4, h_xy_max, 0,
    NULL, NULL);
  CHK_ERR(err);
  printf("x: %d", h_xy_max[0]);
  printf("y: %d", h_xy_max[1]);
  printf("x: %d", h_xy_max[2]);
  printf("y: %d", h_xy_max[3]);
}

