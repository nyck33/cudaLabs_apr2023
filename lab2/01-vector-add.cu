#include <stdio.h>

/*
 * Host function to initialize vector elements. This function
 * simply initializes each element to equal its index in the
 * vector.
 * 
 * 
 * In order to support the GPU's ability to perform as many parallel operations as possible, performance gains can often be had by choosing a grid size that has a number of blocks that is a multiple of the number of SMs on a given GPU.
 * Multiple of 14 SMs for gridDim
 * 
 * Additionally, SMs create, manage, schedule, and execute groupings of 32 threads from within a block called warps. A more in depth coverage of SMs and warps is beyond the scope of this course, however, it is important to know that performance gains can also be had by choosing a block size that has a number of threads that is a multiple of 32.
 Device ID: 0
Number of SMs: 14
Compute Capability Major: 7
Compute Capability Minor: 5
Warp Size: 32
 */

void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}

/*
 * Device kernel stores into `result` the sum of each
 * same-indexed value of `a` and `b`.
 */

__global__
void addVectorsInto(float *result, float *a, float *b, int N)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < N; i += stride)
  {
    result[i] = a[i] + b[i];
  }
}

/*
 * Host function to confirm values in `vector`. This function
 * assumes all values are the same `target` value.
 */

void checkElementsAre(float target, float *vector, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(vector[i] != target)
    {
      printf("FAIL: vector[%d] - %0.0f does not equal %0.0f\n", i, vector[i], target);
      exit(1);
    }
  }
  printf("Success! All values calculated correctly.\n");
}

int main()
{
  const int N = 2<<24;
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;

  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c, size);

  initWith(3, a, N);
  initWith(4, b, N);
  initWith(0, c, N);

  //size_t threadsPerBlock;
  //size_t numberOfBlocks;

  /*
   * nsys should register performance changes when execution configuration
   * is updated.
   * threadsPerBlock: 1,1024, 256
   * dim3 (16, 16, 1)
   * numblocks ((N/threadsperBlock) + 1, ...)
   * numBlocks:1, 1, 4
   */

  //threadsPerBlock = 256;
  //numberOfBlocks = 4;
  //dim3 threadsPerBlock = (16, 16,1);
  //dim3 numBlocks = ((N/threadsPerBlock.x)+1, (N/threadsPerBlock.y)+1, 1);
  int threadsPerBlock = 32 * 8;
  //for cloud device with 40 SMs
  int numBlocks = 40 * 10;
  //int numBlocks = 14 * 18;
  cudaError_t addVectorsErr;
  cudaError_t asyncErr;

  addVectorsInto<<<numBlocks, threadsPerBlock>>>(c, a, b, N);

  addVectorsErr = cudaGetLastError();
  if(addVectorsErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addVectorsErr));

  asyncErr = cudaDeviceSynchronize();
  if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

  checkElementsAre(7, c, N);

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}
