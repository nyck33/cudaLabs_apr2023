#include <stdio.h>

//nvcc -g -G -o vector-add 01-vector-add.cu -run

void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}

__global__
void addVectorsInto(float *result, float *a, float *b, int N)
{
  
  //start here on this thread
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = idx; i < N; i += stride)
  {
    result[i] = a[i] + b[i];
  }
}

void checkElementsAre(float target, float *array, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(array[i] != target)
    {
      printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
      exit(1);
    }
  }
  printf("SUCCESS! All values added correctly.\n");
}

int main()
{
  //in decimal 2097152
  //const int N = 2<<2000000000;
  const int N = 2<<200;

  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;

  //a = (float *)malloc(size);
  //b = (float *)malloc(size);
  //c = (float *)malloc(size);
  cudaMallocManaged(&a, size);
  initWith(3, a, N);
  cudaMallocManaged(&b, size);
  initWith(4, b, N);
  cudaMallocManaged(&c, size);
  initWith(0, c, N);

  
  //N divided by 1024 is 2048 iterations of data
  //4 blocks of 256 threads each = 1024 threads doing 2048 iterations, 
  //skip stride = gridDim.x * blockDim.x  
  //todo: deviceQuery example in cuda samples
  size_t threadsPerBlock = 256;
  size_t numberOfBlocks = 32;
  addVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);
  cudaDeviceSynchronize();

  for(int i=0; i< N; i++){
      printf("res[%d] = %f\n", i, c[i]);
  }

  checkElementsAre(7, c, N);
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  
}



