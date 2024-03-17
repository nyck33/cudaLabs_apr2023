__global__
void deviceKernel(int *a, int N)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < N; i += stride)
  {
    a[i] = 1;
  }
}

void hostFunction(int *a, int N)
{
  for (int i = 0; i < N; ++i)
  {
    a[i] = 1;
  }
}

int main()
{

  int N = 2<<24;
  size_t size = N * sizeof(int);
  int *a;
  cudaMallocManaged(&a, size);
  int numBlocks = 4;
  int threadsPerBlock = 256;
  //cudaError_t deviceErr, asyncErr;

  deviceKernel<<<numBlocks, threadsPerBlock>>>(a, N);
  cudaDeviceSynchronize();
  //deviceErr = cudaGetLastError();
  /*
  if(deviceErr != cudaSuccess) {
    printf("error: %s\n", cudaGetErrorString(deviceErr));

  }

  asyncErr = cudaDeviceSynchronize();
  if(asyncErr != cudaSuccess){
    printf("Error: %s\n", cudaGetErrorString(asyncErr));

  } 
  */
  /*
   *CPU first 
   Conduct experiments to learn more about the behavior of
   * `cudaMallocManaged`.
   *
   * What happens when unified memory is accessed only by the GPU?
   * What happens when unified memory is accessed only by the CPU?
   * What happens when unified memory is accessed first by the GPU then the CPU?
   * What happens when unified memory is accessed first by the CPU then the GPU?
   *
   * Hypothesize about UM behavior, page faulting specificially, before each
   * experiment, and then verify by running `nsys`.
   GPU first

   */
  hostFunction(a, N);

  cudaFree(a);
}
/*
host first, device second
CUDA Memory Operation Statistics (by size in KiB):

   Total     Operations  Average  Minimum  Maximum               Operation            
 ----------  ----------  -------  -------  --------  ---------------------------------
 131072.000         768  170.667    4.000  1016.000  [CUDA Unified Memory memcpy HtoD]

host second, device first
CUDA Memory Operation Statistics (by time):

 Time(%)  Total Time (ns)  Operations  Average  Minimum  Maximum              Operation            
 -------  ---------------  ----------  -------  -------  -------  ---------------------------------
   100.0         21215246         768  27624.0     1599   162494  [CUDA Unified Memory memcpy DtoH]

*/