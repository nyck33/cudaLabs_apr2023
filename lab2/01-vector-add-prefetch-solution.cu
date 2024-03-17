#include <stdio.h>

__global__
void initWith(float num, float *a, int N)
{

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < N; i += stride)
  {
    a[i] = num;
  }
}

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
  int deviceId;
  int numberOfSMs;

  cudaGetDevice(&deviceId);
  cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
  printf("num SMs = %d\n", numberOfSMs);
  const int N = 2<<24;
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;

  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c, size);

  cudaMemPrefetchAsync(a, size, deviceId);
  cudaMemPrefetchAsync(b, size, deviceId);
  cudaMemPrefetchAsync(c, size, deviceId);

  size_t threadsPerBlock;
  size_t numberOfBlocks;

  threadsPerBlock = 256;
  numberOfBlocks = 32 * numberOfSMs;

  cudaError_t addVectorsErr;
  cudaError_t asyncErr;

  initWith<<<numberOfBlocks, threadsPerBlock>>>(3, a, N);
  initWith<<<numberOfBlocks, threadsPerBlock>>>(4, b, N);
  initWith<<<numberOfBlocks, threadsPerBlock>>>(0, c, N);

  addVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);

  addVectorsErr = cudaGetLastError();
  if(addVectorsErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addVectorsErr));

  asyncErr = cudaDeviceSynchronize();
  if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

  checkElementsAre(7, c, N);

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}

/*
1 prefetch
CUDA Memory Operation Statistics (by time):

 Time(%)  Total Time (ns)  Operations  Average  Minimum  Maximum              Operation            
 -------  ---------------  ----------  -------  -------  -------  ---------------------------------
   100.0         21203394         768  27608.6     1599   160828  [CUDA Unified Memory memcpy DtoH]

CUDA Kernel Statistics:

 Time(%)  Total Time (ns)  Instances   Average    Minimum  Maximum                      Name                    
 -------  ---------------  ---------  ----------  -------  --------  -------------------------------------------
    96.0         40732060          3  13577353.3   619599  20614807  initWith(float, float*, int)               
     4.0          1710960          1   1710960.0  1710960   1710960  addVectorsInto(float*, float*, float*, int)

2 prefetches
CUDA Kernel Statistics:

 Time(%)  Total Time (ns)  Instances   Average   Minimum  Maximum                      Name                    
 -------  ---------------  ---------  ---------  -------  --------  -------------------------------------------
    92.9         22514595          3  7504865.0   620718  21269383  initWith(float, float*, int)               
     7.1          1712015          1  1712015.0  1712015   1712015  addVectorsInto(float*, float*, float*, int)
                 40732060


CUDA Memory Operation Statistics (by time):

 Time(%)  Total Time (ns)  Operations  Average  Minimum  Maximum              Operation            
 -------  ---------------  ----------  -------  -------  -------  ---------------------------------
   100.0         21200406         768  27604.7     1599   160123  [CUDA Unified Memory memcpy DtoH]
                 21203394

3 prefetches
CUDA Kernel Statistics:

 Time(%)  Total Time (ns)  Instances   Average   Minimum  Maximum                     Name                    
 -------  ---------------  ---------  ---------  -------  -------  -------------------------------------------
    52.2          1867563          3   622521.0   618639   624558  initWith(float, float*, int)               
    47.8          1711696          1  1711696.0  1711696  1711696  addVectorsInto(float*, float*, float*, int)
                 40732060


CUDA Memory Operation Statistics (by time):

 Time(%)  Total Time (ns)  Operations  Average  Minimum  Maximum              Operation            
 -------  ---------------  ----------  -------  -------  -------  ---------------------------------
   100.0         21201498         768  27606.1     1599   160028  [CUDA Unified Memory memcpy DtoH]
*/