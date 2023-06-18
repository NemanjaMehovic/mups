#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>

double cpu_time(void)
{
  double value;

  value = (double)clock() / (double)CLOCKS_PER_SEC;

  return value;
}

__constant__ int cudaN;

__device__ void warp_reduce(volatile int *sdata, const unsigned int thread_id)
{
    sdata[thread_id] += sdata[thread_id + 32];
    sdata[thread_id] += sdata[thread_id + 16];
    sdata[thread_id] += sdata[thread_id + 8];
    sdata[thread_id] += sdata[thread_id + 4];
    sdata[thread_id] += sdata[thread_id + 2];
    sdata[thread_id] += sdata[thread_id + 1];
}

__global__ void countPrime(int* retData)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int prime = 0;
  extern __shared__ int data[];

  data[threadIdx.x] = 0;
  int val = gridDim.x * blockDim.x;

  for(int i = idx + 2; i <= cudaN; i += val)
  {
    prime = 1;
    for (int j = 2; j < i; j++)
    {
      if ((i % j) == 0)
      {
        prime = 0;
        break;
      }
    }
    data[threadIdx.x] = data[threadIdx.x] + prime;
  }

  __syncthreads();

  for (unsigned int s = blockDim.x/2; s > 32; s >>= 1)
  {
      if (threadIdx.x < s) {
          data[threadIdx.x] += data[threadIdx.x + s];
      }
      __syncthreads();
  }

  if(threadIdx.x < 32)
    warp_reduce(data, threadIdx.x);

  __syncthreads();

  if(threadIdx.x == 0)
    retData[blockIdx.x] = data[0];
}

int prime_number(int n)
{

  int total = 0;
  int* retData;
  int* data;

  const int numOfThreadsPerBlock = 1024;
  int numOfBLocks = ceil(n / (float)numOfThreadsPerBlock);

  data = (int*)malloc(numOfBLocks * sizeof(int));
  cudaMalloc(&retData, numOfBLocks * sizeof(int));
  cudaMemcpyToSymbol(cudaN, &n, sizeof(n));

  countPrime<<<numOfBLocks, numOfThreadsPerBlock, numOfThreadsPerBlock * sizeof(int)>>>(retData);

  cudaMemcpy(data, retData, numOfBLocks * sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(retData);

  for(int i = 0; i < numOfBLocks; i++)
    total += data[i];

  free(data);
  
  return total;
}

void timestamp(void)
{
#define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  size_t len;
  time_t now;

  now = time(NULL);
  tm = localtime(&now);

  len = strftime(time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm);

  printf("%s\n", time_buffer);

  return;
#undef TIME_SIZE
}

void test(int n_lo, int n_hi, int n_factor);

void par(int argc, char *argv[])
{
  int n_factor;
  int n_hi;
  int n_lo;

  timestamp();
  printf("\n");
  printf("PRIME TEST\n");

  if (argc != 4)
  {
    n_lo = 1;
    n_hi = 131072;
    n_factor = 2;
  }
  else
  {
    n_lo = atoi(argv[1]);
    n_hi = atoi(argv[2]);
    n_factor = atoi(argv[3]);
  }

  test(n_lo, n_hi, n_factor);

  printf("\n");
  printf("PRIME_TEST\n");
  printf("  Normal end of execution.\n");
  printf("\n");
  timestamp();
}

int nPar,primesPar;

void test(int n_lo, int n_hi, int n_factor)
{
  int i;
  int n;
  int primes;
  double ctime;

  printf("\n");
  printf("  Call PRIME_NUMBER to count the primes from 1 to N.\n");
  printf("\n");
  printf("         N        Pi          Time\n");
  printf("\n");

  n = n_lo;

  while (n <= n_hi)
  {
    ctime = cpu_time();

    primes = prime_number(n);

    ctime = cpu_time() - ctime;

    printf("  %8d  %8d  %14f\n", n, primes, ctime);
    n = n * n_factor;
  }
  
  nPar = n;
  primesPar = primes;

  return;
}

int prime_numberSeq(int n)
{
  int i;
  int j;
  int prime;
  int total;

  total = 0;

  for (i = 2; i <= n; i++)
  {
    prime = 1;
    for (j = 2; j < i; j++)
    {
      if ((i % j) == 0)
      {
        prime = 0;
        break;
      }
    }
    total = total + prime;
  }
  return total;
}

//seq code
void testSeq(int n_lo, int n_hi, int n_factor);

void seq(int argc, char *argv[])
{
  int n_factor;
  int n_hi;
  int n_lo;

  timestamp();
  printf("\n");
  printf("PRIME TEST\n");

  if (argc != 4)
  {
    n_lo = 1;
    n_hi = 131072;
    n_factor = 2;
  }
  else
  {
    n_lo = atoi(argv[1]);
    n_hi = atoi(argv[2]);
    n_factor = atoi(argv[3]);
  }

  testSeq(n_lo, n_hi, n_factor);

  printf("\n");
  printf("PRIME_TEST\n");
  printf("  Normal end of execution.\n");
  printf("\n");
  timestamp();
}

int nSeq,primesSeq;

void testSeq(int n_lo, int n_hi, int n_factor)
{
  int i;
  int n;
  int primes;
  double ctime;

  printf("\n");
  printf("  Call PRIME_NUMBER to count the primes from 1 to N.\n");
  printf("\n");
  printf("         N        Pi          Time\n");
  printf("\n");

  n = n_lo;

  while (n <= n_hi)
  {
    ctime = cpu_time();

    primes = prime_numberSeq(n);

    ctime = cpu_time() - ctime;

    printf("  %8d  %8d  %14f\n", n, primes, ctime);
    n = n * n_factor;
  }

  nSeq = n;
  primesSeq = primes;

  return;
}

int main(int argc, char *argv[])
{
  double timeS, timeE, parTime;
  printf("SEQUENTIAL EXECUTION\n");
  timeS = omp_get_wtime();
  seq(argc, argv);
  timeE = omp_get_wtime();
  parTime = timeE - timeS;
  printf("Execution time for sequential %f s \n\n", parTime);
  printf("PARALLEL EXECUTION\n");
  timeS = omp_get_wtime();
  par(argc, argv);
  timeE = omp_get_wtime();
  parTime = timeE - timeS;
  printf("Execution time for parallel %f s \n\n", parTime);
  if(nSeq != nPar || primesPar != primesSeq)
    printf("Test FAILED \n\n");
  else
    printf("Test PASSED \n\n");

  return 0;
}
