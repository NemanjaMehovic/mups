#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

int i4_ceiling(double x)
{
  int value = (int)x;
  if (value < x)
    value = value + 1;
  return value;
}

int i4_min(int i1, int i2)
{
  int value;
  if (i1 < i2)
    value = i1;
  else
    value = i2;
  return value;
}

double potential(double a, double b, double c, double x, double y, double z)
{
  return 2.0 * (pow(x / a / a, 2) + pow(y / b / b, 2) + pow(z / c / c, 2)) + 1.0 / a / a + 1.0 / b / b + 1.0 / c / c;
}

double r8_uniform_01(int *seed)
{
  int k;
  double r;

  k = *seed / 127773;

  *seed = 16807 * (*seed - k * 127773) - k * 2836;

  if (*seed < 0)
  {
    *seed = *seed + 2147483647;
  }
  r = (double)(*seed) * 4.656612875E-10;

  return r;
}

__device__ double potentialDevice(double a, double b, double c, double x, double y, double z)
{
  return 2.0 * (pow(x / a / a, 2) + pow(y / b / b, 2) + pow(z / c / c, 2)) + 1.0 / a / a + 1.0 / b / b + 1.0 / c / c;
}

__device__ double r8_uniform_01Device(int *seed)
{
  int k;
  double r;

  k = *seed / 127773;

  *seed = 16807 * (*seed - k * 127773) - k * 2836;

  if (*seed < 0)
  {
    *seed = *seed + 2147483647;
  }
  r = (double)(*seed) * 4.656612875E-10;

  return r;
}

__device__ void warp_reduce(volatile double *sdata, const unsigned int thread_id)
{
    sdata[thread_id] += sdata[thread_id + 32];
    sdata[thread_id] += sdata[thread_id + 16];
    sdata[thread_id] += sdata[thread_id + 8];
    sdata[thread_id] += sdata[thread_id + 4];
    sdata[thread_id] += sdata[thread_id + 2];
    sdata[thread_id] += sdata[thread_id + 1];
}

__device__ void warp_reduce(volatile int *sdata, const unsigned int thread_id)
{
    sdata[thread_id] += sdata[thread_id + 32];
    sdata[thread_id] += sdata[thread_id + 16];
    sdata[thread_id] += sdata[thread_id + 8];
    sdata[thread_id] += sdata[thread_id + 4];
    sdata[thread_id] += sdata[thread_id + 2];
    sdata[thread_id] += sdata[thread_id + 1];
}

__constant__ double xConst, yConst, zConst, aConst, bConst, cConst, hConst, stepszConst;
__constant__ int NConst;

__global__ void gpuExecute(double *wtGlobal, int *stepsGlobal, int *seedTmp)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int val = gridDim.x * blockDim.x;
  double x1, x2, x3, w, we, chk, us, ut , dx, dy, dz, vs, vh;
  __shared__ double wt[1024];
  __shared__ int steps[1024];
  int seed = seedTmp[idx];
  wt[threadIdx.x] = 0.0;
  steps[threadIdx.x] = 0;

  for (int trial = idx; trial < NConst; trial += val)
  {
    //printf("Thread %d\n",idx);
    //printf("Nconst %d\n",Nconst);
    x1 = xConst;
    x2 = yConst;
    x3 = zConst;
    w = 1.0;
    chk = 0.0;
    while (chk < 1.0)
    {
      ut = r8_uniform_01Device(&seed);
      if (ut < 1.0 / 3.0)
      {
        us = r8_uniform_01Device(&seed) - 0.5;
        if (us < 0.0)
          dx = -stepszConst;
        else
          dx = stepszConst;
      }
      else
        dx = 0.0;

      ut = r8_uniform_01Device(&seed);
      if (ut < 1.0 / 3.0)
      {
        us = r8_uniform_01Device(&seed) - 0.5;
        if (us < 0.0)
          dy = -stepszConst;
        else
          dy = stepszConst;
      }
      else
        dy = 0.0;

      ut = r8_uniform_01Device(&seed);
      if (ut < 1.0 / 3.0)
      {
        us = r8_uniform_01Device(&seed) - 0.5;
        if (us < 0.0)
          dz = -stepszConst;
        else
          dz = stepszConst;
      }
      else
        dz = 0.0;

      vs = potentialDevice(aConst, bConst, cConst, x1, x2, x3);
      x1 = x1 + dx;
      x2 = x2 + dy;
      x3 = x3 + dz;

      steps[threadIdx.x]++;

      vh = potentialDevice(aConst, bConst, cConst, x1, x2, x3);

      we = (1.0 - hConst * vs) * w;
      w = w - 0.5 * hConst * (vh * we + vs * w);

      chk = pow(x1 / aConst, 2) + pow(x2 / bConst, 2) + pow(x3 / cConst, 2);
    }
    wt[threadIdx.x] = wt[threadIdx.x] + w;
    //printf("Thread %d : w = %f steps = %d wt = %f\n", idx, w, steps[threadIdx.x], wt[threadIdx.x]);
  }

  __syncthreads();

  for (unsigned int s = blockDim.x/2; s > 32; s >>= 1)
  {
      if (threadIdx.x < s) {
          wt[threadIdx.x] += wt[threadIdx.x + s];
          steps[threadIdx.x] += steps[threadIdx.x + s];
      }
      __syncthreads();
  }

  if(threadIdx.x < 32)
  {
    warp_reduce(wt, threadIdx.x);
    warp_reduce(steps, threadIdx.x);
  }

  __syncthreads();

  if(threadIdx.x == 0)
  {
    wtGlobal[blockIdx.x] = wt[0];
    stepsGlobal[blockIdx.x] = steps[0];
  }

  seedTmp[idx] = seed;
}


void timestamp(void)
{
#define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  time_t now;

  now = time(NULL);
  tm = localtime(&now);

  strftime(time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm);

  printf("%s\n", time_buffer);

  return;
#undef TIME_SIZE
}

double errPar, errSeq;
// print na stdout upotrebiti u validaciji paralelnog resenja
void par(int arc, char **argv)
{
  double a = 3.0;
  double b = 2.0;
  double c = 1.0;
  double chk;
  int dim = 3;
  double err;
  double h = 0.001;
  int i;
  int j;
  int k;
  int n_inside;
  int ni;
  int nj;
  int nk;
  double stepsz;
  int seed = 123456789;
  int steps;
  int steps_ave;
  double x;
  double y;
  double w_exact;
  double wt;
  double z;

  int N = atoi(argv[1]);
  timestamp();

  printf("A = %f\n", a);
  printf("B = %f\n", b);
  printf("C = %f\n", c);
  printf("N = %d\n", N);
  printf("H = %6.4f\n", h);

  stepsz = sqrt((double)dim * h);

  if (a == i4_min(i4_min(a, b), c))
  {
    ni = 6;
    nj = 1 + i4_ceiling(b / a) * (ni - 1);
    nk = 1 + i4_ceiling(c / a) * (ni - 1);
  }
  else if (b == i4_min(i4_min(a, b), c))
  {
    nj = 6;
    ni = 1 + i4_ceiling(a / b) * (nj - 1);
    nk = 1 + i4_ceiling(c / b) * (nj - 1);
  }
  else
  {
    nk = 6;
    ni = 1 + i4_ceiling(a / c) * (nk - 1);
    nj = 1 + i4_ceiling(b / c) * (nk - 1);
  }

  err = 0.0;
  n_inside = 0;

  int *cudaSeeds;
  int flagFirstRun = 1;
  for (i = 1; i <= ni; i++)
  {
    x = ((double)(ni - i) * (-a) + (double)(i - 1) * a) / (double)(ni - 1);

    for (j = 1; j <= nj; j++)
    {
      y = ((double)(nj - j) * (-b) + (double)(j - 1) * b) / (double)(nj - 1);

      for (k = 1; k <= nk; k++)
      {
        z = ((double)(nk - k) * (-c) + (double)(k - 1) * c) / (double)(nk - 1);

        chk = pow(x / a, 2) + pow(y / b, 2) + pow(z / c, 2);

        if (1.0 < chk)
        {
          w_exact = 1.0;
          wt = 1.0;
          steps_ave = 0;
          // printf("  %7.4f  %7.4f  %7.4f  %10.4e  %10.4e  %10.4e  %8d\n",
          //        x, y, z, wt, w_exact, fabs(w_exact - wt), steps_ave);

          continue;
        }

        n_inside++;

        w_exact = exp(pow(x / a, 2) + pow(y / b, 2) + pow(z / c, 2) - 1.0);

        wt = 0.0;
        steps = 0;

        int numOfThreadsPerBlock = 1024;
        int numOfBlocks = ceil(N/1024.0f);
        double *wtRet, *wtTmp;
        int *stepsRet, *stepsTmp, *seedTmp;
        wtTmp = (double*)malloc(sizeof(double) * numOfBlocks);
        stepsTmp = (int*)malloc(sizeof(int) * numOfBlocks);
        if(flagFirstRun)
        {
          cudaSeeds = (int*)malloc(numOfBlocks*numOfThreadsPerBlock*sizeof(int));
          for(int tmpInc = 0; tmpInc < numOfBlocks*numOfThreadsPerBlock; tmpInc++)
          {
            cudaSeeds[tmpInc] = seed+tmpInc;
          }
          flagFirstRun = 0;
        }

        cudaMalloc(&wtRet, numOfBlocks * sizeof(double));
        cudaMalloc(&stepsRet, numOfBlocks * sizeof(int));
        cudaMalloc(&seedTmp, numOfBlocks*numOfThreadsPerBlock*sizeof(int));

        cudaMemcpyToSymbol(xConst, &x, sizeof(x));
        cudaMemcpyToSymbol(yConst, &y, sizeof(y));
        cudaMemcpyToSymbol(zConst, &z, sizeof(z));
        cudaMemcpyToSymbol(aConst, &a, sizeof(a));
        cudaMemcpyToSymbol(bConst, &b, sizeof(b));
        cudaMemcpyToSymbol(cConst, &c, sizeof(c));
        cudaMemcpyToSymbol(hConst, &h, sizeof(h));
        cudaMemcpyToSymbol(NConst, &N, sizeof(N));
        cudaMemcpyToSymbol(stepszConst, &stepsz, sizeof(stepsz));

        cudaMemcpy(seedTmp, cudaSeeds, numOfBlocks*numOfThreadsPerBlock*sizeof(int), cudaMemcpyHostToDevice);

        gpuExecute<<<numOfBlocks, numOfThreadsPerBlock>>>(wtRet, stepsRet, seedTmp);

        cudaMemcpy(wtTmp, wtRet, numOfBlocks * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(stepsTmp, stepsRet, numOfBlocks * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(cudaSeeds, seedTmp, numOfBlocks*numOfThreadsPerBlock*sizeof(int), cudaMemcpyDeviceToHost);

        cudaFree(wtRet);
        cudaFree(stepsRet);
        cudaFree(seedTmp);

        for(int tmpInc = 0; tmpInc < numOfBlocks; tmpInc++)
        {
          wt += wtTmp[tmpInc];
          steps += stepsTmp[tmpInc];
        }

        free(wtTmp);
        free(stepsTmp);
        wt = wt / (double)(N);
        steps_ave = steps / (double)(N);

        err = err + pow(w_exact - wt, 2);

        // printf("  %7.4f  %7.4f  %7.4f  %10.4e  %10.4e  %10.4e  %8d\n",
        //        x, y, z, wt, w_exact, fabs(w_exact - wt), steps_ave);
      }
    }
  }
  free(cudaSeeds);
  err = sqrt(err / (double)(n_inside));
  errPar = err;
  printf("\n\nRMS absolute error in solution = %e\n", err);
  timestamp();
}


void seq(int arc, char **argv)
{
  double a = 3.0;
  double b = 2.0;
  double c = 1.0;
  double chk;
  int dim = 3;
  double dx;
  double dy;
  double dz;
  double err;
  double h = 0.001;
  int i;
  int j;
  int k;
  int n_inside;
  int ni;
  int nj;
  int nk;
  double stepsz;
  int seed = 123456789;
  int steps;
  int steps_ave;
  int trial;
  double us;
  double ut;
  double vh;
  double vs;
  double x;
  double x1;
  double x2;
  double x3;
  double y;
  double w;
  double w_exact;
  double we;
  double wt;
  double z;
  int N = atoi(argv[1]);
  timestamp();

  printf("A = %f\n", a);
  printf("B = %f\n", b);
  printf("C = %f\n", c);
  printf("N = %d\n", N);
  printf("H = %6.4f\n", h);

  stepsz = sqrt((double)dim * h);

  if (a == i4_min(i4_min(a, b), c))
  {
    ni = 6;
    nj = 1 + i4_ceiling(b / a) * (ni - 1);
    nk = 1 + i4_ceiling(c / a) * (ni - 1);
  }
  else if (b == i4_min(i4_min(a, b), c))
  {
    nj = 6;
    ni = 1 + i4_ceiling(a / b) * (nj - 1);
    nk = 1 + i4_ceiling(c / b) * (nj - 1);
  }
  else
  {
    nk = 6;
    ni = 1 + i4_ceiling(a / c) * (nk - 1);
    nj = 1 + i4_ceiling(b / c) * (nk - 1);
  }

  err = 0.0;
  n_inside = 0;
  for (i = 1; i <= ni; i++)
  {
    x = ((double)(ni - i) * (-a) + (double)(i - 1) * a) / (double)(ni - 1);

    for (j = 1; j <= nj; j++)
    {
      y = ((double)(nj - j) * (-b) + (double)(j - 1) * b) / (double)(nj - 1);

      for (k = 1; k <= nk; k++)
      {
        z = ((double)(nk - k) * (-c) + (double)(k - 1) * c) / (double)(nk - 1);

        chk = pow(x / a, 2) + pow(y / b, 2) + pow(z / c, 2);

        if (1.0 < chk)
        {
          w_exact = 1.0;
          wt = 1.0;
          steps_ave = 0;
          // printf("  %7.4f  %7.4f  %7.4f  %10.4e  %10.4e  %10.4e  %8d\n",
          //        x, y, z, wt, w_exact, fabs(w_exact - wt), steps_ave);

          continue;
        }

        n_inside++;

        w_exact = exp(pow(x / a, 2) + pow(y / b, 2) + pow(z / c, 2) - 1.0);

        wt = 0.0;
        steps = 0;
        for (trial = 0; trial < N; trial++)
        {
          x1 = x;
          x2 = y;
          x3 = z;
          w = 1.0;
          chk = 0.0;
          while (chk < 1.0)
          {
            ut = r8_uniform_01(&seed);
            if (ut < 1.0 / 3.0)
            {
              us = r8_uniform_01(&seed) - 0.5;
              if (us < 0.0)
                dx = -stepsz;
              else
                dx = stepsz;
            }
            else
              dx = 0.0;

            ut = r8_uniform_01(&seed);
            if (ut < 1.0 / 3.0)
            {
              us = r8_uniform_01(&seed) - 0.5;
              if (us < 0.0)
                dy = -stepsz;
              else
                dy = stepsz;
            }
            else
              dy = 0.0;

            ut = r8_uniform_01(&seed);
            if (ut < 1.0 / 3.0)
            {
              us = r8_uniform_01(&seed) - 0.5;
              if (us < 0.0)
                dz = -stepsz;
              else
                dz = stepsz;
            }
            else
              dz = 0.0;

            vs = potential(a, b, c, x1, x2, x3);
            x1 = x1 + dx;
            x2 = x2 + dy;
            x3 = x3 + dz;

            steps++;

            vh = potential(a, b, c, x1, x2, x3);

            we = (1.0 - h * vs) * w;
            w = w - 0.5 * h * (vh * we + vs * w);

            chk = pow(x1 / a, 2) + pow(x2 / b, 2) + pow(x3 / c, 2);
          }
          wt = wt + w;
        }
        wt = wt / (double)(N);
        steps_ave = steps / (double)(N);

        err = err + pow(w_exact - wt, 2);

        // printf("  %7.4f  %7.4f  %7.4f  %10.4e  %10.4e  %10.4e  %8d\n",
        //       x, y, z, wt, w_exact, fabs(w_exact - wt), steps_ave);
      }
    }
  }
  err = sqrt(err / (double)(n_inside));
  errSeq = err;
  printf("\n\nRMS absolute error in solution = %e\n", err);
  timestamp();
}

int doubleCMP(double x, double y)
{
  double tmp = fabs(x-y);
  return tmp <= 0.01;
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
  if(!doubleCMP(errPar, errSeq))
    printf("Test FAILED \n\n");
  else
    printf("Test PASSED \n\n");

  return 0;
}