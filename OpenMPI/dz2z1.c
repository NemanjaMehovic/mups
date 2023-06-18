#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define MASTER 0


double cpu_time(void)
{
  double value;

  value = (double)clock() / (double)CLOCKS_PER_SEC;

  return value;
}

int prime_number(int n, int rank, int sizeOfCluster)
{
  int i;
  int j;
  int prime;
  int total;
  int sum = 0;

  total = 0;
  for (i = 2 + rank; i <= n; i += sizeOfCluster)
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

  MPI_Reduce(&total, &sum, 1, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);

  return sum;
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

void test(int n_lo, int n_hi, int n_factor, int rank, int sizeOfCluster);

void par(int argc, char *argv[], int rank, int sizeOfCluster)
{
  int n_factor;
  int n_hi;
  int n_lo;
  int tmp[3];

  if(rank == MASTER)
  {
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
    tmp[0] = n_lo;
    tmp[1] = n_hi;
    tmp[2] = n_factor;
  }

  MPI_Bcast(tmp, 3, MPI_INT, MASTER, MPI_COMM_WORLD);

  test(tmp[0], tmp[1], tmp[2], rank, sizeOfCluster);

  if(rank == MASTER)
  {
    printf("\n");
    printf("PRIME_TEST\n");
    printf("  Normal end of execution.\n");
    printf("\n");
    timestamp();
  }
}

int nPar,primesPar;

void test(int n_lo, int n_hi, int n_factor, int rank, int sizeOfCluster)
{
  int i;
  int n;
  int primes;
  double ctime;

  if(rank == MASTER)
  {
    printf("\n");
    printf("  Call PRIME_NUMBER to count the primes from 1 to N.\n");
    printf("\n");
    printf("         N        Pi          Time\n");
    printf("\n");
  }

  n = n_lo;

  while (n <= n_hi)
  {
    ctime = cpu_time();

    primes = prime_number(n, rank, sizeOfCluster);

    ctime = cpu_time() - ctime;

    if(rank == MASTER)
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
  int rank, sizeOfCluster;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &sizeOfCluster);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(rank == MASTER)
  {
    printf("SEQUENTIAL EXECUTION\n");
    timeS = MPI_Wtime();
    seq(argc, argv);
    timeE = MPI_Wtime();
    parTime = timeE - timeS;
    printf("Execution time for sequential %f s \n\n", parTime);
    printf("PARALLEL EXECUTION\n");
    timeS = MPI_Wtime();
    par(argc, argv, rank, sizeOfCluster);
    timeE = MPI_Wtime();
    parTime = timeE - timeS;
    printf("Execution time for parallel %f s \n\n", parTime);
    if(nSeq != nPar || primesPar != primesSeq)
      printf("Test FAILED \n\n");
    else
      printf("Test PASSED \n\n");
  }
  else
  {
    par(argc, argv, rank, sizeOfCluster);
  }

  MPI_Finalize();
  return 0;
}
