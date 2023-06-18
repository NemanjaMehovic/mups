#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define mmm 15
#define npartt 4 * mmm *mmm *mmm

/*
 *  Variable declarations
 */

double epot;
double vir;
double count;

double moveSeq, ekinSeq, epotSeq, tscaleSeq, virSeq, velSeq, countSeq, npartSeq, denSeq;
double movePar, ekinPar, epotPar, tscalePar, virPar, velPar, countPar, npartPar, denPar;

/*
 *  Function declarations
 */

/*
 *  Compute average velocity
 */
  double
  velavg(int npart, double vh[], double vaver, double h){
    int i;
    double vaverh=vaver*h;
    double vel=0.0;
    double sq;
    extern double count;

    count=0.0;
 #pragma omp parallel for \
 default(none) \
 shared(npart, count, vh, vaverh)\
 private(sq) \
 schedule(dynamic) \
 reduction(+:vel)   
    for (i=0; i<npart*3; i+=3){
      sq=sqrt(vh[i]*vh[i]+vh[i+1]*vh[i+1]+vh[i+2]*vh[i+2]);
      if (sq>vaverh)
      {
        #pragma omp atomic update
        count++;
      }
      vel+=sq;
    }
    vel/=h;

    return(vel);
  }

  double
  velavgSeq(int npart, double vh[], double vaver, double h){
    int i;
    double vaverh=vaver*h;
    double vel=0.0;
    double sq;
    extern double count;

    count=0.0;
    for (i=0; i<npart*3; i+=3){
      sq=sqrt(vh[i]*vh[i]+vh[i+1]*vh[i+1]+vh[i+2]*vh[i+2]);
      if (sq>vaverh) count++;
      vel+=sq;
    }
    vel/=h;

    return(vel);
  }


void
  prnout(int move, double ekin, double epot, double tscale, double vir,
         double vel, double count, int npart, double den){
    double ek, etot, temp, pres, rp;

    ek=24.0*ekin;
    epot*=4.0;
    etot=ek+epot;
    temp=tscale*ekin;
    pres=den*16.0*(ekin-vir)/(double)npart;
    vel/=(double)npart;
    rp=(count/(double)npart)*100.0;
    printf(" %6d%12.4f%12.4f%12.4f%10.4f%10.4f%10.4f%6.1f\n",
           move,ek,epot,etot,temp,pres,vel,rp);

  }

void srand48(long);
  double drand48(void);
/*
 *  Sample Maxwell distribution at temperature tref
 */
  void
  mxwell(double vh[], int n3, double h, double tref){
    int i;
    int npart=n3/3;
    double r, tscale, v1, v2, s, ekin=0.0, sp=0.0, sc;
    
    srand48(4711);
    tscale=16.0/((double)npart-1.0);

//Ako se paralelizuje rezultati odstupaju za vise od +-0.1 zbog drand48
    //#pragma omp parallel for\
    default(none) \
    shared(n3, vh) \
    private(s, v1, v2, r) \
    schedule(dynamic)
    for (i=0; i<n3; i+=2) {
      s=2.0;
      while (s>=1.0) {
        v1=2.0*drand48()-1.0;
        v2=2.0*drand48()-1.0;
        s=v1*v1+v2*v2;
      }
      r=sqrt(-2.0*log(s)/s);
      vh[i]=v1*r;
      vh[i+1]=v2*r;
    }

    #pragma omp parallel for \
    default(none) \
    shared(vh, n3) \
    schedule(dynamic) \
    reduction(+:sp)
    for (i=0; i<n3; i+=3) sp+=vh[i];
    sp/=(double)npart;

    #pragma omp parallel for \
    default(none) \
    shared(n3, sp, vh) \
    schedule(dynamic) \
    reduction(+:ekin)
    for(i=0; i<n3; i+=3) {
      vh[i]-=sp;
      ekin+=vh[i]*vh[i];
    }

    sp=0.0;
    #pragma omp parallel for \
    default(none) \
    shared(vh, n3) \
    schedule(dynamic) \
    reduction(+:sp)
    for (i=1; i<n3; i+=3) sp+=vh[i];
    sp/=(double)npart;

    #pragma omp parallel for \
    default(none) \
    shared(n3, sp, vh) \
    schedule(dynamic) \
    reduction(+:ekin)
    for(i=1; i<n3; i+=3) {
      vh[i]-=sp;
      ekin+=vh[i]*vh[i];
    }

    sp=0.0;
    #pragma omp parallel for \
    default(none) \
    shared(vh, n3) \
    schedule(dynamic) \
    reduction(+:sp)
    for (i=2; i<n3; i+=3) sp+=vh[i];
    sp/=(double)npart;

    #pragma omp parallel for \
    default(none) \
    shared(n3, sp, vh) \
    schedule(dynamic) \
    reduction(+:ekin)
    for(i=2; i<n3; i+=3) {
      vh[i]-=sp;
      ekin+=vh[i]*vh[i];
    }

    sc=h*sqrt(tref/(tscale*ekin));

    #pragma omp parallel for \
    default(none) \
    shared(n3, sc, vh) \
    schedule(dynamic)
    for (i=0; i<n3; i++) vh[i]*=sc;
  }

  void
  mxwellSeq(double vh[], int n3, double h, double tref){
    int i;
    int npart=n3/3;
    double r, tscale, v1, v2, s, ekin=0.0, sp=0.0, sc;
    
    srand48(4711);
    tscale=16.0/((double)npart-1.0);

    for (i=0; i<n3; i+=2) {
      s=2.0;
      while (s>=1.0) {
        v1=2.0*drand48()-1.0;
        v2=2.0*drand48()-1.0;
        s=v1*v1+v2*v2;
      }
      r=sqrt(-2.0*log(s)/s);
      vh[i]=v1*r;
      vh[i+1]=v2*r;
    }

    for (i=0; i<n3; i+=3) sp+=vh[i];
    sp/=(double)npart;
    for(i=0; i<n3; i+=3) {
      vh[i]-=sp;
      ekin+=vh[i]*vh[i];
    }

    sp=0.0;
    for (i=1; i<n3; i+=3) sp+=vh[i];
    sp/=(double)npart;
    for(i=1; i<n3; i+=3) {
      vh[i]-=sp;
      ekin+=vh[i]*vh[i];
    }

    sp=0.0;
    for (i=2; i<n3; i+=3) sp+=vh[i];
    sp/=(double)npart;
    for(i=2; i<n3; i+=3) {
      vh[i]-=sp;
      ekin+=vh[i]*vh[i];
    }

    sc=h*sqrt(tref/(tscale*ekin));
    for (i=0; i<n3; i++) vh[i]*=sc;
  }


double
  mkekin(int npart, double f[], double vh[], double hsq2, double hsq){
    int i;
    double sum=0.0, ekin;
#pragma omp parallel for schedule(dynamic)\
reduction(+:sum)
    for (i=0; i<3*npart; i++) {
      f[i]*=hsq2;
      vh[i]+=f[i];
      sum+=vh[i]*vh[i];
    }
    ekin=sum/hsq;

    return(ekin);
  }

  double
  mkekinSeq(int npart, double f[], double vh[], double hsq2, double hsq){
    int i;
    double sum=0.0, ekin;

    for (i=0; i<3*npart; i++) {
      f[i]*=hsq2;
      vh[i]+=f[i];
      sum+=vh[i]*vh[i];
    }
    ekin=sum/hsq;

    return(ekin);
  }

void forces(int npart, double x[], double f[], double side, double rcoff)
{
  int i, j;
  double sideh, rcoffs;
  double xi, yi, zi, fxi, fyi, fzi, xx, yy, zz;
  double rd, rrd, rrd2, rrd3, rrd4, rrd6, rrd7, r148;
  double forcex, forcey, forcez;

  vir = 0.0;
  epot = 0.0;
  sideh = 0.5 * side;
  rcoffs = rcoff * rcoff;
#pragma omp parallel for \
default(none) \
private(j, xi, yi, zi, fxi, fyi, fzi, xx, yy, zz, rd, rrd, rrd2, rrd3, rrd4, rrd6, rrd7, r148, forcex, forcey, forcez)\
schedule(dynamic)\
shared(x, f, rcoffs, side, sideh, npart)\
reduction(+:epot)\
reduction(-:vir)
  for (i = 0; i < npart * 3; i += 3)
  {
    xi = x[i];
    yi = x[i + 1];
    zi = x[i + 2];
    fxi = 0.0;
    fyi = 0.0;
    fzi = 0.0;

    for (j = i + 3; j < npart * 3; j += 3)
    {
      xx = xi - x[j];
      yy = yi - x[j + 1];
      zz = zi - x[j + 2];
      if (xx < -sideh)
        xx += side;
      if (xx > sideh)
        xx -= side;
      if (yy < -sideh)
        yy += side;
      if (yy > sideh)
        yy -= side;
      if (zz < -sideh)
        zz += side;
      if (zz > sideh)
        zz -= side;
      rd = xx * xx + yy * yy + zz * zz;

      if (rd <= rcoffs)
      {
        rrd = 1.0 / rd;
        rrd2 = rrd * rrd;
        rrd3 = rrd2 * rrd;
        rrd4 = rrd2 * rrd2;
        rrd6 = rrd2 * rrd4;
        rrd7 = rrd6 * rrd;
        epot += (rrd6 - rrd3);
        r148 = rrd7 - 0.5 * rrd4;
        vir -= rd * r148;
        forcex = xx * r148;
        fxi += forcex;
        #pragma omp atomic update
        f[j] -= forcex;
        forcey = yy * r148;
        fyi += forcey;
        #pragma omp atomic update
        f[j + 1] -= forcey;
        forcez = zz * r148;
        fzi += forcez;
        #pragma omp atomic update
        f[j + 2] -= forcez;
      }
    }
    #pragma omp atomic update
    f[i] += fxi;
    #pragma omp atomic update
    f[i + 1] += fyi;
    #pragma omp atomic update
    f[i + 2] += fzi;
  }
}

void forcesSeq(int npart, double x[], double f[], double side, double rcoff)
{
  int i, j;
  double sideh, rcoffs;
  double xi, yi, zi, fxi, fyi, fzi, xx, yy, zz;
  double rd, rrd, rrd2, rrd3, rrd4, rrd6, rrd7, r148;
  double forcex, forcey, forcez;

  vir = 0.0;
  epot = 0.0;
  sideh = 0.5 * side;
  rcoffs = rcoff * rcoff;

  for (i = 0; i < npart * 3; i += 3)
  {
    xi = x[i];
    yi = x[i + 1];
    zi = x[i + 2];
    fxi = 0.0;
    fyi = 0.0;
    fzi = 0.0;

    for (j = i + 3; j < npart * 3; j += 3)
    {
      xx = xi - x[j];
      yy = yi - x[j + 1];
      zz = zi - x[j + 2];
      if (xx < -sideh)
        xx += side;
      if (xx > sideh)
        xx -= side;
      if (yy < -sideh)
        yy += side;
      if (yy > sideh)
        yy -= side;
      if (zz < -sideh)
        zz += side;
      if (zz > sideh)
        zz -= side;
      rd = xx * xx + yy * yy + zz * zz;

      if (rd <= rcoffs)
      {
        rrd = 1.0 / rd;
        rrd2 = rrd * rrd;
        rrd3 = rrd2 * rrd;
        rrd4 = rrd2 * rrd2;
        rrd6 = rrd2 * rrd4;
        rrd7 = rrd6 * rrd;
        epot += (rrd6 - rrd3);
        r148 = rrd7 - 0.5 * rrd4;
        vir -= rd * r148;
        forcex = xx * r148;
        fxi += forcex;
        f[j] -= forcex;
        forcey = yy * r148;
        fyi += forcey;
        f[j + 1] -= forcey;
        forcez = zz * r148;
        fzi += forcez;
        f[j + 2] -= forcez;
      }
    }
    f[i] += fxi;
    f[i + 1] += fyi;
    f[i + 2] += fzi;
  }
}

void
  fcc(double x[], int npart, int mm, double a){
    int ijk=0;
    int i,j,k,lg;

    #pragma omp parallel for collapse(4)\
    default(none) \
    lastprivate(ijk) \
    shared(mm, a, x) \
    schedule(dynamic)
    for (lg=0; lg<2; lg++)
      for (i=0; i<mm; i++)
        for (j=0; j<mm; j++)
          for (k=0; k<mm; k++) {
            ijk = (k + j*mm + i*mm*mm + lg*mm*mm*mm)*3;
            x[ijk]   = i*a+lg*a*0.5;
            x[ijk+1] = j*a+lg*a*0.5;
            x[ijk+2] = k*a;
            ijk += 3;
          }

    #pragma omp parallel for collapse(4)\
    default(none) \
    firstprivate(ijk) \
    shared(mm, a, x) \
    schedule(dynamic)
    for (lg=1; lg<3; lg++)
      for (i=0; i<mm; i++)
        for (j=0; j<mm; j++)
          for (k=0; k<mm; k++) {
            int tmp = ijk + (k + j*mm + i*mm*mm + (lg-1)*mm*mm*mm)*3;
            x[tmp]   = i*a+(2-lg)*a*0.5;
            x[tmp+1] = j*a+(lg-1)*a*0.5;
            x[tmp+2] = k*a+a*0.5;
          }

  }

  void
  fccSeq(double x[], int npart, int mm, double a){
    int ijk=0;
    int i,j,k,lg;

    for (lg=0; lg<2; lg++)
      for (i=0; i<mm; i++)
        for (j=0; j<mm; j++)
          for (k=0; k<mm; k++) {
            x[ijk]   = i*a+lg*a*0.5;
            x[ijk+1] = j*a+lg*a*0.5;
            x[ijk+2] = k*a;
            ijk += 3;
          }

    for (lg=1; lg<3; lg++)
      for (i=0; i<mm; i++)
        for (j=0; j<mm; j++)
          for (k=0; k<mm; k++) {
            x[ijk]   = i*a+(2-lg)*a*0.5;
            x[ijk+1] = j*a+(lg-1)*a*0.5;
            x[ijk+2] = k*a+a*0.5;
            ijk += 3;
          }

  }

void
  dscal(int n,double sa,double sx[], int incx){
    int i,j;

    if (incx == 1) {
      #pragma omp parallel for schedule(dynamic)
      for (i=0; i<n; i++)
        sx[i] *= sa;
    } else {
      #pragma omp parallel for schedule(dynamic)
      for (i=0; i<n; i++) {
        sx[i*incx] *= sa;
      }
    }
  }

  void
  dscalSeq(int n,double sa,double sx[], int incx){
    int i,j;

    if (incx == 1) {
      for (i=0; i<n; i++)
        sx[i] *= sa;
    } else {
      j = 0;
      for (i=0; i<n; i++) {
        sx[j] *= sa;
        j += incx;
      }
    }
  }

  void
  dfill(int n, double val, double a[], int ia){
    int i;
    #pragma omp parallel for schedule(dynamic)
    for (i=0; i<(n-1)*ia+1; i+=ia)
      a[i] = val;
  }

  void
  dfillSeq(int n, double val, double a[], int ia){
    int i;

    for (i=0; i<(n-1)*ia+1; i+=ia)
      a[i] = val;
  }

void
  domove(int n3, double x[], double vh[], double f[], double side){
    int i;
#pragma omp parallel for schedule(dynamic)
    for (i=0; i<n3; i++) {
      x[i] += vh[i]+f[i];
  /*
   *  Periodic boundary conditions
   */
      if (x[i] < 0.0)  x[i] += side;
      if (x[i] > side) x[i] -= side;
  /*
   *  Partial velocity updates
   */
      vh[i] += f[i];
  /*
   *  Initialise forces for the next iteration
   */
      f[i] = 0.0;
    }
  }

  void
  domoveSeq(int n3, double x[], double vh[], double f[], double side){
    int i;

    for (i=0; i<n3; i++) {
      x[i] += vh[i]+f[i];
  /*
   *  Periodic boundary conditions
   */
      if (x[i] < 0.0)  x[i] += side;
      if (x[i] > side) x[i] -= side;
  /*
   *  Partial velocity updates
   */
      vh[i] += f[i];
  /*
   *  Initialise forces for the next iteration
   */
      f[i] = 0.0;
    }
  }

double
secnds(void);



/*
 *  Main program : Molecular Dynamics simulation.
 */
void par()
{
  int move;
  double x[npartt * 3], vh[npartt * 3], f[npartt * 3];
  double ekin;
  double vel;
  double sc;
  double start, time;

  /*
   *  Parameter definitions
   */

  double den = 0.83134;
  double side = pow((double)npartt / den, 0.3333333);
  double tref = 0.722;
  double rcoff = (double)mmm / 4.0;
  double h = 0.064;
  int irep = 10;
  int istop = 20;
  int iprint = 5;
  int movemx = 20;

  double a = side / (double)mmm;
  double hsq = h * h;
  double hsq2 = hsq * 0.5;
  double tscale = 16.0 / ((double)npartt - 1.0);
  double vaver = 1.13 * sqrt(tref / 24.0);

  /*
   *  Initial output
   */

  printf(" Molecular Dynamics Simulation example program\n");
  printf(" ---------------------------------------------\n");
  printf(" number of particles is ............ %6d\n", npartt);
  printf(" side length of the box is ......... %13.6f\n", side);
  printf(" cut off is ........................ %13.6f\n", rcoff);
  printf(" reduced temperature is ............ %13.6f\n", tref);
  printf(" basic timestep is ................. %13.6f\n", h);
  printf(" temperature scale interval ........ %6d\n", irep);
  printf(" stop scaling at move .............. %6d\n", istop);
  printf(" print interval .................... %6d\n", iprint);
  printf(" total no. of steps ................ %6d\n", movemx);

  /*
   *  Generate fcc lattice for atoms inside box
   */
  fcc(x, npartt, mmm, a);
  /*
   *  Initialise velocities and forces (which are zero in fcc positions)
   */
  mxwell(vh, 3 * npartt, h, tref);
  dfill(3 * npartt, 0.0, f, 1);
  /*
   *  Start of md
   */
  printf("\n    i       ke         pe            e         temp   "
         "   pres      vel      rp\n  -----  ----------  ----------"
         "  ----------  --------  --------  --------  ----\n");

  start = secnds();

  for (move = 1; move <= movemx; move++)
  {

    /*
     *  Move the particles and partially update velocities
     */
    domove(3 * npartt, x, vh, f, side);

    /*
     *  Compute forces in the new positions and accumulate the virial
     *  and potential energy.
     */
    forces(npartt, x, f, side, rcoff);

    /*
     *  Scale forces, complete update of velocities and compute k.e.
     */
    ekin = mkekin(npartt, f, vh, hsq2, hsq);

    /*
     *  Average the velocity and temperature scale if desired
     */
    vel = velavg(npartt, vh, vaver, h);
    if (move < istop && fmod(move, irep) == 0)
    {
      sc = sqrt(tref / (tscale * ekin));
      dscal(3 * npartt, sc, vh, 1);
      ekin = tref / tscale;
    }

    /*
     *  Sum to get full potential energy and virial
     */
    if (fmod(move, iprint) == 0)
      prnout(move, ekin, epot, tscale, vir, vel, count, npartt, den);
  }


  movePar = move;
  ekinPar = ekin;
  epotPar = epot;
  tscalePar = tscale;
  virPar = vir;
  velPar = vel;
  countPar = count;
  npartPar = npartt;
  denPar = den;

  time = secnds() - start;

  printf("Time =  %f\n", (float)time);
}

time_t starttime = 0;

double secnds()
{

  return omp_get_wtime();
}

void seq()
{
  int move;
  double x[npartt * 3], vh[npartt * 3], f[npartt * 3];
  double ekin;
  double vel;
  double sc;
  double start, time;

  /*
   *  Parameter definitions
   */

  double den = 0.83134;
  double side = pow((double)npartt / den, 0.3333333);
  double tref = 0.722;
  double rcoff = (double)mmm / 4.0;
  double h = 0.064;
  int irep = 10;
  int istop = 20;
  int iprint = 5;
  int movemx = 20;

  double a = side / (double)mmm;
  double hsq = h * h;
  double hsq2 = hsq * 0.5;
  double tscale = 16.0 / ((double)npartt - 1.0);
  double vaver = 1.13 * sqrt(tref / 24.0);

  /*
   *  Initial output
   */

  printf(" Molecular Dynamics Simulation example program\n");
  printf(" ---------------------------------------------\n");
  printf(" number of particles is ............ %6d\n", npartt);
  printf(" side length of the box is ......... %13.6f\n", side);
  printf(" cut off is ........................ %13.6f\n", rcoff);
  printf(" reduced temperature is ............ %13.6f\n", tref);
  printf(" basic timestep is ................. %13.6f\n", h);
  printf(" temperature scale interval ........ %6d\n", irep);
  printf(" stop scaling at move .............. %6d\n", istop);
  printf(" print interval .................... %6d\n", iprint);
  printf(" total no. of steps ................ %6d\n", movemx);

  /*
   *  Generate fcc lattice for atoms inside box
   */
  fccSeq(x, npartt, mmm, a);
  /*
   *  Initialise velocities and forces (which are zero in fcc positions)
   */
  mxwellSeq(vh, 3 * npartt, h, tref);
  dfillSeq(3 * npartt, 0.0, f, 1);
  /*
   *  Start of md
   */
  printf("\n    i       ke         pe            e         temp   "
         "   pres      vel      rp\n  -----  ----------  ----------"
         "  ----------  --------  --------  --------  ----\n");

  start = secnds();

  for (move = 1; move <= movemx; move++)
  {

    /*
     *  Move the particles and partially update velocities
     */
    domoveSeq(3 * npartt, x, vh, f, side);

    /*
     *  Compute forces in the new positions and accumulate the virial
     *  and potential energy.
     */
    forcesSeq(npartt, x, f, side, rcoff);

    /*
     *  Scale forces, complete update of velocities and compute k.e.
     */
    ekin = mkekinSeq(npartt, f, vh, hsq2, hsq);

    /*
     *  Average the velocity and temperature scale if desired
     */
    vel = velavgSeq(npartt, vh, vaver, h);
    if (move < istop && fmod(move, irep) == 0)
    {
      sc = sqrt(tref / (tscale * ekin));
      dscalSeq(3 * npartt, sc, vh, 1);
      ekin = tref / tscale;
    }

    /*
     *  Sum to get full potential energy and virial
     */
    if (fmod(move, iprint) == 0)
      prnout(move, ekin, epot, tscale, vir, vel, count, npartt, den);
  }

  moveSeq = move;
  ekinSeq = ekin;
  epotSeq = epot;
  tscaleSeq = tscale;
  virSeq = vir;
  velSeq = vel;
  countSeq = count;
  npartSeq = npartt;
  denSeq = den;

  time = secnds() - start;

  printf("Time =  %f\n", (float)time);
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
  seq();
  printf("PARALLEL EXECUTION\n");
  par();
  if(doubleCMP(moveSeq, movePar) && doubleCMP(ekinSeq, ekinPar) && doubleCMP(epotSeq, epotPar) && doubleCMP(tscaleSeq, tscalePar) && doubleCMP(virSeq, virPar)
  && doubleCMP(velSeq, velPar) && doubleCMP(countSeq, countPar) && doubleCMP(npartSeq, npartPar) && doubleCMP(denSeq, denPar))
    printf("Test PASSED \n\n");
  else
    printf("Test FAILED \n\n");

  return 0;
}
