#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gfit.h"
#include "levmar_mle.h"


void test_rotated_gaussian_f() {
  int boxsize = 11;
  coord_data data = meshgrid2d(boxsize);
  int m, n;

  m = 7;
  n = boxsize * boxsize;

  /*
    p[0] is a
    p[1] is b
    p[2] is c
    p[3] is xc
    p[4] is yc
    p[5] is A
    p[6] is bg
   */

  double theta = 30.0 * M_PI / 180.0;
  double sx = 1.4;
  double sy = 1.4;
  double a = pow(cos(theta),2)/(2*pow(sx,2)) + pow(sin(theta),2)/(2*pow(sy,2));
  double b = -sin(2*theta)/(4*pow(sx,2)) + sin(2*theta)/(4*pow(sy,2));
  double c = pow(sin(theta),2)/(2*pow(sx,2)) + pow(cos(theta),2)/(2*pow(sy,2));
  double p[7] = {
    a, b, c, 0.5, 0.5, 100.0, 0.0
  };

  double* f = malloc(sizeof(double)* data.num_coords);
  
  // call function
  rotated_gaussian_f(p, f, m, n, &data);

  // for debugging, print out 'roi' value row-by-row
  for (int i=0; i<boxsize; i++) {
      for (int j=0; j<boxsize; j++) {
	  printf("%6.1f,", f[i * boxsize + j]);
      }
      printf("\n");
  }

  free(f);
  free_coord_data(&data);

}

void test_jacobian() {
  int boxsize = 11;
  coord_data data = meshgrid2d(boxsize);
  int m, n;

  m = 7;
  n = boxsize * boxsize;

  /*
    p[0] is a
    p[1] is b
    p[2] is c
    p[3] is xc
    p[4] is yc
    p[5] is A
    p[6] is bg
   */

  double theta = 30.0 * M_PI / 180.0;
  double sx = 1.4;
  double sy = 1.5;
  double a = pow(cos(theta),2)/(2*pow(sx,2)) + pow(sin(theta),2)/(2*pow(sy,2));
  double b = -sin(2*theta)/(4*pow(sx,2)) + sin(2*theta)/(4*pow(sy,2));
  double c = pow(sin(theta),2)/(2*pow(sx,2)) + pow(cos(theta),2)/(2*pow(sy,2));
  double p[7] = {
    a, b, c, 0.5, 0.5, 100.0, 10.0
  };

  double* err = malloc(sizeof(double)* data.num_coords);
  
  dlevmar_chkjac(&rotated_gaussian_f, &rotated_gaussian_df, p, m, n, &data, err);

  for (int i=0; i < n; i++) {
    printf("%8.4f ", err[i]);
  }
  
  free(err);
  free_coord_data(&data);
  
}

void test_gaussian_fit() {

  int m = 7;
  int n = 121;
  int boxsize = 11;
  
  coord_data coords = meshgrid2d(boxsize);
  
  double obs[121] = {
    15.0, 9.0,11.0,15.0,10.0, 8.0,12.0,16.0,12.0, 8.0, 7.0,
    12.0,11.0, 7.0,11.0,10.0, 9.0,13.0, 8.0,10.0,15.0,13.0,
    11.0, 9.0,11.0,14.0,10.0,11.0,15.0,16.0, 9.0,13.0, 8.0,
     9.0, 9.0, 8.0,13.0,22.0,30.0,31.0,19.0,11.0, 9.0, 8.0,
     6.0, 6.0,12.0,35.0,56.0,58.0,56.0,42.0,21.0,11.0, 5.0,
     9.0, 9.0,16.0,21.0,71.0,117.0,102.0,63.0,26.0,14.0,15.0,
     8.0,10.0,20.0,33.0,75.0,82.0,100.0,61.0,26.0,12.0,11.0,
    10.0, 7.0,10.0,19.0,36.0,59.0,65.0,26.0,28.0,21.0, 8.0,
    12.0, 6.0,12.0,10.0,20.0,30.0,31.0,27.0,21.0,12.0, 7.0,
    12.0,11.0,17.0, 7.0, 9.0,20.0,15.0,14.0,11.0, 7.0, 6.0,
    16.0,15.0, 9.0, 6.0,11.0,13.0, 3.0, 6.0,10.0,12.0, 9.0
  };

  double pars[7] = {0.2, 0.01, 0.2, 0.0, 0.0, 100.0, 5.0};
  double info[LM_INFO_SZ];
  
  dlevmar_mle_der(
		  &rotated_gaussian_f, &rotated_gaussian_df, pars, obs, m, n,
		  100, NULL, info, NULL, NULL, &coords, 0);
  
  for (int j=0; j < m; j++) {
    printf("parameter p[%d] = %12.4f \n", j, pars[j]);
  }

  printf("## Optimization info ##\n");

  printf("info[1] = %10.4E\n", info[1]);
  printf("info[5] = %.0f\n", info[5]);
  printf("info[6] = %.0f\n", info[6]);
  printf("info[7] = %.0f\n", info[7]);
  printf("info[8] = %.0f\n", info[8]);
  
}

int main() {
  /* test_rotated_gaussian_f(); */
  /* test_jacobian(); */
  test_gaussian_fit();
}
  
