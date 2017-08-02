#ifndef _FHT_H_
#define _FHT_H_

int fht_float(float *buf, int log_n);
int fht_double(double *buf, int log_n);

#include "fht_avx.c"

#endif
