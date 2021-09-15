import cython
import numpy as np

cdef double euclidean(double[:] x, double[:] y):
  cdef int n = x.shape[0]
  cdef double res = 0
  cdef double resbis = 0
  cdef int i
  for i in range(n):
    res += (x[i] - y[i])**2
  return res
