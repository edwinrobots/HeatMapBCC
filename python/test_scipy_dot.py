import time;
import scipy as sp;
import scipy.linalg as la;
import numpy as np;

n=1000
A=sp.random.rand(n,n)
B=sp.random.rand(n,n)

t=time.time()
C=np.dot(A,B)
t=time.time()-t
f=2*n**3/t/1e9
print("Numpy dot:   time = %.2f seconds; flop rate = %.2f Gflops/s"%(t,f))

t=time.time()
C=sp.dot(A,B)
t=time.time()-t
f=2*n**3/t/1e9
print("Scipy dot:   time = %.2f seconds; flop rate = %.2f Gflops/s"%(t,f))

t=time.time()
C=la.blas.dgemm(1.0,A,B)
t=time.time()-t
f=2*n**3/t/1e9
print("Scipy dgemm: time = %.2f seconds; flop rate = %.2f Gflops/s"%(t,f))
