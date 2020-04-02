# Integrate GLE using Leap frog or 4th order Runga-Kutta algorithm
import numpy as np
import random
import math
import numba
from numba import jit, float64, int32, int64

@jit(numba.types.UniTuple(float64[:],3)(int64,float64,float64,float64,float64,float64,float64,float64,float64,float64,int32),nopython=True,nogil=True)
def integrate_sing_exp(nsteps=1e6, dt=0.01, k=1, m=1, gamma=1, U0=7.482,
    x0=0, v0=1, y0=0.5, kT=2.494, scheme=1):
    x=np.zeros((nsteps,),dtype=np.float64)
    v=np.zeros((nsteps,),dtype=np.float64)
    y=np.zeros((nsteps,),dtype=np.float64)
    x[0]=x0
    y[0]=y0
    v[0]=v0
    fac_pot=4*U0/m
    if scheme==1:
        fac_rand=math.sqrt(2*kT/gamma/dt)
        xx=x[0]
        vv=v[0]
        yy=y[0]
        for var in range(1,nsteps):
            xi=random.gauss(0.0,1.0)

            kx1=dt*vv
            kv1=dt*(-k*(xx-yy)/m-fac_pot*(xx**3-xx))
            ky1=dt*(-k*(yy-xx)/gamma+fac_rand*xi)
            x1=xx+kx1/2
            v1=vv+kv1/2
            y1=yy+ky1/2

            kx2=dt*v1
            kv2=dt*(-k*(x1-y1)/m-fac_pot*(x1**3-x1))
            ky2=dt*(-k*(y1-x1)/gamma+fac_rand*xi)
            x2=xx+kx2/2
            v2=vv+kv2/2
            y2=yy+ky2/2

            kx3=dt*v2
            kv3=dt*(-k*(x2-y2)/m-fac_pot*(x2**3-x2))
            ky3=dt*(-k*(y2-x2)/gamma+fac_rand*xi)
            x3=xx+kx3
            v3=vv+kv3
            y3=yy+ky3

            kx4=dt*v3
            kv4=dt*(-k*(x3-y3)/m-fac_pot*(x3**3-x3))
            ky4=dt*(-k*(y3-x3)/gamma+fac_rand*xi)
            xx+=(kx1+2*kx2+2*kx3+kx4)/6
            vv+=(kv1+2*kv2+2*kv3+kv4)/6
            yy+=(ky1+2*ky2+2*ky3+ky4)/6

            x[var]=xx
            v[var]=vv
            y[var]=yy
    else:
        fac_rand=math.sqrt(2*kT*dt/gamma)
        for var in range(nsteps-1):
            xi=random.gauss(0.0,1.0)

            xx=x[var]+0.5*v[var]*dt
            v[var+1]=v[var]-dt*(k*(xx-y[var])/m+fac_pot*(xx**3-xx))
            x[var+1]=xx+0.5*v[var+1]*dt

            y[var+1]=-dt*k*(y[var]-xx)+fac_rand*xi
    return x,v,y
