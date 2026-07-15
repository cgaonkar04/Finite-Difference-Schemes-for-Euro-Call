import numpy as np
from scipy.linalg import solve


def implicit_fd(S0,K,r,sigma,T,M,N):

    Smax=3*K

    dS=Smax/M
    dt=T/N

    grid=np.zeros((M+1,N+1))

    S=np.arange(M+1)*dS

    grid[:,N]=np.maximum(S-K,0)

    A=np.zeros((M-1,M-1))

    for i in range(1,M):

        a=-0.5*dt*(sigma**2*i**2-r*i)

        b=1+dt*(sigma**2*i**2+r)

        c=-0.5*dt*(sigma**2*i**2+r*i)

        if i>1:
            A[i-1,i-2]=a

        A[i-1,i-1]=b

        if i<M-1:
            A[i-1,i]=c


    for n in range(N-1,-1,-1):

        rhs=grid[1:M,n+1]

        grid[1:M,n]=solve(A,rhs)


    return np.interp(
        S0,
        S,
        grid[:,0]
    )