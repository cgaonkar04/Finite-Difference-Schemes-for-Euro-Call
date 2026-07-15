import numpy as np
from scipy.linalg import solve


def crank_nicolson(S0,K,r,sigma,T,M,N):

    Smax=3*K

    dS=Smax/M
    dt=T/N

    grid=np.zeros((M+1,N+1))

    S=np.arange(M+1)*dS

    grid[:,N]=np.maximum(S-K,0)

    A=np.zeros((M-1,M-1))
    B=np.zeros((M-1,M-1))

    alpha_last=0
    gamma_last=0

    for i in range(1,M):

        alpha=0.25*dt*(sigma**2*i**2-r*i)

        beta=-0.5*dt*(sigma**2*i**2+r)

        gamma=0.25*dt*(sigma**2*i**2+r*i)

        if i>1:
            A[i-1,i-2]=-alpha

        A[i-1,i-1]=1-beta

        if i<M-1:
            A[i-1,i]=-gamma


        if i>1:
            B[i-1,i-2]=alpha

        B[i-1,i-1]=1+beta

        if i<M-1:
            B[i-1,i]=gamma

        alpha_last=alpha
        gamma_last=gamma


    for n in range(N-1,-1,-1):

        grid[0,n]=0

        grid[M,n]=(
            Smax
            -K*np.exp(-r*(T-n*dt))
        )

        rhs=B @ grid[1:M,n+1]

        rhs[0]+=alpha_last*grid[0,n]

        rhs[-1]+=gamma_last*grid[M,n]

        grid[1:M,n]=solve(A,rhs)

    return np.interp(
        S0,
        S,
        grid[:,0]
    )