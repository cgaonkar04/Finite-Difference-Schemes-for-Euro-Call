import numpy as np


def explicit_fd(S0,K,r,sigma,T,M,N):

    Smax=3*K

    dS=Smax/M
    dt=T/N

    grid=np.zeros((M+1,N+1))

    S=np.arange(M+1)*dS

    grid[:,N]=np.maximum(S-K,0)

    for n in range(N-1,-1,-1):

        grid[0,n]=0

        grid[M,n]=(
            Smax
            -K*np.exp(-r*(T-n*dt))
        )

        for i in range(1,M):

            a=0.5*dt*(sigma**2*i**2-r*i)

            b=1-dt*(sigma**2*i**2+r)

            c=0.5*dt*(sigma**2*i**2+r*i)

            grid[i,n]=(
                a*grid[i-1,n+1]
                +b*grid[i,n+1]
                +c*grid[i+1,n+1]
            )

    return np.interp(
        S0,
        S,
        grid[:,0]
    )