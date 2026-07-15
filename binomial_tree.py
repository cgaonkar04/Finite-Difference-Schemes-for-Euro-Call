import numpy as np


def binomial_call(S0,K,r,sigma,T,N):

    dt=T/N

    u=np.exp(sigma*np.sqrt(dt))
    d=1/u

    p=(np.exp(r*dt)-d)/(u-d)

    stock=np.array(
        [S0*(u**j)*(d**(N-j))
         for j in range(N+1)]
    )

    value=np.maximum(stock-K,0)

    for i in range(N-1,-1,-1):

        value=np.exp(-r*dt)*(
            p*value[1:i+2]
            +(1-p)*value[:i+1]
        )

    return value[0]