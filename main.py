import numpy as np
import matplotlib.pyplot as plt

from BS import BS_call
from binomial import binomial_call
from explicit_FD import explicit_fd
from implicit_FD import implicit_fd
from CN import crank_nicolson

# Parameters

S0=100
K=100
r=0.05
sigma=0.2
T=1

M=100

Nvals=[2,5,8,10,15]

# Exact price
BS=BS_call(
    S0,K,r,sigma,T
)

print(
    "Black-Scholes:",
    BS
)


# Store prices

bin_prices=[]
exp_prices=[]
imp_prices=[]
cn_prices=[]

# Explicit FD separate study

N_exp=[300,500,700,1000,1500,2000]


for N in N_exp:

    exp_prices.append(
        explicit_fd(
            S0,
            K,
            r,
            sigma,
            T,
            M,
            N
        )
    )



plt.figure(figsize=(10,6))

plt.plot(
    N_exp,
    exp_prices,
    marker='o',
    label='Explicit FD'
)

plt.axhline(
    BS,
    linestyle='--',
    label='Black-Scholes'
)

plt.xlabel("Time Steps (N)")
plt.ylabel("Option Price")

plt.title(
    "Explicit FD Convergence"
)

plt.legend()
plt.show()

# Explicit FD error plot

plt.figure(figsize=(10,6))

plt.plot(
    N_exp,
    np.abs(
        np.array(exp_prices)-BS
    ),
    marker='o'
)

plt.yscale('log')

plt.xlabel("Time Steps (N)")
plt.ylabel("Absolute Error")

plt.title(
    "Explicit FD Error"
)

plt.show()

exp_prices=[]
for N in Nvals:

    bin_prices.append(
        binomial_call(
            S0,K,r,sigma,T,N
        )
    )

    exp_prices.append(
        explicit_fd(
            S0,K,r,sigma,T,M,N
        )
    )

    imp_prices.append(
        implicit_fd(
            S0,K,r,sigma,T,M,N
        )
    )

    cn_prices.append(
        crank_nicolson(
            S0,K,r,sigma,T,M,N
        )
    )


# Price convergence plot
plt.figure(figsize=(10,6))

plt.plot(
    Nvals,
    bin_prices,
    marker='o',
    label='Binomial'
)

plt.plot(
    Nvals,
    imp_prices,
    marker='o',
    label='Implicit FD'
)

plt.plot(
    Nvals,
    cn_prices,
    marker='o',
    label='Crank Nicolson'
)

plt.axhline(
    BS,
    linestyle='--',
    label='Black-Scholes'
)

plt.xlabel("Time/Grid Steps")
plt.ylabel("Option Price")

plt.title(
    "Convergence of Pricing Methods"
)

plt.legend()
plt.show()


# Error plot

plt.figure(figsize=(10,6))

plt.plot(
    Nvals,
    np.abs(
        np.array(bin_prices)-BS
    ),
    marker='o',
    label='Binomial'
)

plt.plot(
    Nvals,
    np.abs(
        np.array(exp_prices)-BS
    ),
    marker='o',
    label='Exlicit FD'
)


plt.plot(
    Nvals,
    np.abs(
        np.array(imp_prices)-BS
    ),
    marker='o',
    label='Implicit FD'
)

plt.plot(
    Nvals,
    np.abs(
        np.array(cn_prices)-BS
    ),
    marker='o',
    label='Crank Nicolson'
)

plt.yscale('log')
plt.xlabel("Time/Grid Steps")
plt.ylabel("Absolute Error")
plt.title(
    "Error vs Grid Size"
)

plt.legend()
plt.show()

# Barrier option plots 

plt.figure(figsize=(10,6))

plt.plot(
    S,
    grid[:,0],
    linewidth=2,
    label='Up-and-Out Call'
)

plt.axvline(
    B,
    linestyle='--',
    label='Barrier'
)

plt.xlabel(
    "Stock Price"
)

plt.ylabel(
    "Option Value"
)

plt.title(
    "Up-and-Out Barrier Call"
)

plt.legend()

plt.grid()

plt.show()