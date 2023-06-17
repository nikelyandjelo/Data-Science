import math as mt
import matplotlib.pyplot as plt
import numpy as np

def normal(n, n1, n2):
    S = ((np.random.randn(n)) * n2) + n1
    print('Implementation matrix ВВ=', S)
    statmat(S)
    plt.xlabel('normal')
    plt.hist(S, bins=20, facecolor="blue", alpha=0.5)
    plt.show()
    return S

def uniform(n, n1, n2):
    S = np.zeros((n))
    for i in range(n):
        S[i] = np.random.randint(n1, n2)
    print('Implementation matrix ВВ=', S)
    statmat(S)
    plt.xlabel('Uniform ')
    plt.hist(S, bins=20, facecolor="blue", alpha=0.5)
    plt.show()
    return S

def linemat(n, S, kidnap):
    S3 = np.zeros((n));
    S0 = np.zeros((n))
    for i in range(n):
        S0[i] = (kidnap * i)
        S3[i] = S0[i] + S[i]
    plt.plot(S3)
    plt.plot(S0)
    plt.xlabel('Dynamics (linear-uniform)')
    plt.show()
    return S3, S0

def cubmat(n, S, kidnap):
    S1 = np.zeros((n));
    S2 = np.zeros((n))
    for i in range(n):
        S2[i] = (kidnap * i * i * i)
        S1[i] = S2[i] + S[i]
    plt.plot(S1)
    plt.plot(S2)
    plt.xlabel('Dynamics (cubic-normal)')
    plt.show()
    return S1, S2

def statmat(S):
    mS = np.median(S)
    dS = np.var(S)
    scvS = mt.sqrt(dS)
    print('Mathematical expectation ВВ3=', mS)
    print('Dispersion ВВ3 =', dS)
    print('СКВ ВВ3=', scvS)

def assessment(n, S1, S3, S0, S, name):
    S4 = np.zeros((n))
    for i in range(n):
        S4[i] = (S3[i] - S0[i])
    plt.xlabel(name)
    plt.hist(S, bins=20, alpha=0.5, label='S')
    plt.hist(S1, bins=20, alpha=0.5, label='S1')
    plt.hist(S3, bins=20, alpha=0.5, label='S3')
    plt.hist(S4, bins=20, alpha=0.5, label='S4')
    plt.show()

N = 1000
lN = 5
rN = 5
lU = 0
rU = 100
kidnap1 = 0.000000005
kidnap2 = 0.000000009
mavs = np.random.randn(N)

normavs = normal(N, lN, rN)
unimavs = uniform(N, lU, rU)

cubnormal, s2 = cubmat(N, normavs, kidnap1)
plt.xlabel("Histograms (cubic-normal) ")
plt.hist(normavs, bins=20, alpha=0.5, label='S')
plt.hist(mavs, bins=20, alpha=0.5, label='S1')
plt.hist(cubnormal, bins=20, alpha=0.5, label='S3')
plt.show()
statmat(cubnormal)

assessment(N, mavs, cubnormal, s2, normavs, "Evaluation of statistical characteristics (cubic-normal)")

linuni, s2 = linemat(N, unimavs, kidnap2)
plt.xlabel("Histograms (linear-uniform) ")
plt.hist(unimavs, bins=20, alpha=0.5, label='S')
plt.hist(mavs, bins=20, alpha=0.5, label='S1')
plt.hist(linuni, bins=20, alpha=0.5, label='S3')
plt.show()
statmat(linuni)

assessment(N, mavs, linuni, s2, unimavs, "Evaluation of statistical characteristics (linear-uniform) ")