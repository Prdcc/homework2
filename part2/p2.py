"""M345SC Homework 2, part 2    
Enrico Ancilotto
01210716
"""
import numpy as np
import networkx as nx
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def model1(G,x=0,params=(50,80,105,71,1,0),tf=6,Nt=400,display=False):
    """
    Question 2.1
    Simulate model with tau=0

    Input:
    G: Networkx graph
    params: contains model parameters, see code below.
    tf,Nt: Solutions Nt time steps from t=0 to t=tf (see code below)
    display: A plot of S(t) for the infected node is generated when true

    x: node which is initially infected

    Output:
    S: Array containing S(t) for infected node
    """
    a,theta0,theta1,g,k,tau=params
    tarray = np.linspace(0,tf,Nt+1)
    S = 0.05
    V = 0.1
    I = 0.05
    y0=(V,I,S)

    B = np.array([[-k,0,0], [0,-k-a,0], [0,a,-g-k]])
    v=np.array([0,0,0])

    def RHS(y,t):
        theta = theta0 + theta1*(1 - np.sin(2*np.pi*t))
        thetaSV = theta*y[2]*y[0]
        v[0] = k - thetaSV
        v[1] = thetaSV

        return B@y + v
    y = odeint(RHS, y0, tarray)
    S = y[:,2]
    if display:
        plt.plot(tarray, S)
        plt.show()
    return S

def modelN(G,x=0,params=(50,80,105,71,1,0.01),tf=6,Nt=400,display=False):
    """
    Question 2.1
    Simulate model with tau=0

    Input:
    G: Networkx graph
    params: contains model parameters, see code below.
    tf,Nt: Solutions Nt time steps from t=0 to t=tf (see code below)
    display: A plot of S(t) for the infected node is generated when true

    x: node which is initially infected

    Output:
    Smean,Svar: Array containing mean and variance of S across network nodes at
                each time step.
    """
    a,theta0,theta1,k,g,tau=params
    tarray = np.linspace(0,tf,Nt+1)
    Smean = np.zeros(Nt+1)
    Svar = np.zeros(Nt+1)

    #Add code here
    A = nx.to_numpy_matrix(G)
    n = A.shape[0]
    q = np.sum(A, axis=0)
    
    F=np.zeros((n,n))
    for j in range(n):          #only called once and not resource intensive, so little gains from optimising this
        sumqA = 0
        for k in range(n):
            sumqA += q[k]*A[k,j]
        scaling = tau / sumqA
        for i in range(n):
            F[i,j] = scaling * q[i]*A[i,j]

    Fp = F - F.transpose()
    v = np.zeros(3*n)
    B = np.zeros((3*n, 3*n))
    
    B[:n,:n] += Fp
    B[n:2*n,n:2*n] += Fp
    B[2*n:,2*n] += Fp

    for i in range(n):
        B[i,i] += -g-k
        B[i,i+n] += a
        B[n+i, n+i] += -k-a
        B[2*n+i, 2*n+1] = -k


    def RHS(y,t):
        """Compute RHS of model at time t
        input: y should be a 3N x 1 array containing with
        y[:N],y[N:2*N],y[2*N:3*N] corresponding to
        S on nodes 0 to N-1, I on nodes 0 to N-1, and
        V on nodes 0 to N-1, respectively.
        output: dy: also a 3N x 1 array corresponding to dy/dt

        Discussion: add discussion here
        """
        theta = theta0 + theta1*(1 - np.sin(2*np.pi*t))
        thetaSV = theta*y[:n]*y[2*n:]
        v[n:2*n] = thetaSV
        v[2*n:] = k - thetaSV

        return B@y + v
    
    y0 = np.ones(3*n, dtype = float)    #initial condition
    y0[:2*n] = 0.0
    y0[x] = 0.05
    y0[x+n] = 0.05
    y0[x+2*n] = 0.1

    y = odeint(RHS, y0, tarray)


    return Smean,Svar


def diffusion(input=(None)):
    """Analyze similarities and differences
    between simplified infection model and linear diffusion on
    Barabasi-Albert networks.
    Modify input and output as needed.

    Discussion: add discussion here
    """


    return None #modify as needed


if __name__=='__main__':
    #add code here to call diffusion and generate figures equivalent
    #to those you are submitting
    G=None #modify as needed