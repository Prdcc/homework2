"""M345SC Homework 2, part 2    
Enrico Ancilotto
01210716
"""
import numpy as np
import networkx as nx
from scipy.integrate import odeint
import scipy.sparse as sp
import matplotlib.pyplot as plt

class util:
    @staticmethod
    def getG():
        G=nx.barabasi_albert_graph(100,5)
        return G

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
    if tau != 0:
        raise ValueError("Tau expected to be zero, got %d" % (tau))

    tarray = np.linspace(0,tf,Nt+1)
    S = 0.05
    V = 0.1
    I = 0.05
    y0=np.array((S,I,V))

    B = np.array([[-g-k, a, 0], [0, -k-a, 0], [0, 0, -k]], dtype = float)
    v=np.array([0.0,0.0,0.0])

    def RHS(y,t):
        theta = theta0 + theta1*(1.0 - np.sin(2*np.pi*t))
        thetaSV = theta*y[2]*y[0]
        v[2] = k - thetaSV
        v[1] = thetaSV

        return B@y + v

    y = odeint(RHS, y0, tarray)
    S = y[:,0]
    V = y[:,2]
    I = y[:,1]
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
    a,theta0,theta1,g,k,tau=params
    tarray = np.linspace(0,tf,Nt+1)

    #Add code here
    A = nx.to_scipy_sparse_matrix(G,dtype=float)
    n = A.shape[0]
    doublen = 2*n
    doublepi = 2*np.pi

    q = A.sum(axis=0).A[0]
    F = sp.lil_matrix((n,n))

    for j in range(n):          #only called once and not too resource intensive, so little gains from optimising this
        sumqA = 0.0
        for l in range(n):
            sumqA += q[l]*A[l,j]
        scaling = 0 if sumqA == 0 else tau / sumqA 
        for i in range(n):
            F[i,j] = scaling * q[i]*A[i,j]
    
    del A   #no longer needed
    del q
    B = sp.block_diag((F,F,F), format="lil")

    rowF = np.zeros(n)
    for i in range(n):
        for j in range(n):
            rowF[i] -= F[j,i]
    del F

    B.setdiag(np.concatenate((rowF-g-k, rowF-k-a, rowF-k),axis=None))
    B.setdiag([a]*n, n)

    B = B.tocsr()       #csr is optimised for matrix-vector multiplication
    global dy
    global thetaSV
    dy = np.zeros(3*n)
    thetaSV = np.zeros(n)

    def RHS(y,t):
        """Compute RHS of model at time t
        input: y should be a 3N x 1 array containing with
        y[:N],y[N:2*N],y[2*N:3*N] corresponding to
        S on nodes 0 to N-1, I on nodes 0 to N-1, and
        V on nodes 0 to N-1, respectively.
        output: dy: also a 3N x 1 array corresponding to dy/dt

        Discussion: add discussion here
        """
        global dy       #using pre-allocation from before
        global thetaSV
        theta = theta0 + theta1*(1.0 - np.sin(doublepi*t))
        thetaSV = theta*y[:n]*y[doublen:]
        dy = B@y
        
        dy[n:doublen] += thetaSV
        dy[doublen:] += k - thetaSV
        return dy
    
    y0 = np.zeros(3*n, dtype = float)    #initial condition
    y0[doublen:] = 1.0
    y0[x] = 0.05
    y0[x+n] = 0.05
    y0[x+doublen] = 0.1

    y = odeint(RHS, y0, tarray)
    Smean = y[:, :n].mean(axis = 1)
    Imean = y[:, n:doublen].mean(axis = 1)
    Vmean = y[:, doublen:].mean(axis = 1)
    Svar = y[:, :n].var(axis = 1)
    if display:
        plt.plot(tarray,Smean)
        plt.plot(tarray,Imean)
        plt.plot(tarray,Vmean)
        plt.show()
        plt.plot(tarray, Svar)
        plt.show()
    return Smean,Svar


def diffusion(input=(None)):
    """Analyze similarities and differences
    between simplified infection model and linear diffusion on
    Barabasi-Albert networks.
    Modify input and output as needed.

    Discussion: add discussion here
    """
    theta0 = 80
    tau = 0.01
    params = (0,theta0,0,0,0,tau)

    return None #modify as needed


if __name__=='__main__':
    #add code here to call diffusion and generate figures equivalent
    #to those you are submitting
    G=None #modify as needed