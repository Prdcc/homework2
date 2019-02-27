"""M345SC Homework 2, part 2    
Enrico Ancilotto
01210716
"""
import numpy as np
import networkx as nx
from scipy.integrate import odeint
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns

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

def getB(A, params):
    a,theta0,theta1,g,k,tau=params
    n = A.shape[0]
    doublen = 2*n
    doublepi = 2*np.pi

    q = A.sum(axis=0).A[0]
    F = sp.lil_matrix((n,n))

    #generate F
    for j in range(n):          #only called once and not too resource intensive, so little gains from optimising this
        sumqA = 0.0
        for l in range(n):
            sumqA += q[l]*A[l,j]
        scaling = 0 if sumqA == 0 else tau / sumqA 
        for i in range(n):
            F[i,j] = scaling * q[i]*A[i,j]
    
    del A   #no longer needed
    B = sp.block_diag((F,F,F), format="lil")
    del F
    del q
    B.setdiag(np.concatenate((-tau-g-k, -tau-k-a, -tau-k),axis=None))
    B.setdiag([a]*n, n)

    return B.tocsr()       #csr is optimised for matrix-vector multiplication

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

    B = getB(A, params)
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
    Svar = y[:, :n].var(axis = 1)
    if display:
        plt.plot(tarray,Smean)
        plt.show()
        plt.plot(tarray, Svar)
        plt.show()
    return Smean,Svar

def getInfected(I,tolerance=1e-10):
    infected = np.where(I > tolerance,1,0)
    infectedCount = infected.sum(axis=1)
    return infectedCount

def modelNQ3(G,x=0,theta0=80,tau=0.01,tf=6,Nt=400):
    A = nx.to_scipy_sparse_matrix(G,dtype=float)
    n = A.shape[0]
    doublen = 2*n
    doublepi = 2*np.pi
    tarray = np.linspace(0,tf,Nt+1)
    B = getB(A, (0,theta0,0,0,0,tau))
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
        thetaSV = theta0*y[:n]*y[doublen:]
        dy = B@y
        
        dy[n:doublen] += thetaSV
        dy[doublen:] -= thetaSV
        return dy
    
    y0 = np.zeros(3*n, dtype = float)    #initial condition
    y0[doublen:] = 1.0
    y0[x] = 0.05
    y0[x+n] = 0.05
    y0[x+doublen] = 0.1

    y = odeint(RHS, y0, tarray)
    return y

def linearDiffusion(G,x=0,tau=0.01,tf=6,Nt=400):
    A = nx.to_scipy_sparse_matrix(G,dtype=float)
    n = A.shape[0]
    doublen = 2*n
    doublepi = 2*np.pi
    tarray = np.linspace(0,tf,Nt+1)
    
    q = A.sum(axis=0).A[0]
    A.setdiag(-q)

    B = sp.block_diag((A,A),format="csr")
    B *= tau
    y0 = np.zeros(2*n, dtype = float)    #initial condition
    y0[n:] = 1.0
    y0[x] = 0.05
    y0[x+n] = 0.1

    y = odeint(lambda y,t: B@y, y0, tarray)
    yComplete = np.zeros((Nt+1,3*n))
    yComplete[:,:n] = y[:,:n]
    yComplete[:,n:] = y       #will get same solution for S and I, no need to compute twice
    return yComplete

def linearDiffusionExact(G,x=0,tau=0.01,tf=6,Nt=400):
    """
    Exact solution with matrix exponential, not used as slower than ODEint
    """
    A = nx.to_scipy_sparse_matrix(G,dtype=float)
    n = A.shape[0]
    
    q = A.sum(axis=0).A[0]
    A.setdiag(-q)
    A *= tau
    
    y0 = np.zeros(2*n, dtype = float)    #initial condition
    y0[n:] = 1.0
    y0[x] = 0.05
    y0[x+n] = 0.1

    S = sp.linalg.expm_multiply(A, y0[:n], 0, tf, Nt+1)
    V = sp.linalg.expm_multiply(A, y0[n:], 0, tf, Nt+1)
    yComplete = np.zeros((Nt+1,3*n))
    yComplete[:,:n] = S
    yComplete[:,n:2*n] = S       #will get same solution for S and I, no need to compute twice
    del S
    yComplete[:,2*n:] = V
    return yComplete

def plotMean(data, tarray, thetas, tau):
    Imean, Vmean, IvarLinear, VvarLinear = data
    f,axes = plt.subplots(2,3,sharex=True,sharey=True)
    axes = axes.flatten()
    f.suptitle('Enrico Ancilotto - Created by plotMean\nChange in evolution of simple model as theta0 is varied for tau='+str(tau))

    for i in range(6):
        lI, = axes[i].plot(tarray,Imean[i], label = "Infected")
        lV, = axes[i].plot(tarray,Vmean[i], label = "Vulnerable")
        axes[i].set_title("theta0 = " + str(thetas[i]))
    axes[0].legend()
    for ax in axes:
        ax.set(xlabel='Time')
    for ax in axes:
        ax.label_outer()
    plt.show()

def getData(percentile,Nt,tf,theta0,tau,m):
    n=100
    SvarLinear = np.zeros(Nt+1)
    SvarNormal = np.zeros(Nt+1)
    
    infectedAverage = np.zeros(Nt+1)
    infectedLinear = np.zeros(Nt+1)

    sickAverage = np.zeros(Nt+1)
    sickLinear = np.zeros(Nt+1)
    for i in range(m):
        G=util.getG()
        nodeDegrees = sorted(G.degree, key=lambda x: x[1])
        x = nodeDegrees[percentile][0]
        yModelN = modelNQ3(G,x,theta0=theta0,Nt=Nt,tf=tf,tau=tau)
        yLinear = linearDiffusion(G,x,Nt=Nt,tf=tf,tau=tau)

        infectedAverage += getInfected(yModelN[:, n:2*n],1e-5)
        infectedLinear += getInfected(yLinear[:, n:2*n],1e-5)
        sickAverage += getInfected(yModelN[:, n:2*n],1e-3)
        sickLinear += getInfected(yLinear[:, n:2*n],1e-3)
        SvarLinear += yLinear[:,:n].var(axis = 1)
        SvarNormal += yModelN[:,:n].var(axis = 1)
        print(i+1,"/",m,"  ",end="\r")
    SvarLinear /= m
    SvarNormal /= m
    infectedAverage /= m
    infectedLinear /= m
    sickAverage /= m
    sickLinear /= m

    return SvarNormal, infectedAverage, sickAverage, SvarLinear, infectedLinear, sickLinear

def plotComprehensive(percentiles,Nt,tf,theta0,tau,m):
    data = []
    tarray = np.linspace(0,tf,Nt+1)
    for percentile in percentiles:
        data.append(getData(percentile,Nt,tf,theta0,tau,m))
    print()
    titles = ["Variability of Infected","Number of Infected Nodes","Number of Sick Nodes"]
    ylabels = ["Variability", "Number", "Number"]
    for i in range(3):
        f,axes = plt.subplots(len(percentiles),sharex=True)
        f.suptitle("Enrico Ancilotto - Created by plotComprehensive\n"+titles[i]+" for tau="+str(tau))
        lI=None
        lV=None
        for j in range(len(percentiles)):
            lI,=axes[j].plot(tarray,data[j][i])
            lV,=axes[j].plot(tarray,data[j][i+3])
            axes[j].set_title("Node degree percentile: "+str(percentiles[j]))

        plt.legend([lI, lV],["Simple model", "Linear Diffusion"])
        for ax in axes.flat:
            ax.set(xlabel='Time',ylabel=ylabels[i])
        for ax in axes.flat:
            ax.label_outer()
        plt.show()

def diffusion(input=(None)):
    """Analyze similarities and differences
    between simplified infection model and linear diffusion on
    Barabasi-Albert networks.
    Modify input and output as needed.

    Discussion: add discussion here
    """
    theta0 = 80
    m=20
    n=100
    Nt=2000
    tf=30
    tarray = np.linspace(0,tf,Nt+1)
    """
    thetas = [0, 10, 40, 80, 200, 2000]
    def getMeans(tau):
        Imean = np.zeros((len(thetas),Nt+1))
        Vmean = np.zeros((len(thetas),Nt+1))
        IvarLinear = np.zeros(Nt+1)
        VvarLinear = np.zeros(Nt+1)

        for i in range(m):
            print("\n",i," ",end="")
            G = util.getG()
            for j,theta in enumerate(thetas):
                print(j,end="")
                y = modelNQ3(G,x=0,theta0=theta,Nt=Nt,tf=tf,tau=tau)
                Imean[j] += y[:, n:2*n].mean(axis = 1)
                Vmean[j] += y[:, 2*n:].mean(axis = 1)
            y = linearDiffusion(G,x=0,Nt=Nt,tf=tf,tau=tau)
            IvarLinear += y[:, n:2*n].var(axis = 1)
            VvarLinear += y[:, n:2*n].var(axis = 1)
        Imean /= m
        Vmean /= m
        IvarLinear /= m
        VvarLinear /= m
        return Imean, Vmean, IvarLinear, VvarLinear

    plotMean(getMeans(0.1),tarray,thetas,0.1)
    plotMean(getMeans(0.01),tarray,thetas,0.01)
    print("")
    """
    percentiles = [0,75,99]
    #plotComprehensive(percentiles,Nt,tf,theta0,0.01,m)
    plotComprehensive(percentiles,Nt,tf,theta0,0.1,m)

    return None #modify as needed


if __name__=='__main__':
    #add code here to call diffusion and generate figures equivalent
    #to those you are submitting
    G=None #modify as needed
    sns.set()