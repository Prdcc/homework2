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

    @staticmethod
    def test(percentile):
        total = 0
        for i in range(1000):
            G=nx.barabasi_albert_graph(100,5)
            nodeDegrees = sorted(G.degree, key=lambda x: x[1])
            total += nodeDegrees[percentile][1]
        print(total)

    @staticmethod
    def getG2():
        G=nx.Graph()
        G.add_node(0)
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
    #del F
    del q
    #The tau term is given by sum F_ji = tau
    B.setdiag(np.concatenate((np.zeros(n)-tau-g-k, np.zeros(n)-tau-k-a, np.zeros(n)-tau-k),axis=None))
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
    global dy   #broadcast dy and thetaSV
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

        Discussion: The number of operation is kept at a minimum as B is a sparse
        matrix. dy and thetaSV are pre-allocated to avoid expensive memory allocation
        operations. Calculating theta is a very straight-forward operation, the 
        only expensive operation is computing sin which takes about an order of
        magnitude longer than all other operations. In any case this is a constant
        time operation so for large networks it won't affect runtime.
        Calculating thetaSV requires 2 multiplications for each node, so 2*n 
        operations, and adding it to dy at the end needs 3 additions per node. 

        Finally as B is a sparse matrix, it only performs the number of operations
        it requires, so 1 multiplication and 1 addition for each non-zero entry
        of B. We have 3n elements on the main diagonal, and another n terms given
        by alpha*I in the expression for dS/dt. We then have two entries for every
        edge given by the adjacency matrix. Potentially this could be an O(n^2) term,
        but more likely (eg in Barabasi-Albert graphs) it will be O(n), let's assume
        it is kn. This gives us an estimate of:
        (6+k)n multiplications
        (7+k)n additions
        As Python is an interpreted language, it requires expensive type-checks 
        however the code is vectorised giving us only a few of those per function-call
        so we can disregard them. There is also the time needed for the various
        memory accesses which is a lot harder to estimate and depends on the 
        exact implementation of the various functions, but it can hardly be more 
        than three per arithmetic operation (2 operands to be called and saving 
        the result). Giving us a further 3(13+2k)n operations.
        The total is then in the order of 4(13+2k)n operations
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
    """
    Returns the number of nodes that have at least tolerance infected cells
    """
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
    Imean, Vmean = data
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
    titles = ["Variance of Infected","Number of Infected Nodes","Number of Sick Nodes"]
    ylabels = ["Variance", "Number", "Number"]
    for i in range(3):
        f,axes = plt.subplots(len(percentiles),sharex=True)
        f.suptitle("Enrico Ancilotto - Created by plotComprehensive\n"+titles[i]+" for theta0="+str(theta0))
        lI=None
        lV=None
        for j in range(len(percentiles)):
            lI,=axes[j].plot(tarray,data[j][i])
            lV,=axes[j].plot(tarray,data[j][i+3])
            axes[j].set_title("Start node degree percentile: "+str(percentiles[j]))

        plt.legend([lI, lV],["Simple model", "Linear Diffusion"])
        for ax in axes.flat:
            ax.set(xlabel='Time',ylabel=ylabels[i])
        for ax in axes.flat:
            ax.label_outer()
        plt.show()

def plotSickLong(percentiles,Nt,tf,theta0,tau,m):
    data = []
    tarray = np.linspace(0,tf,Nt+1)
    for percentile in percentiles:
        data.append(getData(percentile,Nt,tf,theta0,tau,m))
    print()
    f,axes = plt.subplots(len(percentiles),sharex=True)
    f.suptitle("Enrico Ancilotto - Created by plotSickLong\nNumber of sick nodes for theta0="+str(theta0))
    lI=None
    lV=None
    for j in range(len(percentiles)):
        axes[j].plot(tarray,data[j][2])
        axes[j].set_title("Start node degree percentile: "+str(percentiles[j]))

    for ax in axes.flat:
        ax.set(xlabel='Time',ylabel="Number")
    for ax in axes.flat:
        ax.label_outer()
    plt.show()

def diffusion(input=(None)):
    """Analyze similarities and differences
    between simplified infection model and linear diffusion on
    Barabasi-Albert networks.
    Modify input and output as needed.

    Discussion: The first two figures show how changing parameters affects the evolution
    of the simplified model. Tau only compresses the x-axis so we can fix it and keep
    it constant. Theta0 determines how quickly vulnerable cells become infected. If
    it is zero then the total amount of infected and vulnerable cells remains
    constant and so will their average. Otherwise the vulnerable cells become infected 
    at a faster rate as theta0 is increased, until 100% of the cells become infected.
    It is interesting to note that at first the infection rate is slower as the 
    infection spreads through the network, when the infection has reached every 
    node it starts growing much faster, until it burns through the vulnerable cells
    and slows down again.

    The next figures compare the spread of the infection in the two models for two
    different values of theta0: 0 and 80. A cell is considered infected if there
    is any trace of the infection (ie I>10^-5) and sick if I is at least 10^-3. 
    The two data points tell us where the infection has reached and where it is 
    becoming more serious. The model is repeated changing the degree of the starting
    node, from ~4 to ~11 to ~38.

    In the first figures (3-5) theta0 is set to 80, this means that the simplified 
    model will see the number of infected cells slowly growing (though an infection
    rate of close to 100% is only reached after about 150 unit of times as can be
    crudely approximated from figure 2). We can see how the degree of the starting
    node strongly affects how quickly the variance of the linear model goes to 0,
    while it makes hardly any difference to the simplified model: in the latter 
    case nodes with higher degree will spread the infection to more nodes but at 
    a slower pace. The variance of the linear diffusion model decreases much faster
    and asymptotically reaches zero, while in the simple model we don't approach 
    zero: eventually 100% of the cells will be infected but they will cluster on
    the nodes with highest degree.

    In figure 4 we get a better idea of how the infection spreads: there are three
    points at which the infection plateaus: call them A, B, and C. A corresponds 
    to the degree of the starting node, C is the total number of nodes, while B
    is somewhere in between getting bigger as the degree of the starting number is
    increased and all but disappearing in the 99-th percentile case. Both models
    reach A extremely quickly. The simple model then grows in a pseudo-logistic 
    way between A and B and again between B and C. On the other hand the linear 
    diffusion moves between A and B in discrete jumps, before also growing 
    pseudo-logistically between B and C.

    In figure 5 we again see three distinct plateaus for the simple model (the
    behaviour of the linear model will be analysed in the discussion for figure
    8). Furthermore the three plateaus are the same as in figure 4 and give us a
    better idea of what B represents: it seems to be the number of neighbours
    of neighbours of the starting node. The infection spreads to the first ring 
    of neighbours, before stopping until it has grown enough to spread to the 
    second ring and then repeating the process between B and C.

    Figures 6 through 8 cover the case where theta0=0. The timescale is increased
    but otherwise the linear model will behave in the exact same way. In figure 
    6 we again see the variability approaching a small but non-zero number for the
    simple model as the small starting number of infected cells is spread between 
    the nodes. Figure 7 also shows a similar behaviour as before, albeit at a much
    slower pace.

    In figure 8 and 9 we get a better sense of the main difference between the 
    two models: the linear model quickly reaches a peak number of sick nodes given
    by the minimum between the degree of the starting node and the maximum sustainable
    sick population which seems to be about 12. It then decreases back to zero 
    at a speed which increases as the degree increases. In figure 9 (which is just
    figure 8 with a longer simulation time) we see how the simple model instead
    slowly asymptotically approaches the same limit as the maximum sustainable sick 
    population from before and levels off there.
    
    """
    m=5
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
        Imean /= m
        Vmean /= m
        return Imean, Vmean

    plotMean(getMeans(0.1),tarray,thetas,0.1)
    plotMean(getMeans(0.01),tarray,thetas,0.01)

    print("")
    """
    percentiles = [0,75,99]
    #plotComprehensive(percentiles,Nt,30,80,0.01,m)
    #plotComprehensive(percentiles,Nt,200,0,0.01,m)
    plotSickLong(percentiles,Nt,1000,0,0.01,m)
    return None #modify as needed


if __name__=='__main__':
    sns.set()
    diffusion()