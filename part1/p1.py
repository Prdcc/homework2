"""M345SC Homework 2, part 1
Enrico Ancilotto
01210716
"""
from collections import deque, defaultdict
import random       #used in util which is only a testing class, not part of any algorithm
import networkx as nx       #used in util
import numpy as np       #used in util
import time         #used in util

class util:
    """
    Utility class for testing
    """
    @staticmethod
    def randA(n, numberOfEdges = -1, minStrength = 0, maxStrength = 1):
        numberOfEdges = numberOfEdges if numberOfEdges > 0 else 2*n
        A=[[] for i in range(n)]
        for i in range(numberOfEdges):
            startNode = random.randint(0, n-1)
            endNode = random.randint(0,n-1)
            if(startNode == endNode):
                endNode = (endNode +1) % n
            found = False
            for node,strength in A[startNode]:
                if node == endNode:
                    found = True
            if not found:
                strength = random.uniform(minStrength, maxStrength)
                A[startNode].append((endNode, strength))
                A[endNode].append((startNode, strength))
        return A
    
    @staticmethod
    def edgesL(L):
        return sum([len(i) for i in L])

    @staticmethod
    def randL(n, levels = 5, maxDependencies = 3):
        L = [[] for i in range(n)]
        intervals = np.sort(np.array(random.sample(range(1,n-1), levels)))
        intervals = np.insert(intervals,0,0)
        for i in range(len(intervals)-1):
            node = intervals[i]
            while(node < intervals[i+1]):
                numberOfDependencies = random.randint(0,maxDependencies)
                for j in range(numberOfDependencies):
                    newDependency = random.randint(intervals[i+1], n-1)
                    if not (newDependency in L[node]): L[node].append(newDependency)
                node += 1
        #np.random.shuffle(L)
        return L
        
    @staticmethod
    def timef(f,params=()):
        start = time.time()
        f(params)
        return time.time()-start

def getDay(node, dependencies, out_S):
        """
        Finds the day on which node and its dependencies should be scheduled
        Input:
        node: id of the node
        dependencies: a list containing the dependencies of the network
        out_S: a list of the form [node: day], this will be altered during execution

        Output: 1+day on which node should be scheduled

        This function calls itself recursively to figure out the day on which the 
        dependencies should be scheduled, it then takes a maximum of the outputs 
        which gives the day on which the task node should be scheduled. The recursion
        terminate as we are assuming that no two tasks depend on each other and that
        there are finitely many tasks.
        """
        if out_S[node] >= 0:    #if the day has already be found simply return it
            return out_S[node] + 1
        else:
            day = 0
            for i in dependencies[node]:        #iterate over dependencies of the node
                day = max(day, getDay(i,dependencies, out_S))
            out_S[node] = day       #set day on which task should be scheduled
            return day + 1


def scheduler(L):
    """
    Question 1.1
    Schedule tasks using dependency list provided as input

    Input:
    L: Dependency list for tasks. L contains N sub-lists, and L[i] is a sub-list
    containing integers (the sub-list my also be empty). An integer, j, in this
    sub-list indicates that task j must be completed before task i can be started.

    Output:
    S: A list of integers corresponding to the schedule of tasks. S[i] indicates
    the day on which task i should be carried out. Days are numbered starting
    from 0.

    Discussion: The main code lays in the function getDay (see documentation 
    below) which is executed for every node. The function finds the day that the 
    i-th task and its dependencies (and the dependencies of the dependencies and 
    so on) should be completed. Each edge (dependency) is accessed once and for every node we
    have to take the maximum over every neighbour, which is an O(q) operation (q
    is the degree of the node ie the number of dependencies). We also have to visit every node so in total the 
    number of operations is O(N+E) where N is the number of nodes and E the number
    of edges. This asymptotical speed is about as good as can be expected without
    knowing anything about the structure of the dependencies, as each node and edge
    has to be visited at least once.
    Note that this algorithm doesn't need L[i] to include dependencies of dependencies
    (it is actually slowed down by it). Which could potentially save time in pre-
    computation.
    """

    n = len(L)
    S = [-1] * n
    for node in range(n):   #iterate over all nodes
        getDay(node, L, S)
    return S


def findPath(A,a0,amin,J1,J2):
    """
    Question 1.2 i)
    Search for feasible path for successful propagation of signal
    from node J1 to J2

    Input:
    A: Adjacency list for graph. A[i] is a sub-list containing two-element tuples (the
    sub-list my also be empty) of the form (j,Lij). The integer, j, indicates that there is a link
    between nodes i and j and Lij is the loss parameter for the link.

    a0: Initial amplitude of signal at node J1

    amin: If a>=amin when the signal reaches a junction, it is boosted to a0.
    Otherwise, the signal is discarded and has not successfully
    reached the junction.

    J1: Signal starts at node J1 with amplitude, a0
    J2: Function should determine if the signal can successfully reach node J2 from node J1

    Output:
    L: A list of integers corresponding to a feasible path from J1 to J2.

    Discussion: This is a straightforward application of a breadth-first search.
    The only difference is that we must discard edges whose weight is not sufficient
    to transmit the signal, but in the worst case scenario (where every edge is
    visited) this is an O(M) operation (M is the number of edges, N the number of
    nodes). So we can repeat the analysis from lectures to get an O(M+N) total
    running time. (A normal breadth-first search would simply remove the first 
    condition in the first if-statement). Constructing the path is an O(d) operation
    where d is the length of the path found between the two nodes. As d is at most
    N (in practice d<<N) we again remain at O(M+N) for the total execution time.

    Breadth-first was chosen over depth-first as the path found by it is the one
    that requires the least number of steps, which is usually a desirable property.
    """
    class Found(Exception): pass
    previousNode = {}   #the n-th entry contains the node visited before reaching the n-th node
    minRetain = amin / a0
    Q = deque((J1,))
    previousNode[J1]=J1

    try:
        while(True):    #exit conditions managed by exceptions
            currNode = Q.popleft()  #raises exception if Q is empty which will break the loop
            for neighbourNode,retain in A[currNode]: #iterate through neighbours of n
                if (retain > minRetain) and not (neighbourNode in previousNode):
                    previousNode[neighbourNode] = currNode  
                    if neighbourNode == J2:
                        raise Found         #exception used to break double loop
                    Q.append(neighbourNode)
    except IndexError:      #raised by deque.popLeft if the deque is empty, ie when there is no path
        return []
    except Found:   #raised when a path has been found
        L=deque((J2,))
        while L[0] != J1:       #traverse the path backwards
            L.appendleft(previousNode[L[0]])
        return list(L)
    raise RuntimeError      #If reached something has gone wrong 


def a0min(A,amin,J1,J2):
    """
    Question 1.2 ii)
    Find minimum initial amplitude needed for signal to be able to
    successfully propagate from node J1 to J2 in network (defined by adjacency list, A)

    Input:
    A: Adjacency list for graph. A[i] is a sub-list containing two-element tuples (the
    sub-list my also be empty) of the form (j,Lij). The integer, j, indicates that there is a link
    between nodes i and j and Lij is the loss parameter for the link.

    amin: Threshold for signal boost
    If a>=amin when the signal reaches a junction, it is boosted to a0.
    Otherwise, the signal is discarded and has not successfully
    reached the junction.

    J1: Signal starts at node J1 with amplitude, a0
    J2: Function should determine min(a0) needed so the signal can successfully
    reach node J2 from node J1

    Output:
    (a0min,L) a two element tuple containing:
    a0min: minimum initial amplitude needed for signal to successfully reach J2 from J1
    L: A list of integers corresponding to a feasible path from J1 to J2 with
    a0=a0min
    If no feasible path exists for any a0, return output as shown below.

    Discussion: this algorithm is a straightforward application of Dijkstra's 
    algorithm where the normal path length of sum(w_ij) is substituted by min(w_ij).
    As this function is monotone, the algorithm will still give the correct answer.
    This has been implemented using a dictionary giving an execution time of 
    O(M+N^2) (M number of edges, N number of nodes), as M < N(N-1) which is O(N^2)
    we can simplify this to O(N^2). Instead of a dictionary we could use a binary 
    heap to get better performance. In this case we would have O((M+N)logN) 
    operations as seen in class. This is faster as long as M is at most O(N^2/logN)
    which is the case in many applications with large networks (as it is unfeasible 
    to have a direct connection between any two points in a graph).
    """
    previousNode = {}
    visited = set()
    unexploredNodes = defaultdict(lambda: -1)       #dictionary with default value -1, will contain the temporary cost estimate for unexplored nodes
    unexploredNodes[J1] = 1

    while len(unexploredNodes) > 0:     #implementation of Dijkstra's algorithm 
        maxRetain = -1
        for node, tempMax in unexploredNodes.items():   #find minimum
            if tempMax > maxRetain:
                maxRetain = tempMax
                nodeToExplore = node
        visited.add(nodeToExplore) 
        if nodeToExplore == J2:         #if we have reached the target we can break the loop
            break   
        unexploredNodes.pop(nodeToExplore)      #by having this after break statement the value is preserved for the last node

        for neighbour, retain in A[nodeToExplore]:      
            retain = min(maxRetain, retain)
            if not(neighbour in visited) and (retain > unexploredNodes[neighbour]):
                unexploredNodes[neighbour] = retain
                previousNode[neighbour] = nodeToExplore
        
    if(J2 in visited):
        L=deque((J2,))
        while L[0] != J1:       #reconstruct the path backwards
            L.appendleft(previousNode[L[0]])
        a0min = amin/unexploredNodes[J2]
        return a0min, list(L)
    return -1,[]


if __name__=='__main__':
    #add code here if/as desired
    L=None #modify as needed