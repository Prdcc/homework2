"""M345SC Homework 2, part 1
Enrico Ancilotto
01210716
"""
from collections import deque, defaultdict
import random       #used in util
import networkx as nx       #used in util
import numpy as np       #used in util


class util:
    
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
                A[startNode].append([endNode, strength])
                A[endNode].append([startNode, strength])
        return A
    
    @staticmethod
    def randL(n, levels = 5, maxDependencies = 3):
        L = [[] for i in range(n)]
        intervals = np.sort(np.array(random.sample(range(1,n-1), levels)))
        intervals = np.insert(intervals,0,0)
        for i in range(len(intervals)-1):
            node = intervals[i]
            while(node < intervals[i+1]):
                numberOfDependencies = random.randint(0,3)
                for j in range(numberOfDependencies):
                    newDependency = random.randint(intervals[i+1], n-1)
                    if not (newDependency in L[node]): L[node].append(newDependency)
                node += 1
        return L
        

def getDay(node, dependecies, outS):
    if outS[node] >= 0:
        return outS[node] + 1
    elif dependecies[node] == []:
        outS[node] = 0
        return 1
    else:
        day = max([getDay(i,dependecies, outS) for i in dependecies[node]])
        outS[node] = day
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

    Discussion: Add analysis here
    """
    n = len(L)
    S = [-1] * n
    for node in range(n):
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

    Discussion: Add analysis here
    """
    class Found(Exception): pass
    previousNode = {}
    minRetain = amin / a0
    Q = deque((J1,))
    previousNode[J1]=J1

    try:
        while(True):    #exit conditions managed by exceptions
            currNode = Q.popleft()
            for neighbourNode,retain in A[currNode]: #iterate through neighbours of n
                if (retain > minRetain) and not (neighbourNode in previousNode):
                    previousNode[neighbourNode] = currNode
                    if neighbourNode == J2:
                        raise Found         #exception used to break double loop
                    Q.append(neighbourNode)
    except IndexError:      #raised by deque.popLeft if the deque is empty
        return []
    except Found:
        L=deque((J2,))
        while L[0] != J1:
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

    Discussion: Add analysis here
    """
    previousNode = {}
    visited = set()
    unexploredNodes = defaultdict(lambda: -1)       #dictionary with default value -1
    unexploredNodes[J1] = 1
    while len(unexploredNodes) > 0:
        maxRetain = -1
        for node, tempMax in unexploredNodes.items():
            if tempMax > maxRetain:
                maxRetain = tempMax
                nodeToExplore = node
        visited.add(nodeToExplore)
        if nodeToExplore == J2:
            break
        unexploredNodes.pop(nodeToExplore)      #by having this after break statement the value is preserved for the last node


        for neighbour, retain in A[nodeToExplore]:
            retain = min(maxRetain, retain)
            if not(neighbour in visited) and (retain > unexploredNodes[neighbour]):
                unexploredNodes[neighbour] = retain
                previousNode[neighbour] = nodeToExplore
        
    if(J2 in visited):
        L=deque((J2,))
        while L[0] != J1:
            L.appendleft(previousNode[L[0]])
        a0min = amin/unexploredNodes[J2]
        return a0min, list(L)
    return -1,[]


if __name__=='__main__':
    #add code here if/as desired
    L=None #modify as needed