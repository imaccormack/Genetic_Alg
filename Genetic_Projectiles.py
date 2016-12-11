#!/usr/bin/env python

import random
import Tkinter
from math import sqrt
import numpy as numpy
import scipy
from numpy import random
from numpy import linalg
from scipy import integrate
from scipy import linalg
import pylab as P
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
#from showmat import showmat
from scipy.sparse import *
from scipy.sparse.linalg import eigsh
import Fluid
from scipy.cluster import vq
import scipy.spatial.distance


class Projectile(object):
    
    gen= 0
    fitness= 0.0
    numParams= 0
    params= [] #Will parameterize the shape of the projectile
        
    def __init__(self, generation, params):
        #Initialize a projectile of generation gen with shape parameters given by params
        self.params=params
        self.numParams= len(params)
        S= Fluid.simNoAnimation(self.params)
        #print S
        self.fitness = numpy.exp(-10000.0*S)
        
    def Mate(self, partner):
        #Produces a new Projectile by taking the average of this projectile's params with those of another, up to Gaussian variation
        tempParams= []
        temp= 0.0
        if random.rand() < 0.9:
            for i in range(self.numParams):
                if random.rand() < 0.9:
                    temp=  (self.params[i] + partner.params[i])/2.0 + numpy.random.normal(0.0,0.01)
                else:
                    temp=  (self.params[i] + partner.params[i])/2.0 + numpy.random.normal(0.0,1.0)
                tempParams.append(temp)
        else:
            temp= numpy.random.normal(0.0,1.0)
            tempParams.append(temp)
        return Projectile(self.gen,tempParams)
    
    def Fitness(self):
        #Returns fitness of projectile, which should be exponentially suppressed by drag coefficient
        return self.fitness 
    
    def Step(self):
        self.gen+=1
    
  
class Population(object):
    
    population= []
    popSize= 0
    gen= 0
    avgFitness= 0.0
    numParams= 6
    
    def __init__(self, size):
        #Initialize generation 0 of population with popSize given by size
        self.popSize= size
        for i in range(size):
            tempParams= []
            for k in range(self.numParams):
                if k<4:
                    tempParams.append(1.0)
                else:
                    temp= numpy.random.normal(0.0,1.5)
                    tempParams.append(temp)
            newMember= Projectile(self.gen,tempParams)
            self.population.append(newMember)
        self.AvgFitness()
        self.population.sort(key=lambda x: x.fitness, reverse= True)
        
    def Purge(self):
        #Use Fitness of population members to determine whether or not they get to survive
        tempPop= []
        for j in range(self.popSize):
            if random.rand() < self.population[j].Fitness():
                tempPop.append(self.population[j])
        self.population= tempPop
        
    def GetAllParams(self):
        #Returns a matrix of parameters
        n= self.numParams
        m= len(self.population)
        allParams= numpy.zeros((m,n))
        for i in range(m):
            allParams[i]= numpy.array(self.population[i].params)
        return allParams
        
    def Reproduce(self):
        #Once we have purged the weaklings, we can use each remaining memebers' Mate() methods to create the next generation
        ind1= 0
        ind2= 0
        willMate= False
        temp= 0.0
        fit1= 0.0
        fit2=0.0
        distMat= self.GetDistances()
        #print distMat
        tempSize= len(self.population)
        while len(self.population) < self.popSize:
            """#Random pairing
            ind1= random.randint(0, len(self.population)-1)
            ind2= random.randint(0, len(self.population)-1)"""
            #Pairing by similarity
            ind= numpy.argmin(distMat)
            ind1= ind/tempSize
            ind2= ind%tempSize
            distMat[ind1,ind2]= 10000.0
            self.population[ind1].Step()
            self.population.append(self.population[ind1].Mate(self.population[ind2]))
            
    def GetDistances(self):
        #Calculates distances between the parameters of all members of the population
        tempPopSize= len(self.population)
        distances= numpy.ones((tempPopSize,tempPopSize))*10000.0
        tempdiff= 0.0
        for i in range(tempPopSize):
            for j in range(i):
                for k in range(len(self.population[i].params)):
                    tempdiff+= (self.population[i].params[k] - self.population[j].params[k])**2
                distances[i,j]= sqrt(tempdiff)
                tempdiff= 0.0
        return distances
            
        

    def AvgFitness(self):
        temp = 0.0
        for p in self.population:
            temp+= p.fitness
        self.avgFitness= temp/self.popSize
    
    def GenStep(self):
        #Kill members of population using thier respectve Fitness() methods, mate remaining members to maintain population size, increment gen by 1
        #self.Purge()
        #self.population.sort(key=lambda x: x.fitness, reverse= True)
        self.population= self.population[0:(self.popSize/10)]
        print "Gen " + str(self.gen) + " max fitness " + str(self.population[0].fitness)
        self.Reproduce()
        self.AvgFitness()
        self.population.sort(key=lambda x: x.fitness, reverse= True)
        self.gen+=1
    

thePop= Population(150)
numgens= 10

initParams= thePop.GetAllParams()
for i in range(numgens):
    print "Current gen: " + str(i) 
    print "Gen " + str(i) + " average fitness " + str(thePop.avgFitness)
    thePop.GenStep()
    
print "Current gen: " + str(numgens) 
print "Gen " + str(numgens) + " average fitness " + str(thePop.avgFitness)

finParams= thePop.GetAllParams()
whitened= vq.whiten(finParams)

"""for i in range(1,10):
    c, d= vq.kmeans(whitened, i)
    print "Distortion with " + str(i) + " clusters: "
    print d"""

dst = scipy.spatial.distance.euclidean

def gap(data, refs=None, nrefs=20, ks=range(1,6)):
    """
    Compute the Gap statistic for an nxm dataset in data.
    Either give a precomputed set of reference distributions in refs as an (n,m,k) scipy array,
    or state the number k of reference distributions in nrefs for automatic generation with a
    uniformed distribution within the bounding box of data.
    Give the list of k-values for which you want to compute the statistic in ks.
    """
    shape = data.shape
    if refs==None:
        tops = data.max(axis=0)
        bots = data.min(axis=0)
        dists = scipy.matrix(scipy.diag(tops-bots))

        rands = scipy.random.random_sample(size=(shape[0],shape[1],nrefs))
        for i in range(nrefs):
            rands[:,:,i] = rands[:,:,i]*dists+bots
    else:
        rands = refs
    gaps = scipy.zeros((len(ks),))
    for (i,k) in enumerate(ks):
        (kmc,kml) = scipy.cluster.vq.kmeans2(data, k)
        disp = sum([dst(data[m,:],kmc[kml[m],:]) for m in range(shape[0])])

        refdisps = scipy.zeros((rands.shape[2],))
        for j in range(rands.shape[2]):
            (kmc,kml) = scipy.cluster.vq.kmeans2(rands[:,:,j], k)
            refdisps[j] = sum([dst(rands[m,:,j],kmc[kml[m],:]) for m in range(shape[0])])
        gaps[i] = scipy.log(scipy.mean(refdisps))-scipy.log(disp)
    return gaps

#print gap(whitened)


initx3= numpy.transpose(initParams)[4][0:100]
inity3= numpy.transpose(initParams)[5][0:100]

finx3= numpy.transpose(finParams)[4][0:100]
finy3= numpy.transpose(finParams)[5][0:100]

plt.scatter(initx3,inity3,c='r')
plt.scatter(finx3,finy3,c='g')
plt.show()


