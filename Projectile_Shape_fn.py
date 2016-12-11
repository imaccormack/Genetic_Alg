#!/usr/bin/env python

import random
import Tkinter
import numpy as numpy
import scipy
from numpy import random
from numpy import linalg
from scipy import integrate
from scipy import linalg
import pylab as P
import matplotlib.animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
#from showmat import showmat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import *
from scipy.sparse.linalg import eigsh

def r(x, y):
    return numpy.sqrt(x**2 +y**2)

def cosine(x,y,freq):
    return numpy.cos(freq*numpy.arctan2(x,y))

def sine(x,y,freq):
    return numpy.sin(freq*numpy.arctan2(x,y))

def rSeries(params,x,y):
    func= numpy.zeros(numpy.shape(x))
    R= r(x,y)
    for i in range(len(params)):
        func= numpy.add(func, (R**i)*params[i] )
    return func

def cosineSeries(params,x,y):
    func= numpy.zeros(numpy.shape(x))
    for i in range(len(params)):
        func+= params[i]*(cosine(x,y,i)**2)
    return func

def sineSeries(params,x,y):
    func= numpy.zeros(numpy.shape(x))
    for i in range(len(params)):
        func+= params[i]*(sine(x,y,i)**2)
    return func

def xSeries(params,x):
    func= numpy.zeros(numpy.shape(x))
    for i in range(0,len(params)):
        func+= params[i]*(x**(i+1))
    return func

def ySeries(params,y):
    func= numpy.zeros(numpy.shape(y))
    for i in range(0,len(params)):
        func+= params[i]*(y**(i+1))
    return func

def evalFunc(params,x,y):
    ind = len(params)/2
    xParams= params[0:ind]
    yParams= params[ind:]
    fofxy= numpy.zeros(numpy.shape(x))
    fofxy= fofxy + xSeries(xParams,x)
    fofxy= fofxy + ySeries(yParams,y)
    #Modulate by a gaussian
    R= r(x,y)
    one= R/R
    #fofxy= fofxy*numpy.exp(-(R**2)/0.25)
    fofxy = numpy.exp(-(R**2)/0.1)*fofxy
    return fofxy #/numpy.mean(fofxy)

"""height= 200
width= 200

x= numpy.arange(-1.0,1.0 ,1.0/width)

y= numpy.arange(-0.5,0.5 ,1.0/height)

xx, yy= numpy.meshgrid(x,y)
#testparams= [0.5, -0.4, 0.1, 0.05, 0.0, 0.5, -0.3, 0.0]
testparams = [0.0, 0.0, 0.9, 0.5, 0.0, 0.0, 0.0, 0.5]
Fofx= evalFunc(testparams,xx,yy)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(xx,yy,Fofx)

plt.show()"""
