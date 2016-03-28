# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pylab as pl
# myarray = np.fromfile('usagov.txt')
# myarray = np.loadtxt('usagov.txt')
# myarray = np.fromfile('iris.csv')
raw = pd.read_csv('iris.csv')
data = raw.values
print data
x = data[:,0]
y = data[:,1]
# z = data[:,2]
# w = data[:,3]
pl.scatter(x,y,color = ['r','g','b'], s = [30,40,50], alpha=0.5)
# pl.legend(loc = 'upper right')
pl.figure()
pl.show()
s = set(data[:,4])
print s