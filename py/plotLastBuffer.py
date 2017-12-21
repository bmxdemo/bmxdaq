#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import sys, glob, os, time
from optparse import OptionParser

## This assumes 2 channels.

parser = OptionParser()
parser.add_option("--points", dest="points", default = 1000, 
            type = 'int', help = "number of points to plot")
parser.add_option("--f", dest="file", help= "name of file containing buffer")
(o, args) = parser.parse_args()

wfile = open(o.file)
de=[('ch1','i1'),('ch2','i1')]
da=np.fromfile(wfile,de)
size = o.points  ##number of points to plot
xx=np.arange(size)
plt.figure(1)
plt.subplot(211)
plt.plot(xx,da['ch1'][0:size])
plt.title("CH 1")
plt.subplot(212)
plt.plot(xx,da['ch2'][0:size])
plt.title("CH 2")
plt.show()





