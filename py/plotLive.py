#!/usr/bin/env python
import bmxdata as bmx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
from optparse import OptionParser

parser = OptionParser()

parser.add_option("--wf", dest="wf", default="",
                  help="Plot waveform")
parser.add_option("--interval", dest="interval", default=1000,
                  help="plotting interval", type='int')
parser.add_option("--psavg", dest="psavg", action="store_true",
                  help="average ps")
parser.add_option("--log", dest="log", action="store_true",
                  help="plot log")

(o, args) = parser.parse_args()

fig = plt.figure()

ny=2
nx=1
if (len(o.wf)>0):
    nx+=1
    
cc=0
ax=[]
for iy in range(ny):
    ax.append([])
    for ix in range(nx):
        nax=fig.add_subplot(nx,ny,cc+1)
        ax[-1].append(nax)
        cc+=1

d=bmx.BMXFile(args[0])
print ax[1][0]
print ax[0][0]
def animate(i):
    nr=d.update(replace=not o.psavg)
    print "New records:",nr
    if (nr>0):
        ax[0][0].clear()
        ax[1][0].clear()
        ax[0][0].plot(d.freq[0],d.data['chan1_0'].mean(axis=0))
        ax[1][0].plot(d.freq[0],d.data['chan2_0'].mean(axis=0))
        if o.log:
            ax[0][0].semilogy()
            ax[1][0].semilogy()

    if (len(o.wf)>0):
        wfile = open(o.wf)
        de=[('ch1','i1'),('ch2','i1')]
        da=np.fromfile(wfile,de)
        xx=np.arange(len(da))
        ax[0][1].clear()
        ax[1][1].clear()
        ax[0][1].plot(xx,da['ch1'])
        ax[1][1].plot(xx,da['ch2'])

ani = animation.FuncAnimation(fig, animate, interval=o.interval)
plt.show()


plt.show()


