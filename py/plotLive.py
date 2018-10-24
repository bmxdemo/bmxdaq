#!/usr/bin/env python
import bmxdata as bmx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys, glob, os, time
from optparse import OptionParser


def main():
    ## need those globals for animate callback
    global o, args, ax, fig, fname, d
    o,args=getOpts()
    ax,fig=initFig(o,args)
    fname,d=initData(o,args)
    # start animation
    ani = animation.FuncAnimation(fig, animate, interval=o.interval)
    plt.show()

def getOpts():

    parser = OptionParser()
    parser.add_option("--wf", dest="wf", action="store_true",
                      help="Plot waveform. It will assume it lives in data/wave.bin")
    parser.add_option("--interval", dest="interval", default=1000,
                      help="plotting interval", type='int')
    parser.add_option("--ymax", dest="ymax", default=0.0,
                      help="ymax", type='float')

    parser.add_option("--psavg", dest="psavg", action="store_true",
                      help="average ps")
    parser.add_option("--log", dest="log", action="store_true",
                      help="plot log")
    return parser.parse_args()


def initFig(o,args):

    fig = plt.figure()
    ny=2
    nx=1
    if (o.wf>0):
        nx+=1

    cc=0
    ax=[]
    for iy in range(ny):
        ax.append([])
        for ix in range(nx):
            nax=fig.add_subplot(nx,ny,cc+1)
            ax[-1].append(nax)
            cc+=1
    return ax,fig

def initData(o,args):
    if len(args)>0:
        fname=args[0]
    else:
        fname=max(glob.iglob('data/*.data.new'), key=os.path.getctime)

    print "Reading ",fname
    d=bmx.BMXFile(fname)
    return fname,d


def animate(i):
    global fname,d
    nr=d.update(replace=not o.psavg)
    print nr
    print "New records:",nr
    if d.haveMJD:
        print "Last MJD:",d.data['mjd'][-1]
    
    if (nr>0):
        ax[0][0].clear()
        ax[1][0].clear()
        ax[0][0].plot(d.freq[0],d.data['chan1_0'].mean(axis=0))
        ax[1][0].plot(d.freq[0],d.data['chan2_0'].mean(axis=0))

        if o.log:
            ax[0][0].semilogy()
            ax[1][0].semilogy()
        if (o.ymax>0):
            ymin,ymax=ax[0][0].get_ylim()
            ax[0][0].set_ylim(ymin,o.ymax)
            ymin,ymax=ax[1][0].get_ylim()
            ax[1][0].set_ylim(0,o.ymax)
    else:
        if (len(args)==0):
            print "Looking for new file..."
            fnamet=max(glob.iglob('data/*.data.new'), key=os.path.getctime)
            if fnamet!=fname:
                 print "Picked up ",fnamet
                 time.sleep(2) ## wait for the file to start for real
                 fname=fnamet
                 d=bmx.BMXFile(fname)



    if (o.wf>0):
        wfile = open("data/wave.bin")
        de=[('ch1','i1'),('ch2','i1')]
        da=np.fromfile(wfile,de)
        xx=np.arange(len(da))
        ax[0][1].clear()
        ax[1][1].clear()
        ax[0][1].plot(xx,da['ch1'])
        ax[1][1].plot(xx,da['ch2'])


if __name__=="__main__":
    main()



