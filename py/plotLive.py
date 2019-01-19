#!/usr/bin/env python
from __future__ import print_function, division
import bmxdata as bmx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys, glob, os, time
from argparse import ArgumentParser


def main():
    ## need those globals for animate callback
    o=getOpts()
    ax,fig=initFig(o)
    fname,d=initData(o)
    state=[o,fname,d,fig,ax]
    # start animation
    ani = animation.FuncAnimation(fig, animate, interval=o.interval, fargs=[state])
    plt.show()

def getOpts():

    parser = ArgumentParser(description="Live plotting of BMX daq data")
    parser.add_argument('filename', type=str, nargs="?",
                    help='Filename to use, otherwise first .new file in data')
    parser.add_argument("-f", "--fields", dest="fields", type=str,default="chan1_0,chan2_0,chan3_0,chan4_0",
                      help="Which fields to plot, comma separated. Valid options are formed like: chan1_0 (for channel 1, cut0)"+
                         "chan13R_0 (for channel 13 real, cut 0), chan24I_1 (for channel 24 imag, cut1), " +
                         "wform:data/mywave.dat:x3 (for channel 3 waveform from wave.dat, can also do "+
                         "wform::x3 to assume data/wave.bin;  'x' in front of channel # signifies 2 card mode). "+
                         "Default: chan1_0,chan2_0,chan3_0,chan4_0")
    parser.add_argument("--interval", dest="interval", default=1000,
                      help="Plotting interval", type=int)
    parser.add_argument("--nx", dest="nx", help="Number of panels in x, if not set automatic.")
    parser.add_argument("--ymax", dest="ymax", default=0.0,
                      help="ymax", type=float)
    parser.add_argument("--psavg", dest="psavg", action="store_true",
                      help="Average power spectra")
    parser.add_argument("--linear", dest="log", action="store_false",
                      help="Use linear y scale (default is log)")
    o=parser.parse_args()
    if o.fields=='all':
        panels=['chan%i_0'%i for i in range(1,5)]
        for i in range(1,5):
            for j in range(i+1,5):
                for c in ['R','I']:
                    panels.append('chan%i%i%s_0'%(i,j,c))
    else:    
        panels=o.fields.split(",")
    panels=[s.strip() for s in panels]
    nx=o.nx if o.nx is not None else int(np.sqrt(len(panels)))
    ny=int(len(panels)/nx)
    if (nx*ny<len(panels)):
        nx+=1
    o.panels=panels
    o.nx=nx
    o.ny=ny
    #print(o)
    return o



def initFig(o):

    fig = plt.figure()
    ax=[fig.add_subplot(o.nx,o.ny,cc+1) for cc,_ in enumerate(o.panels)]
    return ax,fig

def initData(o):
    fname=o.filename
    if o.filename is None:
        globs=list(glob.iglob('data/*.data.new'))
        if len(globs)>0:
            fname=max(globs, key=os.path.getctime)
        else:
            print ("No filename specified and nothing in data/*.data.new. Stoping.")
            sys.exit(1)
    print("Reading ",fname)
    d=bmx.BMXFile(fname)
    return fname,d


def animate(i,state):
    o,fname,d,fig,ax=state
    nr=d.update(replace=not o.psavg)
    print(nr)
    print("New records:",nr)
    
    if (nr==0) and (o.filename is None):
        print("Looking for new file...")
        fnamet=max(glob.iglob('data/*.data.new'), key=os.path.getctime)
        if fnamet!=fname:
             print("Picked up ",fnamet)
             time.sleep(2) ## wait for the file to start for real
             fname=fnamet
             while True:
                 try:
                     d=bmx.BMXFile(fname)
                     break
                 except:
                     time.sleep(1)

             state[1]=fname
             state[2]=d

    if (nr>0):
        if d.haveMJD:
            print("Last MJD:",d.data['mjd'][-1])

        for cc,(name,ax) in enumerate(zip(o.panels,ax)):
            ax.clear()
            if 'chan'==name[:4]:
                cut=int(name.split('_')[-1])
                if (name in d.names):
                    ax.plot(d.freq[cut],d.data[name].mean(axis=0),'b-')

                    if o.log:
                        if ("R" in name) or ("I" in name):
                            ax.plot(d.freq[cut],-(d.data[name].mean(axis=0)),'r--')
                        ax.semilogy()
                    if (o.ymax>0):
                        ymin,ymax=ax.get_ylim()
                        ax.set_ylim(ymin,o.ymax)
                ax.text(.5,.9,name,
                        horizontalalignment='center',
                        transform=ax.transAxes)
            elif 'wform'==name[:5]:
                _,wfname,chan=name.split(':')
                if wfname=="":
                    wfname="data/wave.bin"
                wfile = open(wfname)
                de=[('1','i1'),('2','i1')]
                da=np.fromfile(wfile,de)
                N=len(da)
                print(N,da['1'].shape)
                if (chan=='x1'):
                    y=da['1'][:N//2]
                elif (chan=='x2'):
                    y=da['2'][:N//2]
                elif (chan=='x3'):
                    y=da['1'][N//2:]
                elif (chan=='x4'):
                    y=da['2'][N//2:]
                else:
                    y=da[chan]
                xx=np.arange(len(y))
                ax.plot(xx,y)


if __name__=="__main__":
    main()



