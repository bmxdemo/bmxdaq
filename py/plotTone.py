#!/usr/bin/env python3
import bmxdata as bmx
import matplotlib.pyplot as plt
import numpy as np
import sys

for fn in sys.argv[1:]:

    d=bmx.BMXFile(fn)
    d.freq-=250.0
    d.freq*=1000

    def rnm(x, N):
        return np.convolve(x, np.ones((N,))/N,mode='valid')[(N-1):]

    if False:
        for i, n in enumerate(d.names):
            plt.subplot(2,2,i+1)
            y=d.data[n].mean(axis=0)
            plt.plot(d.freq,y)
            plt.xlabel('delta freq [kHz] ' + n)
    if True:
        plt.figure(figsize=(10,8))
        for i,n in enumerate(d.names):
            mxf=[]
            mx=[]
            for line in d.data[n]:
                i=abs(line).argmax()
                mxf.append(d.freq[i])
                #print (line.sum(), line[i-20:i+20].sum())
                #plt.plot(range(len(line)),line)
                #plt.show()
                mx.append(line[max(0,i-20):i+20].sum())

            plt.subplot(2,1,1)
            x=np.arange(len(mx))*0.107
            mx=np.array(mx)
            mx/=mx[0]
            plt.plot(rnm(x,50),rnm(mx,50),label=n)
            plt.subplot(2,1,2)
            plt.plot(rnm(x,50),rnm(mxf,50),label=n)
        plt.subplot(2,1,1)
        plt.ylabel("Rel. amplitude")
        plt.legend()
        plt.subplot(2,1,2)
        plt.ylabel("delta f [kHz]")
        plt.xlabel("time[s]")
        plt.show()
        #plt.savefig(fn.replace('.data','.png'))

