#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import sys

class BMXFile(object):

    def __init__(self,fname):
        head_desc=[('nChan','i4'),
                   ('fftsize','i4'),('fft_avg','i4'),('sample_rate','f4'),
                   ('numin','f4'),('numax','f4'),('pssize','i4')]

        f=open(fname);
        H=np.fromfile(f,head_desc,count=1)
        print("header=",H)
        self.nP=H['pssize']
        self.numin=H['numin']/1e6
        self.numax=H['numax']/1e6
        self.nChan=H['nChan']
        self.freq=self.numin+(np.arange(self.nP)+0.5)*(self.numax-self.numin)/self.nP
        if self.nChan==1:
            rec_desc=[('chan1','f4',self.nP)]
        else:
            rec_desc=[('chan1','f4',self.nP),('chan2','f4',self.nP), ('chanXR','f4',self.nP),('chanXI','f4',self.nP)]
        rec_dt=np.dtype(rec_desc,align=False)
        self.rec_dt=rec_dt
        self.names=rec_dt.names
        self.data=np.fromfile(f,rec_dt)


    def plotAvgSpec(self):
        for i, n in enumerate(self.names):
            plt.subplot(2,2,i+1)
            y=self.data[n].mean(axis=0)
            plt.plot(self.freq,y)
            plt.xlabel('freq [MHz] ' + n)


