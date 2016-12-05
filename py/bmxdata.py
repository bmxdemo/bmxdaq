#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import sys

class BMXFile(object):

    def __init__(self,fname):
        ## old header!!
        #head_desc=[('nChan','i4'),
        #           ('fftsize','i4'),('fft_avg','i4'),('sample_rate','f4'),
        #           ('numin','f4'),('numax','f4'),('pssize','i4')]

        prehead_desc=[('magic','S8'),('version','i4')]
        f=open(fname);
        H=np.fromfile(f,prehead_desc,count=1)
        if H['magic'][:7]!=b'>>BMX<<':
            print("Bad magic.",H['magic'])
            sys.exit(1)
        if H['version']==1:
            maxcuts=10
            head_desc=[('nChan','i4'),('sample_rate','f4'),('fft_size','u4'),
                   ('ncuts','i4'),
                   ('numin','10f4'),('numax','10f4'),('fft_avg','10u4'),
                   ('pssize','10i4')]
        else:
            print ("Unknown version",H['version'])
            sys.exit(1)
        H=np.fromfile(f,head_desc,count=1)
        self.ncuts=H['ncuts'][0]
        self.nChan=H['nChan'][0]
        self.nP=H['pssize'][0]
        self.numin=H['numin'][0]/1e6
        self.numax=H['numax'][0]/1e6
        print("We have ",self.ncuts,"cuts:")
        self.freq=[]
        for i in range(self.ncuts):
            print("    Cut ",i," ",self.numin[i],'-',self.numax[i],'MHz #P=',self.nP[i])
            self.freq.append(self.numin[i]+(np.arange(self.nP[i])+0.5)*(self.numax[i]-self.numin[i])/self.nP[i])
        rec_desc=[]
        if self.nChan==1:
            for i in range(self.ncuts):
                rec_desc+=[('chan1_'+str(i),'f4',self.nP[i])]
        else:
            for i in range(self.ncuts):
                rec_desc+=[('chan1_'+str(i),'f4',self.nP[i]),
                           ('chan2_'+str(i),'f4',self.nP[i]), 
                           ('chanXR_'+str(i),'f4',self.nP[i]),
                           ('chanXI_'+str(i),'f4',self.nP[i])]
        rec_dt=np.dtype(rec_desc,align=False)
        self.rec_dt=rec_dt
        self.names=rec_dt.names
        self.data=np.fromfile(f,rec_dt)
        print ("Loading done.")

    def getNames(self, chan):
        if self.nChan==1:
            return ['chan1_'+str(chan)]
        else:
            return ['chan1_'+str(chan),'chan2_'+str(chan),'chanXR_'+str(chan),'chanXI_'+str(chan)]
        
    def plotAvgSpec(self, cut=0):
        for i, n in enumerate(self.getNames(cut)):
            plt.subplot(2,2,i+1)
            y=self.data[n].mean(axis=0)
            plt.plot(self.freq[cut],y)
            plt.xlabel('freq [MHz] ' + n)

    def nullBin(self, bin):
        for cut in [0]:
            for i, n in enumerate(self.getNames(cut)):
                self.data[n][:,bin]=0.0
                
