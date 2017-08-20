import numpy as np
import matplotlib.pyplot as plt
import sys
import  matplotlib.colors as colors

class BMXFile(object):
    freqOffset = 1100
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
        self.version=H['version']
        if self.version<=2:
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
        self.fft_size=H['fft_size'][0]
        self.sample_rate=H['sample_rate']/1e6
        self.deltaT = 1./self.sample_rate*self.fft_size/1e6
        self.nP=H['pssize'][0]
        self.numin=(H['numin'][0]/1e6)[:self.ncuts]
        self.numax=(H['numax'][0]/1e6)[:self.ncuts]
        print("We have ",self.ncuts,"cuts:")
        self.freq=[]
        for i in range(self.ncuts):
            print("    Cut ",i," ",self.numin[i],'-',self.numax[i],'MHz #P=',self.nP[i])
            self.freq.append(self.freqOffset+self.numin[i]+(np.arange(self.nP[i])+0.5)*(self.numax[i]-self.numin[i])/self.nP[i])
        rec_desc=[]
        if self.version>=2:
            rec_desc+=[('num_nulled','i4',H['nChan'])]
        if self.nChan==1:
            for i in range(self.ncuts):
                rec_desc+=[('chan1_'+str(i),'f4',self.nP[i])]
        else:
            for i in range(self.ncuts):
                rec_desc+=[('chan1_'+str(i),'f4',self.nP[i]),
                           ('chan2_'+str(i),'f4',self.nP[i]), 
                           ('chanXR_'+str(i),'f4',self.nP[i]),
                           ('chanXI_'+str(i),'f4',self.nP[i])]
        if self.version>=2:
            rec_desc+=[('nu_tone','f4')]
        rec_dt=np.dtype(rec_desc,align=False)
        self.rec_dt=rec_dt
        self.names=rec_dt.names
        self.data=np.fromfile(f,rec_dt)
        self.fhandle=f
        self.nSamples = len(self.data)
        print ("Loading done, %i samples"%(len(self.data)))

    def update(self,replace=False):
        ndata=np.fromfile(self.fhandle,self.rec_dt)
        nd=len(ndata)
        if replace:
            self.data=ndata
        else:
            self.data=np.hstack((self.data,ndata))
        return nd

    def getNames(self, cut):
        if self.nChan==1:
            return ['chan1_'+str(cut)]
        else:
            return ['chan1_'+str(cut),'chan2_'+str(cut),'chanXR_'+str(cut),'chanXI_'+str(cut)]

    def getAutoName(self,chan,cut):
        return 'chan%i_%i'%(chan,cut)
    
    def getRadar (self,fmin=1242., fmax=1247.):
        imin,imax,_,_=self.f2iminmax(fmin,fmax)
        da=(self.data[:]['chan1_0'])[:,imin:imax]+(self.data[:]['chan2_0'])[:,imin:imax]
        da=da.mean(axis=1)
        self.radarOn=(da>da.mean()) ## this seems to work pretty well!

    def filterRadar(self):
        ## fill in radar sections with neighbors
        for i,r in enumerate(self.radarOn):
            if r:
                ## find the closest non-radar point
                l=i-1
                h=i+1
                while True:
                    if not(self.radarOn[l]):
                        j=l
                        break
                    if not(self.radarOn[h]):
                        j=h
                        break
                    if (l>0):
                        l-=1
                    if (h<self.nSamples):
                        h+=1
                self.data[i]=self.data[j]

    def normalizeOnRegion(self,fmin,fmax,cut=0):
        imin,imax,_,_=self.f2iminmax(fmin,fmax)

        for ch in [1,2]:
            name=self.getAutoName(ch,cut)
            nfactor=(self.data[:][name])[:,imin:imax].mean(axis=1)
            for i in range(self.nSamples):
                self.data[i][name]/=nfactor[i]

    def f2iminmax(self,fmin,fmax,cut=0,binSize=1):
        if fmin is None:
            imin=0
            fmin=self.freq[cut][0]
        else:
            imin=(self.freq[cut]>fmin).argmax()-1
            if (imin<0):
                imin=0
            fmin=self.freq[cut][imin]
        if fmax is None:
            imax=len(self.freq[cut])
            fmax=self.freq[cut][-1]
        else:
            imax=(self.freq[cut]<fmax).argmin()+1
            if (imax-imin)%binSize!=0:
                imax-=(imax-imin)%binSize+binSize
            if imax>=len(self.freq[cut]):
                imax=len(self.freq[cut])-1
            fmax=self.freq[cut][imax]
        return imin,imax,fmin,fmax
    
    def plotAvgSpec(self, cut=0):
        for i, n in enumerate(self.getNames(cut)):
            plt.subplot(2,2,i+1)
            y=self.data[n].mean(axis=0)
            plt.plot(self.freq[cut],y)
            plt.xlabel('freq [MHz] ' + n)
   
    #waterfall plot of frequencies over time. Can either use log scale, or subtract and divide off the mean 
    def plotWaterfall(self, fmin=None, fmax=None, nsamples=None, cut=0, binSize = 4, subtractMean = False, minmax=None):
        if nsamples is  None:
            nsamples = self.nSamples  #plot all samples in file
        imin,imax,fmin,fmax=self.f2iminmax(fmin,fmax,cut,binSize)
        
        for n in range(2):
           plt.subplot(2,1, n+1)
           arr = []
           for i in range(nsamples):
               arr.append(self.data[i]['chan' + str(n+1)+'_' + str(cut)][imin:imax])
               arr[i] = np.reshape(arr[i],(-1, binSize )) #bin frequencies
               arr[i] = np.mean(arr[i], axis = 1)  #average the bins
   	   arr=np.array(arr)
           if(subtractMean):
               means = np.mean(arr, axis=0) #mean for each freq bin
   	       for j in range(nsamples):
   	           arr[j,:] -= means
                   arr[j,:] /=means
	       if minmax is not None:
		   vmin = -minmax
		   vmax = minmax
	       else:
	           vmin = None
	           vmax = None
               plt.imshow(arr, interpolation="nearest", vmin=vmin,vmax=vmax, aspect = "auto", extent=[fmin, fmax, nsamples*self.deltaT, 0])
           else:
                plt.imshow(arr, norm=colors.LogNorm(), interpolation="nearest" , aspect = "auto", extent=[fmin, fmax, nsamples*self.deltaT, 0]) 
           plt.colorbar()
           plt.xlabel('freq [MHz] Channel ' + str(n+1))
           plt.ylabel('time [s]')
        plt.show()


    #return image array to use as input for numpy.imshow
    #inputs:
    #       binSize [x, y]: how many samples to average per bin on both the x and y  axis
    #       cut: which cut to use
    def getImageArray(self, binSize = [1, 1], nsamples=None, cut=0 ):
        if nsamples is  None:
            nsamples = self.nSamples
        reducedArr = []  #for reduced array after binning
	for n in range(2):
           arr = []
           for i in range(nsamples):
               arr.append(self.data[i]['chan' + str(n+1)+'_' + str(cut)])
           
	       #bin along x axis (frequency bins)
               if binSize[0] > 1:
               	  arr[i] = np.reshape(arr[i],(-1, binSize[0])) 
                  arr[i] = np.mean(arr[i], axis = 1)  
           
           arr = np.array(arr) 

           #bin along y axis (time bins)
	   if binSize[1] > 1:
	       reducedArr.append([])
	       for i in range(int(len(arr)/binSize[1])):
	        	reducedArr[n].append(arr[binSize[1]*i:binSize[1]*(i+1)].mean(axis=0))
	   else:
               reducedArr = arr

        return (reducedArr)


    def nullBin(self, bin):
        for cut in [0]:
            for i, n in enumerate(self.getNames(cut)):
                self.data[n][:,bin]=0.0
                
    def getToneAmplFreq(self,chan, pm=20,freq="index"):
        mxf=[]
        mx=[]
        for line in self.data[chan]:
            i=abs(line).argmax()
            mxf.append(i)
            mx.append(line[max(0,i-pm):i+pm].sum())
        mx=np.array(mx)
        if freq=="index":
            mxf=np.array(mxf)
        elif freq=="freq" or "dfreq":
            mxf=np.array([self.freq[chan][i] for i in mxf])
            if freq=="dfreq":
                mxf-=self.freq[chan].mean()
        return mxf,mx
            
