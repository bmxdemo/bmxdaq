#!/usr/bin/env python3
import bmxdata as bmx
import numpy as np
import pickle

if False:
    flist="""data/161116_1603.tone.data
    data/161116_1800.tone.data
    data/161116_1900.tone.data
    data/161116_2000.tone.data
    data/161116_2100.tone.data
    data/161116_2200.tone.data
    data/161116_2300.tone.data
    data/161117_0000.tone.data
    data/161117_0100.tone.data
    data/161117_0200.tone.data
    data/161117_0300.tone.data
    data/161117_0400.tone.data
    data/161117_0500.tone.data
    data/161117_0600.tone.data
    data/161117_0700.tone.data
    data/161117_0800.tone.data
    data/161117_0900.tone.data
    data/161117_1000.tone.data
    data/161117_1100.tone.data
    data/161117_1200.tone.data
    data/161117_1300.tone.data
    data/161117_1400.tone.data
    data/161117_1500.tone.data
    data/161117_1600.tone.data
    data/161117_1700.tone.data
    data/161117_1800.tone.data
    data/161117_1900.tone.data
    data/161117_2000.tone.data
    data/161117_2100.tone.data
    data/161117_2200.tone.data
    data/161117_2300.tone.data
    data/161118_0000.tone.data
    data/161118_0100.tone.data
    data/161118_0200.tone.data
    data/161118_0300.tone.data
    data/161118_0400.tone.data
    data/161118_0500.tone.data
    data/161118_0600.tone.data
    data/161118_0700.tone.data
    data/161118_0800.tone.data
    data/161118_0900.tone.data
    data/161118_1000.tone.data
    data/161118_1100.tone.data
    data/161118_1200.tone.data
    data/161118_1300.tone.data
    data/161118_1400.tone.data
    data/161118_1500.tone.data
    data/161118_1600.tone.data
    data/161118_1700.tone.data
    data/161118_1800.tone.data
    data/161118_1900.tone.data
    data/161118_2000.tone.data
    data/161118_2100.tone.data
    data/161118_2200.tone.data
    data/161118_2300.tone.data
    data/161119_0000.tone.data
    data/161119_0100.tone.data
    data/161119_0200.tone.data
    data/161119_0300.tone.data
    data/161119_0400.tone.data
    data/161119_0500.tone.data
    data/161119_0600.tone.data
    data/161119_0700.tone.data
    data/161119_0800.tone.data
    data/161119_0900.tone.data
    data/161119_1000.tone.data
    data/161119_1100.tone.data
    data/161119_1200.tone.data
    data/161119_1300.tone.data
    data/161119_1400.tone.data
    data/161119_1500.tone.data
    data/161119_1600.tone.data
    data/161119_1700.tone.data
    data/161119_1800.tone.data
    data/161119_1900.tone.data
    data/161119_2000.tone.data
    data/161119_2100.tone.data
    data/161119_2200.tone.data
    data/161119_2300.tone.data
    data/161120_0000.tone.data
    data/161120_0100.tone.data
    data/161120_0200.tone.data
    data/161120_0300.tone.data
    data/161120_0400.tone.data
    data/161120_0500.tone.data
    data/161120_0600.tone.data
    data/161120_0700.tone.data
    data/161120_0800.tone.data
    data/161120_0900.tone.data
    data/161120_1000.tone.data
    data/161120_1100.tone.data
    data/161120_1200.tone.data
    data/161120_1300.tone.data
    data/161120_1400.tone.data
    data/161120_1500.tone.data
    data/161120_1600.tone.data
    data/161120_1700.tone.data
    data/161120_1800.tone.data
    data/161120_1900.tone.data
    data/161120_2000.tone.data
    data/161120_2100.tone.data
    data/161120_2200.tone.data
    data/161120_2300.tone.data""".split()
    outfn="data.1amp2tone.npz"
if True:
    flist="""data/161115_2206.tone.data
data/161116_0000.tone.data
data/161116_0100.tone.data
data/161116_0200.tone.data
data/161116_0300.tone.data
data/161116_0400.tone.data
data/161116_0500.tone.data
data/161116_0600.tone.data
data/161116_0700.tone.data
data/161116_0800.tone.data
data/161116_0900.tone.data""".split()
    outfn="data.1tone2tone.npz"




dtype=[('freq','f4'),('chan1','f4'),('chan2','f4'),('chanXR','f4'),('chanXI','f4')]
fulldata=[]

for fn in flist[:]:
    print(fn)
    d=bmx.BMXFile(fn)
    d.freq-=250.0
    d.freq*=1000

    for i,n in enumerate(d.names):
        mxf=[]
        mx=[]
        print(len(d.data))
        for line in d.data[n]:
            o=abs(line).argmax()
            mxf.append(d.freq[o])
            mx.append(line[max(0,o-20):o+20].sum())
        if (i==0):
            cdata=np.zeros(len(mxf),dtype=dtype)
            cdata["freq"]=np.array(mxf)
        cdata[n]=np.array(mx)
    fulldata.append(cdata)

fulldata=np.hstack(fulldata)
print(len(fulldata))
np.save(outfn,fulldata)

