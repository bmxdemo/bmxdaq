#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from optparse import OptionParser

parser = OptionParser()
parser.add_option("--f", "--file", dest="fname")
parser.add_option("--delays", dest="fdelays")
parser.add_option("--peaks",  dest="fpeaks")

(o, args) = parser.parse_args()

## read in file
wfile = open(o.fname)
dat_type=[('ch1','i1'),('ch2','i1')]
data=np.fromfile(wfile,dat_type)

nsamples = 10
size = 2**23
data = data['ch2']

peaks = [[], []] 
for i in range(nsamples):
  for j in range(2):
    ##find where waveform starts peaking (20 should be large enough voltage)
    a = np.where(data[(2*i+j)*size:(2*i+j+1)*size] > 20)
    ##make sure sample isn't starting in middle of peak
    if i > 0:
      b = max(data[(2*(i-1)+j+1)*size-1000:(2*(i-1)+j+1)*size])
    else:
      b = 0
    if any(map(len,a)) and  b < 10:
      peaks[j].append(np.min(a) + i*2**27)  

delayFile = open(o.fdelays, 'a')
peakFile = open(o.fpeaks, 'a')

for i in range(min(len(peaks[0]), len(peaks[1]))):
  delay = peaks[0][i] - peaks[1][i]
  delay = (1.0 * delay)/(1100000)
  delayFile.write(str(delay))
  delayFile.write(" ")

delayFile.write("\n")

for i in range(2):
  for n in peaks[i]:
    time = 1.0*n/1100000
    peakFile.write(str(time))
    peakFile.write(" ")
  peakFile.write("\n")
   

delayFile.close()
peakFile.close()
