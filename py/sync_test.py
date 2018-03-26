#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from optparse import OptionParser
import sys

parser = OptionParser()
parser.add_option("--f", "--file", dest="fname")
parser.add_option("--delays", dest="fdelays", default = "std")
parser.add_option("--peaks1",  dest="fpeaks1", default = "std")
parser.add_option("--peaks2",  dest="fpeaks2", default = "std")
parser.add_option("--ch", "--channel", dest ="channel", default = 'ch2')
parser.add_option("--nsamples", dest="nsamples", default = 12, type="int")
parser.add_option("--size", dest = "size", default = 2**27, type="int")

(o, args) = parser.parse_args()

## read in file
wfile = open(o.fname)
dat_type=[('ch1','i1'),('ch2','i1')]
data=np.fromfile(wfile,dat_type)

data = data[o.channel]

peaks = [[], []] 
for i in range(o.nsamples):
  for j in range(2):
    ##find where waveform starts peaking (20 should be large enough voltage)
    a = np.where(data[(2*(i)+j)*o.size:(2*(i)+j+1)*o.size] > 20)
    ##make sure sample isn't starting in middle of peak
    ##if i > 0:
    ##  b = max(data[(2*(i)+j+1)*o.size-100000:(2*(i)+j+1)*o.size])
    ##else:
    ##  b = 0
    b = 0
    if any(map(len,a)) and  b < 20:
      peaks[j].append(np.min(a) + i*o.size)  

if o.fdelays == "std":
  delayFile = sys.stdout
else:
  delayFile = open(o.fdelays, 'a')
if o.fpeaks1 == "std":
  peakFile = [sys.stdout, sys.stdout]
else:
  peakFile = [open(o.fpeaks1, 'a'), open(o.fpeaks2, 'a')]

for i in range(min(len(peaks[0]), len(peaks[1]))):
  delay = peaks[0][i] - peaks[1][i]
  delay = (1.0 * delay)/(1100000)
  delayFile.write(str(delay) + " ")

delayFile.write("\n")

for i in range(2):
  for n in peaks[i]:
    time = 1.0*n/1100000
    peakFile[i].write(str(time) + " ")
  peakFile[i].write("\n")
   

delayFile.close()
peakFile[0].close()
peakFile[1].close()
