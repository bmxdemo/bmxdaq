#!/usr/bin/env python
import bmxdata as bmx
import matplotlib.pyplot as plt
import sys
from optparse import OptionParser

parser = OptionParser()

parser.add_option("-s", "--show", dest="show", default=False,
                  action="store_true", help="call show at the end")
parser.add_option("-p", "--pdf", dest="pdf", default=None,
                  help="save to pdf", metavar="filename")
parser.add_option("--log", dest="log", default=False,
                  action="store_true", help="Log scale")
parser.add_option("--cut", dest="cut", default=0,type='int',
                  help="which cutout", metavar="cutout")

parser.add_option("--null", dest="null", default=None,type='int',
                  help="null a certain bin", metavar="cutout")

(o, args) = parser.parse_args()

d=bmx.BMXFile(args[0])
if (o.null):
    d.nullBin(o.null)
    
d.plotAvgSpec(o.cut)

if o.log:
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.semilogy()

if o.pdf:
    pylab.savefig(o.pdf)

if o.show:
    pylab.show()

plt.show()


