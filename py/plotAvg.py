#!/usr/bin/env python3
import bmxdata as bmx
import matplotlib.pyplot as plt
import sys

d=bmx.BMXFile(sys.argv[1])
#print (d.data['chan1_0'][0])
#print (d.data['chan1_0'][1])
#print (d.data['chan1_1'][0].sum())
#print (d.data['chan1_1'][1].sum())
d.plotAvgSpec(0)
plt.show()
