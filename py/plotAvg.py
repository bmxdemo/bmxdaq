#!/usr/bin/env python3
import bmxdata as bmx
import matplotlib.pyplot as plt
import sys

d=bmx.BMXFile(sys.argv[1])
d.freq-=250.0
d.plotAvgSpec()
plt.show()
