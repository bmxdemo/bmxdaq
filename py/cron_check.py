#!/usr/bin/env python
import time, psutil, datetime, glob, os, bmxdata

logfile='/home/bmx/bmxdaq/log/daq.log'
## let's try to read the last line of log and see where we stand
last_alive=False
if os.path.isfile(logfile):
    lastline=open(logfile).readlines()[-1]
    if "ALIVE" in lastline:
        last_alive=True
## see if alive now
alive=False
for proc in psutil.process_iter():
    if 'daq.out' in proc.name():
        alive=True
        break

outline= datetime.datetime.now().strftime("%Y-%m-%d %H.%M :")

outline += " ALIVE" if alive else " DEAD"
if (last_alive and not alive):
    ## we have died since last hour
    print ("WE HAVE DIED, DO SOMETHING")

if alive:
    ## find the last .new file
    fnamet=max(glob.iglob('/home/bmx/bmxdaq/data/*.data.new'), key=os.path.getmtime)
    dmtime=abs(os.path.getmtime(fnamet)-time.time())
    sname=fnamet.replace("/home/bmx/bmxdaq/data/","")
    if dmtime<3:
        outline += " %s"%(sname)
        d=bmxdata.BMXFile(fnamet, nsamples=10)
        for ch in range(1,5):
            name='chan%i_0'%ch
            pow=d.data[name].mean()/1e12
            outline += " CH%i: %f3.2"%(ch,pow)
            if (pow<1):
                print ("CH%i, power too low!"%ch)
                ## do something
    else:
        outline+=" not recording"
open (logfile,'a+').write(outline+"\n")
