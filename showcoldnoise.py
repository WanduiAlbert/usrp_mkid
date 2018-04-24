
import sys
import numpy as np
import matplotlib.pyplot as pl

fn = sys.argv[1]

t,temp,psd = np.loadtxt(fn).T
ok = (temp > 1.4)&(temp < 3.7)
t = t[ok]
temp = temp[ok]
psd = psd[ok]

g,b = np.polyfit(psd,temp,1)
T0 = -b

tempfit = np.polyval((g,b),psd)
resid = temp - tempfit

print "temp rate: ",(temp[-1]-temp[0])/(t[-1]-t[0]),"K/s"
print "T0: ",T0
print "gain: ",1/g,"ADC^2/K"

pl.plot(psd,tempfit,label='fit')
pl.scatter(psd,temp,label='measured')
print "resid rms: ",np.std(resid),"K"
pl.grid()
pl.xlabel('Received power')
pl.ylabel('Cold stage Temperature (K)')
pl.title('Receiver noise temperature %0.2f K'%T0)
pl.savefig(fn+'.jpg')
pl.show()

