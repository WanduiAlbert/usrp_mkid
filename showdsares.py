#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as pl
from scipy import signal, optimize

def gaincal(fs,psd,vdc,vchop):
	df = fs[1]-fs[0]
	fmin = 28.
	fmax = 32.
	ok = (fmin < fs) & (fs < fmax)
	w = np.sqrt(df*np.sum(psd[ok]))
	rb = 0.851e6
	rh = 0.073
	p0 = (vdc/rb)**2 * rh
	print "p0: ",p0*1e12,"pW"
	pmax = ((vdc+vchop/2.)/rb)**2 * rh
	pmin = ((vdc-vchop/2.)/rb)**2 * rh
	ppp = pmax - pmin
	prms = ppp / (2.*np.sqrt(2.))
	print "prms: ",prms*1e12,"pW"
	gain = w/prms
	psd /= gain**2
	return psd,gain

NFFT = 512
resultsfn = sys.argv[1]
results = np.loadtxt(resultsfn)

#for i in range(results.shape[0]):
#	if np.isnan(results[i,0]):
#		results[i,:] = results[i-1,:]

results -= np.mean(results,axis=0)
results /= np.std(results,axis=0)

sample_rate = 2e6/1024.
nt = results.shape[0]
t = np.arange(nt)/sample_rate

def model(t,A,B,dc,f):
	phi = 2.*np.pi*f*t
	return dc+A*np.cos(phi)+B*np.sin(phi)

y = results[:,0]
f0 = 30.0
p0 = (0.,0.,np.mean(y),f0)
popt,pcov = optimize.curve_fit(model,t[:400],y[:400],p0=p0)
popt,pcov = optimize.curve_fit(model,t,y,p0=popt)
ym = model(t,*popt)
ym /= np.sqrt(np.dot(ym,ym))

gains = np.zeros(results.shape[1])
results_clean = results + 0.0
for i in range(results.shape[1]):
	gains[i] = np.dot(results[:,i],ym)
	results_clean[:,i] -= ym*gains[i]

cov = np.cov(results_clean.T)

'''
pl.imshow(cov,interpolation='nearest')
pl.colorbar()
pl.show()
exit()
'''

covi = np.linalg.inv(cov)
cal = np.dot(covi,gains)
template = np.dot(results,cal)

results2 = np.zeros((results.shape[0],results.shape[1]+1))
results2[:,1:] = results
results2[:,0] = template
results = results2

'''
pl.figure()
pl.psd(y-model(t,*popt))
pl.show()
exit()
'''



ares = []
vdc = 5.0
vchop = 0.10
for i in range(results.shape[1]):
	fs,psd = signal.welch(results[:,i],nperseg=NFFT,fs=sample_rate)
	psd,gain = gaincal(fs,psd,vdc,vchop)
	results[:,i] /= gain
	asd = np.sqrt(psd)*1e18
	ares.append(asd)

#names = 'f0 Qr A B'.split()
names = 'template f0 Qr A B Qe_re Qe_im D'.split()


pl.figure()
for i in range(1,2):
	pl.plot(fs,ares[i],label=names[i])
pl.gca().set_xscale('log')
pl.gca().set_yscale('log')
pl.xlabel('Frequency (Hz)')
pl.ylabel('NEP (aW/rtHz)')
pl.legend()
pl.grid()

pl.savefig('fit_psd01.png')

pl.figure()
nshow = results.shape[0]
t = np.arange(nshow)/sample_rate
for i in range(1,2):
	pl.plot(t,results[:nshow,i],label=names[i])
pl.title('Fit timestreams')
pl.grid()
pl.xlabel('Time (seconds)')
pl.ylabel('Fit timestream')
pl.legend()
pl.savefig('fit_ts01.png')

pl.show()

