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

sample_rate = 1e6/1024.
nt = results.shape[0]
t = np.arange(nt)/sample_rate

def logsweepasd(fin):
	T = 1.0
	nt = int(T * sample_rate)
	t = np.arange(nt)/sample_rate
	f1 = 3.0
	f2 = 500.0
	lnf = np.log(f2/f1)

	phase = (2.*np.pi*f1*T/lnf) * np.exp(lnf*t/T)-1.
	xt = np.sin(phase)
	asd = np.abs(np.fft.fft(xt))[:nt/2]
	fs = np.fft.fftfreq(nt,d=1./sample_rate)[:nt/2]
	asdout = np.interp(fin,fs,asd)
	return asdout



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


def thermal_model(f,A,f3db):
	return np.log10(A / np.sqrt(1.0+(f/f3db)**2))

ares = []
for i in range(results.shape[1]):
	fs,psd = signal.welch(results[:,i],nperseg=NFFT,fs=sample_rate)
	asd_ref = logsweepasd(fs)
	psd /= asd_ref*asd_ref
	asd = np.sqrt(psd)
	ares.append(asd)

#names = 'f0 Qr A B'.split()
names = 'template f0 Qr A B Qe_re Qe_im D'.split()


pl.figure()
for i in range(1,2):
	y = ares[i]
	ly = np.log10(y)
	p0 = (y[0],30.0)
	#thermal_model(f,A,f3db)
	popt,pcov = optimize.curve_fit(thermal_model,fs,ly,p0=p0)
	ym = 10**thermal_model(fs,*popt)
	print "f3db: ",popt[1]
	print "A: ",popt[0]
	pl.plot(fs,ares[i],label=names[i])
	pl.plot(fs,ym,label=names[i]+' fit f3db=%.1fHz'%popt[1])
pl.gca().set_xscale('log')
pl.gca().set_yscale('log')
pl.xlabel('Frequency (Hz)')
pl.ylabel('ASD (ADC)')
pl.legend()
pl.title('Time constant 313.5MHz Vdc=5V 0.851Mohm txgain=10dB')
pl.grid()

pl.savefig('timeconstant.png')

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

