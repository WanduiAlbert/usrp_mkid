#!/usr/bin/env python

import sys, os
import numpy as np
import matplotlib.pyplot as pl
from scipy import signal, optimize, fftpack
pi = np.pi

sample_rate = 100.0e6
sourcefn = sys.argv[1]
measfn = sys.argv[2]

sre,sim = np.loadtxt(sourcefn,dtype=np.int16,skiprows=1).T
ns = sre.size
chirp_rate = sample_rate / ns
sz = sre+sim*1.0j

def loadmeas(measfn):
	d = np.fromfile(measfn,dtype=np.int16)
	mre,mim = d[::2],d[1::2]
	print "std(re): ",np.std(mre)
	print "std(im): ",np.std(mim)
	print "max(re),min(re): ",np.max(mre),np.min(mre)
	print "max(im),min(im): ",np.max(mim),np.min(mim)

	#pl.plot(mre[:20000])
	#pl.plot(mim[:20000])
	#pl.show()
	#exit()

	ratio = np.zeros(mre.shape,dtype=np.complex64)
	ratio.real = mre
	ratio.imag = mim
	return ratio

ratio = loadmeas(measfn)

nt = ratio.size
nchunk = (nt / ns) - 10
nt2 = nchunk * ns
ratio = ratio[nt-nt2:]
ratio = ratio.reshape((nchunk,ns))

fs = np.fft.fftshift(np.fft.fftfreq(ns,d=1.0/sample_rate))
szf = np.fft.fftshift(np.fft.fft(sz))

#mzf = np.fft.fftshift(np.fft.fft(mz,axis=1),axes=1)
ratio[:,:] = np.fft.fftshift(fftpack.fft(ratio,axis=1),axes=1)

ratio[:,1:] /= szf[np.newaxis,1:]
ratio[:,0] = 0

do_cal = False
if do_cal:
	ratio_mean = np.mean(ratio[10:,:],axis=0)
	print ratio.shape
	print ratio_mean.shape
	np.save('cal.npy',ratio_mean)
	exit()

calfn = 'cal.npy'
if os.path.exists(calfn):
	cal = np.load('cal.npy')
	print "cal.shape: ",cal.shape
	print "ratio.shape: ",ratio.shape
	ratio /= cal[np.newaxis,:]
else:
	print 'no calibration found'

clip = 64
#fs += 314.3e6
#fs = fs[clip:-clip]
#ratio = ratio[:,clip:-clip]
ratio[:,-clip:] = 0
ratio[:,:clip] = 0

ratio_mean = np.mean(ratio,axis=1)
angle_mean = np.unwrap(np.angle(ratio_mean))
ii = np.arange(ratio.shape[0])
phase = np.polyval(np.polyfit(ii,angle_mean,1),ii)
phasecorr = np.exp(-1j*phase)
ratio *= phasecorr[:,np.newaxis]

ratio_mean = np.mean(ratio,axis=0)
angle_mean = np.unwrap(np.angle(ratio_mean))
phase = np.polyval(np.polyfit(fs,angle_mean,1),fs)
phasecorr = np.exp(-1j*phase)
ratio *= phasecorr

ratio_mean = np.mean(ratio,axis=1)
angle_mean = np.unwrap(np.angle(ratio_mean))
ii = np.arange(ratio.shape[0])
phase = np.polyval(np.polyfit(ii,angle_mean,1),ii)
phasecorr = np.exp(-1j*phase)
ratio *= phasecorr[:,np.newaxis]

ratio_mean = np.mean(ratio,axis=0)
angle_mean = np.unwrap(np.angle(ratio_mean))

pl.imshow(np.abs(ratio),aspect='auto',interpolation='nearest')
#pl.imshow(np.angle(ratio),aspect='auto',interpolation='nearest')
pl.show()
exit()

#pl.plot(fs,np.abs(ratio_mean))
#pl.show()
#exit()


#### Frequency template construction

tre = ratio_mean.real
tim = ratio_mean.imag
tre -= np.mean(tre)
tim -= np.mean(tim)
tre *= np.hamming(tre.size)
tim *= np.hamming(tim.size)

ratio -= np.mean(ratio,axis=1)[:,np.newaxis]
ratio *= np.hamming(tre.size)[np.newaxis,:]

rf = fftpack.fft(ratio,axis=1)
tf = np.fft.fft(tre + tim*1.0j)

rf *= np.conjugate(tf)[np.newaxis,:]
nt = tf.size

df = fs[1] - fs[0]
print rf.dtype
ratio[:,:] = fftpack.ifft(rf,axis=1)
corr = np.fft.fftshift(np.abs(ratio))
ts = np.argmax(corr,axis=1)
ts2 = np.zeros(ts.size)

ii = np.arange(5) - 2
for i in range(corr.shape[0]):
	c = corr[i]
	t = ts[i]
	a,b,c = np.polyfit(ii,c[t-2:t+3],2)
	ts2[i] = -b/(2*a) + t

ts *= df
ts2 *= df

ts = ts - np.mean(ts)
ts2 = ts2 - np.mean(ts2)

def gaincal(fs,rmssig,vdc,vchop):
	rb = 0.854e6
	rh = 0.106
	p0 = (vdc/rb)**2 * rh
	print "p0: ",p0*1e12,"pW"
	pmax = ((vdc+vchop/2.)/rb)**2 * rh
	pmin = ((vdc-vchop/2.)/rb)**2 * rh
	ppp = pmax - pmin
	prms = ppp / (2.*np.sqrt(2.))
	print "prms: ",prms*1e12,"pW"
	gain = rmssig/prms
	return gain

def model(t,A,B,f,dc):
	phi = 2.0*pi*t*f
	return A * np.cos(phi) + B * np.sin(phi) + dc

vdc = 7.0
vchop = 0.02
t = np.arange(ts2.size) / chirp_rate
ftone = 5.0

c = np.cos(2*pi*t*ftone)
s = np.sin(2*pi*t*ftone)

c0 = 2*np.mean(c*ts2)
s0 = 2*np.mean(s*ts2)

p0 = [c0,s0,ftone,np.mean(ts2)]
popt,pcov = optimize.curve_fit(model,t[:1000],ts2[:1000],p0)
popt,pcov = optimize.curve_fit(model,t,ts2,popt)


A,B,_,_ = popt
rmssig = np.sqrt(A*A+B*B)

ym = model(t,*popt)

print ts2.shape
print 'before: ',np.std(ts2)
ts2 = ts2 - ym
print 'after: ',np.std(ts2)

fs,psd = signal.welch(ts2,fs=chirp_rate,detrend='constant',scaling='density',nperseg=8192)


asd = np.sqrt(psd)
pl.plot(fs,asd)
pl.gca().set_xscale('log')
pl.gca().set_yscale('log')
pl.ylabel('NEF (Hz/rtHz)')
pl.grid()

pl.figure()
#gain = gaincal(fs,rmssig,vdc,vchop)
gain = 1.056e17
print "gain: ",gain,"Hz/W"
psd /= (gain*gain)
asd = np.sqrt(psd) * 1e18
pl.plot(fs,asd)
pl.gca().set_xscale('log')
pl.gca().set_yscale('log')
pl.ylabel('NEP (aW/rtHz)')
pl.grid()

pl.show()
exit()

#pl.plot(np.fft.fftshift(corr[0]))
pl.imshow(np.fft.fftshift(corr),interpolation='nearest',aspect='auto')
pl.grid()
pl.show()

exit()

pl.figure()
pl.subplot(211)
pl.plot(1e-6*fs,np.abs(ratio_mean))
pl.ylabel('|S21|')
pl.title('Mean S21')
pl.grid()

pl.subplot(212)
pl.plot(1e-6*fs,np.angle(ratio_mean)*180/np.pi)
pl.ylabel('angle(S21) (deg)')
pl.grid()
pl.xlabel('Frequency (MHz)')
pl.savefig(measfn+'_mean.png')

extent = (1e-3*np.min(fs),1e-3*np.max(fs),nchunk,0)

pl.figure()
pl.xlabel('Frequency (kHz)')
pl.ylabel('Sweep')
pl.title('Waterfall |S21|')
pl.imshow(np.abs(ratio[:100,:]),interpolation='nearest',extent=extent)
pl.colorbar().set_label('|S21|')
pl.gca().set_aspect('auto')
pl.title('abs')
pl.savefig(measfn+'_waterfall_abs.png')

pl.figure()
pl.xlabel('Frequency (kHz)')
pl.ylabel('Sweep')
pl.title('Waterfall angle(S21)')
pl.imshow(np.unwrap(np.angle(ratio[:100,:]),axis=1)*180/np.pi,interpolation='nearest',extent=extent)
pl.colorbar().set_label('angle(S21) (Deg)')
pl.gca().set_aspect('auto')
pl.savefig(measfn+'_waterfall_angle.png')
pl.show()

