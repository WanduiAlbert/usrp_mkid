#!/usr/bin/env python

import sys
import numpy as np
from scipy import signal
import matplotlib.pyplot as pl

def db(x):
	return 10*np.log10(x)

def demod(z,step,decimate):
	ii = np.arange(z.size,dtype=np.uint32) * step
	phi = 2.*np.pi*(ii / (2.0**32 - 1.0))
	zlo = np.exp(-1j*phi)
	z = z * zlo
	z = signal.decimate(z,decimate)
	cut = 20000
	z = z[cut:]
	zmu = np.mean(z)
	#z /= zmu #np.exp(-1j*np.angle(zmu))
	z *= np.exp(-1j*np.angle(zmu))
	return z

def glitchsearch(y):
	y = np.abs(y)
	y = y - np.median(y)
	scale = np.std(y)
	thresh = 10*scale
	above_thresh = y > thresh
	diff = np.diff(0.+above_thresh)
	starts = np.argwhere(diff > 0)
	stops = np.argwhere(diff < 0)
	if stops[0] < starts[0]:
		stops = stops[1:]

	if starts[-1] > stops[-1]:
		starts = starts[:-1]

	ws = []
	hs = []
	for i in range(len(starts)):
		start = starts[i]
		stop = stops[i]
		yc = y[start:stop]
		w = yc.size
		h = np.max(yc)
		ws.append(w)
		hs.append(h)
	ws = np.array(ws)
	hs = np.array(hs)

	u = np.zeros(y.size)
	for i in range(len(starts)):
		i0 = starts[i]
		i1 = i0 + ws[i]
		u[i0:i1] = hs[i]

	pl.scatter(ws*16.0,hs)
	pl.title('Resonator glitches in 60 seconds')
	pl.xlabel('Width (microseconds)')
	pl.ylabel('Height (DAC)')
	pl.grid()
	pl.savefig('glitchscatter.png')
	pl.show()
	exit()

def main():
	fn = sys.argv[1]

	decimate = 16
	Fs = 1.0e6 / decimate

	ntone = np.fromfile(fn,dtype=np.uint32,count=1)[0]
	print ntone
	words = np.fromfile(fn,dtype=np.float32,count=1+2*ntone)[1:]
	amps = words[::2]
	words = np.fromfile(fn,dtype=np.uint32,count=1+2*ntone)[2:]
	steps = words[::2]
	print amps
	print steps
	rawdata = np.fromfile(fn,dtype=np.int16)[2*(1+2*ntone):]

	re = rawdata[::2]*1.0
	im = rawdata[1::2]*1.0

#	re = re[4*103200:4*104000]
#	im = im[4*103200:4*104000]

	#re = re*re
	#im = im*im
	
	z = re + 1.j*im
	z = z[1000000:]
	
	print "max:",np.max(re),np.max(im)
	print "min:",np.min(re),np.min(im)
	print "mean:",np.mean(re),np.mean(im)
	print "std:",np.std(re),np.std(im)
	
	z1 = demod(z,steps[0],decimate)
	z2 = demod(z,steps[1],decimate)
	#z1 /= np.mean(z1)
	#z2 /= np.mean(z2)

	dt = 1.0 / Fs
	t = np.arange(z1.size)*dt

	z1 -= np.median(z1)
	z2 -= np.median(z2)

	glitchsearch(z1)
	exit()

	pl.plot(t,np.abs(z1),label='on res')
	pl.plot(t,np.abs(z2),label='off res')
	pl.xlabel('Time (s)')
	pl.grid()
	pl.ylabel('|S21| (ADC)')
	pl.legend()
	pl.title('Unreleased bolometer glitches')
	pl.savefig('glitch3.png')
#260.5352
	pl.show()
	exit()

	def clip(f,p):
		maxf = 1e5
		ok = f < maxf
		f = f[ok]
		p = p[ok]
		return f,p
		
	NFFT = int(2**19)
	print "nt:",z1.size*1e-6,"Msamples"
	print "nfft:",NFFT*1e-6,"MSamples"
	
	detrend = 'linear'
	freqs,Pxxre = signal.welch(z1.real,nperseg=NFFT,fs=Fs,detrend=detrend,scaling='density')
	freqs,Pxxre = clip(freqs,Pxxre)
	freqs,Pxxim = signal.welch(z1.imag,nperseg=NFFT,fs=Fs,detrend=detrend,scaling='density')
	freqs,Pxxim = clip(freqs,Pxxim)

	freqs,Pxxre2 = signal.welch(z2.real,nperseg=NFFT,fs=Fs,detrend=detrend,scaling='density')
	freqs,Pxxre2 = clip(freqs,Pxxre2)
	freqs,Pxxim2 = signal.welch(z2.imag,nperseg=NFFT,fs=Fs,detrend=detrend,scaling='density')
	freqs,Pxxim2 = clip(freqs,Pxxim2)

	'''
	df = freqs[1]-freqs[0]
	fmin = 29.0
	fmax = 31.0
	ok = (fmin < freqs) & (freqs < fmax)
	w = np.sqrt(df*np.sum(Pxxim[ok]))
	vdc = 5.
	vchop = 0.1
	rb = 2e6
	rh = 0.073
	p0 = (vdc/rb)**2 * rh
	pmax = ((vdc+vchop/2.)/rb)**2 * rh
	pmin = ((vdc-vchop/2.)/rb)**2 * rh
	ppp = pmax - pmin
	prms = ppp / (2.*np.sqrt(2.))
	print prms
	print w
	gain = w/prms
	'''

	pl.plot(freqs,10*np.log10(Pxxre),label='Re 1',color='blue')
	pl.plot(freqs,10*np.log10(Pxxim),label='Im 1',color='green')
	pl.plot(freqs,10*np.log10(Pxxre2),label='Re 2',color='blue',linestyle='--')
	pl.plot(freqs,10*np.log10(Pxxim2),label='Im 2',color='green',linestyle='--')
	#pl.psd(z1.imag,NFFT=NFFT,Fs=Fs)
	#pl.psd(z2.imag,NFFT=NFFT,Fs=Fs)

	pl.ylim(-40,5)
	pl.gca().set_xscale('log')
	pl.grid()
	pl.legend(loc='lower left')
	pl.xlabel('Frequency (Hz)')
	pl.ylabel('PSD (dB ADC)')
	#pl.ylabel('PSD (dBc/Hz)')
	pl.show()


if __name__=='__main__':
	main()
