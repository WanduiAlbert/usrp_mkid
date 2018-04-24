#!/usr/bin/env python

import sys
import numpy as np
from scipy import signal
import matplotlib.pyplot as pl

def db(x):
	return 10*np.log10(x)

def demod(z,step,decimate):
	ii = np.arange(z.size,dtype=np.uint32) * step
	phi = 2.*np.pi*(ii/(2.0**32 -1.0))
	del ii
	zlo = np.zeros(z.size,dtype=np.complex64)
	zlo.real = np.cos(phi)
	zlo.imag = np.sin(-phi)
	del phi
	z = z * zlo
	del zlo
	z = signal.decimate(z,decimate)
	cut = 150000
	z = z[cut:]
	zmu = np.mean(z)
	z *= np.exp(-1j*np.angle(zmu))
	return z

def loaddata(fn,ntone):
	rawdata = np.fromfile(fn,dtype=np.int16)[2*(1+2*ntone):]
	z = np.zeros(rawdata.size/2,dtype=np.complex64)
	z.real = rawdata[::2]
	z.imag = rawdata[1::2]
	del rawdata
	return z

def main():
	fn = sys.argv[1]

	decimate = 16
	Fs = 0.5e6 / decimate

	ntone = np.fromfile(fn,dtype=np.uint32,count=1)[0]
	print ntone
	words = np.fromfile(fn,dtype=np.float32,count=1+2*ntone)[1:]
	amps = words[::2]
	words = np.fromfile(fn,dtype=np.uint32,count=1+2*ntone)[2:]
	steps = words[::2]
	print amps
	print steps
	
	z = loaddata(fn,ntone)
	z = z[1000000:60000000]
	
	print "max:",np.max(z.real),np.max(z.imag)
	print "min:",np.min(z.real),np.min(z.imag)
	print "mean:",np.mean(z.real),np.mean(z.imag)
	print "std:",np.std(z.real),np.std(z.imag)
	
	z1 = demod(z,steps[0],decimate)
	z1 = signal.decimate(z1,16)
	z1 = z1[50:]
	z1 = signal.decimate(z1,16)
	z1 = z1[50:]
	pl.plot(np.abs(z1))
	pl.show()
	exit()

	z2 = demod(z,steps[1],decimate)
	z1 /= np.mean(z1)
	z2 /= np.mean(z2)

	dt = 1.0 / Fs
	t = np.arange(z1.size)*dt

	z1 -= np.median(z1)
	z2 -= np.median(z2)

	'''
	pl.plot(z1.real)
	pl.plot(z1.imag)
	pl.show()
	exit()
	'''

	'''
	pl.hist(np.abs(z1),bins=50,log=True)
	pl.hist(np.abs(z2),bins=50,log=True)
	pl.show()
	exit()
	'''

	'''
	pl.plot(t,np.abs(z1),label='on res')
#	pl.plot(t,np.abs(z2),label='off res')
	pl.xlabel('Time (s)')
	pl.grid()
	pl.ylabel('|S21| (ADC)')
	pl.legend()
	pl.title('Unreleased bolometer glitches')
	pl.savefig('glitch3.png')
	pl.show()
	exit()
	'''

	def clip(f,p):
		maxf = 1e5
		ok = f < maxf
		f = f[ok]
		p = p[ok]
		return f,p
		
	NFFT = int(2**17)
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

	df = freqs[1]-freqs[0]
	fmin = 29.0
	fmax = 31.0
	ok = (fmin < freqs) & (freqs < fmax)
	w = np.sqrt(df*np.sum(Pxxim[ok]))
	vdc = 8.
	vchop = 0.1
	rb = 2e6
	rh = 0.073
	p0 = (vdc/rb)**2 * rh
	print "p0: ",p0
	pmax = ((vdc+vchop/2.)/rb)**2 * rh
	pmin = ((vdc-vchop/2.)/rb)**2 * rh
	ppp = pmax - pmin
	prms = ppp / (2.*np.sqrt(2.))
	print prms
	print w
	gain = w/prms

	pl.figure()
	pl.plot(freqs,10*np.log10(Pxxre),label='Re 1',color='blue')
	pl.plot(freqs,10*np.log10(Pxxim),label='Im 1',color='green')
	pl.plot(freqs,10*np.log10(Pxxre2),label='Re 2',color='blue',linestyle='--')
	pl.plot(freqs,10*np.log10(Pxxim2),label='Im 2',color='green',linestyle='--')

	pl.ylim(-120,-40)
	pl.gca().set_xscale('log')
	pl.grid()
	pl.legend(loc='lower left')
	pl.xlabel('Frequency (Hz)')
	#pl.ylabel('PSD (dB ADC)')
	pl.ylabel('PSD (dBc/Hz)')

	pl.figure()
	pl.plot(freqs,1e18*np.sqrt(Pxxim)/gain)
	pl.xlabel('Frequency (Hz)')
	pl.ylabel('NEP (aW/rtHz)')
	pl.grid()
	pl.gca().set_xscale('log')
	pl.gca().set_yscale('log')

	pl.show()


if __name__=='__main__':
	main()
