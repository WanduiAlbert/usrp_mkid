#!/usr/bin/env python

import argparse
import numpy as np
from numpy import pi
import matplotlib.pyplot as pl

def getargs():
	parser = argparse.ArgumentParser()
	parser.add_argument('fn')
	parser.add_argument('-n',default=32768,type=int)
	args = parser.parse_args()
	return args

def main():
	args = getargs()

	n = args.n
	fn = args.fn

	sample_rate = 1e6
	f0 = -0.5*sample_rate
	f1 = 0.5*sample_rate
	T = n / sample_rate
	t = np.linspace(0,T,n)

	k = (f1 - f0)/T
	phit = 2.0*np.pi*(f0*t+(k/2.)*t*t)
	zs = np.exp(1.0j*phit)

	ntaper = n/16
	nrest = n - 2*ntaper
	#taper = np.linspace(0,1,ntaper)
	taper = 0.5*(1+np.cos(np.linspace(0,np.pi,ntaper))[::-1])
	ones = np.ones(nrest)
	window = np.hstack((taper,ones,taper[::-1]))

	zs *= window
	
	zs *= 32000.0;
	zsr = np.round(zs.real).astype(np.int16)
	zsi = np.round(zs.imag).astype(np.int16)

	zsri = np.vstack((zsr,zsi)).T
	np.savetxt(fn,zsri,header=str(n),fmt='%d',comments='')

	zs = zsr+1.j*zsi
	zf = np.fft.fft(zs)
	fs = np.fft.fftshift(np.fft.fftfreq(n,d=1.0/sample_rate))

	pl.figure()
	pl.plot(zs.real,label='real')
	pl.plot(zs.imag,label='imag')
	pl.xlabel('Time (nsample)')
	pl.ylabel('Signal (DAC)')
	pl.legend()
	pl.grid()

	pl.figure()
	pl.plot(fs*1e-6,np.abs(np.fft.fftshift(zf)))
	pl.xlabel('Frequency (MHz)')
	pl.ylabel('ASD')
	pl.grid()
	pl.show()

if __name__=='__main__':
	main()
