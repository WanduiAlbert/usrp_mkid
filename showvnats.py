#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as pl
from vnadata import VNAData

def db(x):
	return 10*np.log10(x)

def load_mag(fn):
	d = VNAData(fn)
	re = d.z.real[0]
	im = d.z.imag[0]
	print d.z.shape
	exit()
	pl.plot(re)
	pl.plot(im)
	pl.show()
	exit()
	print re.dtype
	fs = d.fs
	mag = np.sqrt(re*re+im*im)
	dbmag = db(mag)
	phi = np.arctan2(im,re)
	phi = np.unwrap(phi)
	p = np.polyfit(fs,phi,1)
	print "p: ",p
	print "time delay: ",1e6*p[0] / (2 * np.pi),"us"
	phi = phi - np.polyval(p,fs)
	return fs,dbmag,phi*180./np.pi

def main():
	fn = sys.argv[1]
	calfn = 'vna_cal.dat'

	#fs,cal = load_mag(calfn)
	fs,mag,phase = load_mag(fn)
	#mag -= cal
	
	imin = np.argmin(mag)
	fmin = fs[imin]*1e-6
	print "Minimum frequency: %.6f MHz"%fmin
	print "minmag: ",mag[imin]
	print "maxmag: ",np.max(mag)
	dphidf = np.gradient(phase)
	
	imax = np.argmax(np.abs(dphidf))
	fmax = fs[imax]*1e-6
	print "Max dphidf frequency: %.6f MHz"%fmax

	pl.subplot(211)
	pl.title('fmin=%.4f MHz fmax=%.4f MHz'%(fmin,fmax))
	pl.plot(fs*1e-6,mag,marker='o')
	pl.axvline(fmin,color='green',linewidth=2)
	pl.xlabel('Frequency (MHz)')
	pl.ylabel('S21 (dB)')
	pl.grid()
	pl.subplot(212)
	pl.plot(fs*1e-6,phase,marker='o')
	pl.axvline(fmax,color='green',linewidth=2)
	pl.xlabel('Frequency (MHz)')
	pl.ylabel('Phase (deg)')
	pl.grid()
	pl.show()

if __name__=='__main__':
	main()
