#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as pl
from vnadata import VNAData

def db(x):
	return 10*np.log10(x)

def load_fn(fn):
	d = VNAData(fn)
	re = d.z.real[0]
	im = d.z.imag[0]
	fs = d.fs

	mag = np.sqrt(re*re+im*im)
	phi = np.arctan2(im,re)
	phi = np.unwrap(phi)
	p = np.polyfit(fs,phi,1)
	print "p: ",p
	print "time delay: ",1e6*p[0] / (2 * np.pi),"us"
	phi = phi - np.polyval(p,fs)

	re = mag*np.cos(phi)
	im = mag*np.sin(phi)

	return fs,re,im

def load_mag(fn):
	d = VNAData(fn)
	re = d.z.real[0]
	im = d.z.imag[0]
	print re.dtype
	fs = d.fs
	#mag = np.sqrt(re*re+im*im)
	mag = re*re+im*im
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

	for fn in sys.argv[1:]:
		fs,re,im = load_fn(fn)
		pl.figure(1)
		pl.plot(re,im)
		pl.figure(2)
		pl.plot(fs,re*re+im*im)
	
	pl.show()
	exit()

	#fs,cal,phasecal = load_mag(calfn)
	fs,mag,phase = load_mag(fn)
	#mag -= cal
	
	imin = np.argmin(mag)
	if imin == 0  or imin == fs.size-1:
		fmin = fs[imin]*1e-6
	else:
		x = fs[imin-1:imin+2] - fs[imin]
		y = mag[imin-1:imin+2] - mag[imin]
		a,b,c = np.polyfit(x,y,2)
		x0 = -b/(2*a) + fs[imin]
		fmin = x0*1e-6
	print "Minimum frequency fit: %.6f MHz"%fmin
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
	#pl.xlim(260.4,260.65)
	pl.grid()
	pl.subplot(212)
	pl.plot(fs*1e-6,phase,marker='o')
	pl.axvline(fmax,color='green',linewidth=2)
	pl.xlabel('Frequency (MHz)')
	pl.ylabel('Phase (deg)')
	#pl.xlim(260.4,260.65)
	pl.grid()
	pl.savefig('usrpvna.png')
	pl.show()

if __name__=='__main__':
	main()
