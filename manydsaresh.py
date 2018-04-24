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
	pmax = ((vdc+vchop/2.)/rb)**2 * rh
	pmin = ((vdc-vchop/2.)/rb)**2 * rh
	ppp = pmax - pmin
	prms = ppp / (2.*np.sqrt(2.))
	gain = w/prms
	psd /= gain**2
	return psd,gain,p0

setfn = sys.argv[1]
gains = []
for line in open(setfn):
	resultsfn,vdc,vchop,pread = line.split()
	vdc = float(vdc)
	vchop = float(vchop)
	pread = float(pread)

	NFFT = 1024
	results = np.loadtxt(resultsfn)

	results[:,0]*=1e6
	results -= np.mean(results,axis=0)

	sample_rate = 1e6/1024.

	ares = []
	anocal = []
	for i in range(1):
		fs,psd = signal.welch(results[:,i],nperseg=NFFT,fs=sample_rate)
		anocal.append(np.sqrt(psd))
		psd,gain,pdc = gaincal(fs,psd,vdc,vchop)
		results[:,i] /= gain
		asd = np.sqrt(psd)*1e18
		ares.append(asd)
	
	gains.append(gain)

#names = 'f0 Qr A B'.split()
	names = 'f0 Qr A B Qe_re Qe_im D'.split()


	pl.figure(1)
	for i in range(1):
		pl.plot(fs,ares[i],label='heat=%.2fpW pread=%ddB'%(pdc*1e12,pread))
	pl.figure(2)
	for i in range(1):
		pl.plot(fs,anocal[i],label='heat=%.2fpW pread=%ddB'%(pdc*1e12,pread))

pl.figure(1)
pl.gca().set_xscale('log')
pl.gca().set_yscale('log')
pl.xlabel('Frequency (Hz)')
pl.ylabel('NEP (aW/rtHz)')
pl.legend(loc='upper left')
pl.grid()
pl.savefig('chirp_nep1.png')

pl.figure(2)
pl.gca().set_xscale('log')
pl.gca().set_yscale('log')
pl.xlabel('Frequency (Hz)')
pl.ylabel('NEF (Hz/rtHz)')
pl.legend(loc='upper left')
pl.grid()
pl.savefig('chirp_nef1.png')

gains = np.array(gains)*1e-18
pl.figure()
pl.plot(gains,marker='o')
pl.xlabel('Test #')
pl.ylabel('Responsivity (MHz/pW)')
pl.ylim(0,1.3*np.max(gains))
pl.ylim(ymax=1.3*np.max(gains))
pl.grid()
pl.savefig('chirp_responsivity1.png')

pl.show()
