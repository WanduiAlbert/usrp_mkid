#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as pl
from scipy import signal, optimize

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
	asd = []
	for i in range(1):
		fs,psd = signal.welch(results[:,i],nperseg=NFFT,fs=sample_rate)
		asd.append(np.sqrt(psd))
	
#names = 'f0 Qr A B'.split()
	names = 'f0 Qr A B Qe_re Qe_im D'.split()


	for i in range(1):
		pl.plot(fs,anocal[i],label='heat=%.2fpW pread=%ddB'%(pdc*1e12,pread))

pl.gca().set_xscale('log')
pl.gca().set_yscale('log')
pl.xlabel('Frequency (Hz)')
pl.ylabel('NEF (Hz/rtHz)')
pl.legend(loc='upper left')
pl.grid()
pl.savefig('chirp_nef1.png')

pl.show()
