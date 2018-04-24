#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as pl
from scipy import signal, optimize
from statsmodels import robust

#names = ['0dB 0.18K','5dB 0.18K','10dB 0.18K','0dB 0.407K', '5dB 0.407K', '10dB 0.407K']
#styles = ['b-','g-','r-','b--','g--','r--']
names = ['0dB 0.18K','5dB 0.18K','0dB 0.407K', '5dB 0.407K']
styles = ['b-','g-','b--','g--']

def deglitch(y):
	y2 = y + 0.0
	for i in range(5):
		med = np.median(y2)
		std = np.std(y2)
		cut = std*5
		clip = np.abs(y - med) > cut
		y2[clip] = med

	return y2


i = 0
for resultsfn in sys.argv[1:]:
	NFFT = 2048
	results = np.loadtxt(resultsfn)

	sample_rate = 0.390625e6/1024.

	index = 0

	y = results[:,index]
	y2 = deglitch(y)
	'''
	pl.plot(y)
	pl.plot(y2)
	pl.show()
	exit()
	'''
	fs,psd = signal.welch(y2,nperseg=NFFT,fs=sample_rate,detrend='constant')
	asd = np.sqrt(psd)
	pl.plot(fs,asd,styles[i],label=names[i])
	i += 1

pl.gca().set_xscale('log')
pl.gca().set_yscale('log')
pl.xlabel('Frequency (Hz)')
pl.ylabel('NEF (Hz/rtHz)')
#pl.ylabel('NEP (aW/rtHz)')
#pl.title('Estimated NEP from 84.5kHz/pW')
pl.legend()
pl.grid()
pl.savefig('nbalsi_nef.png')
pl.title('f=300.51MHz')
pl.show()

