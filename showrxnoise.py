#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as pl

sample_rate = 1e6
gains = [0]
colors = 'blue green red orange black'.split()

def makespec(fn,linestyle,color,label):
	d = np.fromfile(fn,dtype=np.int16)
	x,y = d[::2],d[1::2]
	z = x + 1.0j*y
	z -= np.mean(z)
	pl.psd(z,Fs=sample_rate,linestyle=linestyle,color=color,label=label)

for i in range(len(gains)):
	gain = gains[i]
	color = colors[i]
	fnon = 'rxgain'+str(gain)+'noiseon_5dB'
	fnoff = 'rxgain'+str(gain)+'noiseoff_5dB'

	makespec(fnon,linestyle='-',color=color,label='%d on'%gain)
	makespec(fnoff,linestyle='--',color=color,label='%d off'%gain)

pl.legend()
pl.title('PSD of RX input with amps on and off, different USRP rx gain\n5dB RX attenuation')
pl.savefig('rxnoise5dB.png')
pl.show()
