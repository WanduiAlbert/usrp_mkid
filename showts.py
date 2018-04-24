#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as pl


fn = sys.argv[1]
sample_rate = 100e6
nfft = 1024

d = np.fromfile(fn,dtype=np.int16)
re,im = d[::2],d[1::2]
z = re +1.j*im
z = z[100000:]

pl.figure()
pl.plot(z.real[:100000],label='re')
pl.plot(z.imag[:100000],label='im')
pl.grid()
pl.legend()

pl.figure()
pl.psd(re+1.j*im,NFFT=nfft,Fs=sample_rate,scale_by_freq=True)
#pl.figure()
#pl.psd(re+1.j*im,NFFT=8192,Fs=sample_rate,scale_by_freq=False)
#pl.psd(z,NFFT=nfft,Fs=sample_rate,scale_by_freq=True)
pl.show()
