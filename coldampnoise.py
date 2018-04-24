#!/usr/bin/env python

import sys
import time, datetime
import uhd
import numpy as np
import matplotlib.pyplot as pl
from scipy import signal

sys.path.append('/home/lebicep_admin/pyhk')
from pyhkcmd import adr

rate = 1e6
freq = 300e6
nt = int(2e6)
gain = 0
channels = [0]
fmin = 0.1
fmax = 0.4

def get_datestr():
	now = datetime.datetime.now()
	datestr = now.strftime('%Y%m%d_%H%M%S')
	return datestr

datestr = get_datestr()
logfn = datestr+'_log.txt'
logf = open(logfn,'w')

# Record variance in timestream with time and temperature of cold stage
# Slowly change the cold stage temperature to measure the cold amp noise temp

usrp = uhd.usrp.MultiUSRP("type=x300,addr=192.168.40.2")

while True:
	t = time.time()
	temp = adr.get_temp()
	z = usrp.recv_num_samps(nt,freq,rate,channels,gain)[0]
	z = z[nt/2:]
	z *= 32767

	fs,psd = signal.welch(z,fs=rate*1e-6,nperseg=1024,detrend='linear')
	fs = np.fft.fftshift(fs)
	psd = np.fft.fftshift(psd)
	ok = (fmin < np.abs(fs)) & (np.abs(fs) < fmax)
	meanpsd = np.sum(psd*ok)
	print t,temp,meanpsd
	print>>logf,t,temp,meanpsd
	logf.flush()
	sys.stdout.flush()
	time.sleep(10.)


