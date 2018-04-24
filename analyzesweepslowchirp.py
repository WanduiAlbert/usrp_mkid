#!/usr/bin/env python

import os,sys
import numpy as np
from scipy import optimize,fftpack,signal
import matplotlib.pyplot as pl
import cPickle as pickle
pi = np.pi

import chirpanal

sample_rate = 1.0e6
sourcefn = 'chirp1024.txt'
ns = 1024
chirp_rate = 1.0e6 /ns 

gaincal_chopfreq = 5.0
gaincal_chopamp = 0.04 #0.04
gaincal_rb = 1.03e6#0.854e6
gaincal_rh = 0.106

if not os.path.exists('fig'):
	os.mkdir('fig')

def loadmeas(measfn):
	d = np.fromfile(measfn,dtype=np.int16)
	mre,mim = d[::2],d[1::2]
	print "std(re,im): ",np.std(mre),np.std(mim)

	ratio = np.zeros(mre.shape,dtype=np.complex64)
	ratio.real = mre
	ratio.imag = mim
	return ratio

def reduce_slowchirp(slowchirpfn):
	fndisk = slowchirpfn+'_freqts.npy'
	if os.path.exists(fndisk):
		y = np.load(fndisk)
		return y

	sre,sim = np.loadtxt(sourcefn,dtype=np.int16,skiprows=1).T
	assert sre.size == ns
	chirp_rate = sample_rate / ns
	sz = sre+sim*1.0j

	### Extract network analysis from chirp through FFT(meas)/FFT(source)
	ratio = loadmeas(slowchirpfn)
	nt = ratio.size
	nchunk = (nt / ns) - 10
	nt2 = nchunk * ns
	ratio = ratio[nt-nt2:]
	ratio = ratio.reshape((nchunk,ns))

	fs = np.fft.fftshift(np.fft.fftfreq(ns,d=1.0/sample_rate))
	szf = np.fft.fftshift(np.fft.fft(sz))

	ratio[:,:] = np.fft.fftshift(np.fft.fft(ratio,axis=1),axes=1)

	ratio[:,1:] /= szf[np.newaxis,1:]
	ratio[:,0] = 0

	'''
	do_cal = False
	if do_cal:
		ratio_mean = np.mean(ratio[10:,:],axis=0)
		print ratio.shape
		print ratio_mean.shape
		np.save('cal.npy',ratio_mean)
		exit()

	calfn = 'cal.npy'
	if os.path.exists(calfn):
		cal = np.load('cal.npy')
		print "cal.shape: ",cal.shape
		print "ratio.shape: ",ratio.shape
		ratio /= cal[np.newaxis,:]
	else:
		print 'no calibration found'
	'''

	# Ignore boundaries where there is no RF power in source
	clip = 64
	ratio[:,-clip:] = 0
	ratio[:,:clip] = 0

	# High pass filter network analyses
	ratio_mean = np.mean(ratio,axis=1)
	angle_mean = np.unwrap(np.angle(ratio_mean))
	ii = np.arange(ratio.shape[0])
	phase = np.polyval(np.polyfit(ii,angle_mean,1),ii)
	phasecorr = np.exp(-1j*phase)
	ratio *= phasecorr[:,np.newaxis]

	ratio_mean = np.mean(ratio,axis=0)
	angle_mean = np.unwrap(np.angle(ratio_mean))
	phase = np.polyval(np.polyfit(fs,angle_mean,1),fs)
	phasecorr = np.exp(-1j*phase)
	ratio *= phasecorr

	ratio_mean = np.mean(ratio,axis=1)
	angle_mean = np.unwrap(np.angle(ratio_mean))
	ii = np.arange(ratio.shape[0])
	phase = np.polyval(np.polyfit(ii,angle_mean,1),ii)
	phasecorr = np.exp(-1j*phase)
	ratio *= phasecorr[:,np.newaxis]

	ratio_mean = np.mean(ratio,axis=0)
	angle_mean = np.unwrap(np.angle(ratio_mean))
	ts = chirpanal.peakfind_convolve(ratio)

	fs = np.fft.fftfreq(ns,d=1.0/sample_rate)
	df = fs[1] - fs[0]
	ts *= df

	np.save(fndisk,ts)

	return ts

def analyze_noise(date):
	fn = os.path.join('raw',date+'noise')
	y = reduce_slowchirp(fn)
	print y.size
	#b,a = signal.butter(4, .01)
	#y2 = signal.filtfilt(b, a,y)
	#pl.figure(10)
	#pl.plot(y)
	#pl.plot(y2, linewidth=2)
	#pl.show()
	#exit()
	fs,psd = signal.welch(y,fs=chirp_rate,detrend='linear',scaling='density',nperseg=8192)
	asd = np.sqrt(psd)
	return fs,asd

def analyze_gain(date,vdc):
	fn = os.path.join('raw',date+'gaincal')
	y = reduce_slowchirp(fn)
	gain = chirpanal.gaincal(y,vdc,gaincal_chopamp,gaincal_chopfreq,gaincal_rh,gaincal_rb,chirp_rate)
	print "gain: ",gain,"Hz/W"
	return gain

def rawreduce(fn):
	dates = []
	temps = []
	txgains = []
	vdcs = []
	gains = []
	asds = []
	for line in open(fn):
		_,date,temp,txgain,vdc = line.split()
		dates.append(date)
		temps.append(float(temp))
		txgains.append(float(txgain))
		vdcs.append(float(vdc))

	nacq = len(dates)
	for i in range(nacq):
		date,vdc = dates[i],vdcs[i]
		gain = analyze_gain(date,vdc)
		gains.append(gain)
	
	for i in range(nacq):
		date,temp,txgain,vdc = dates[i],temps[i],txgains[i],vdcs[i]
		fs,asd = analyze_noise(date)
		asds.append(asd)
	
	d = {}
	d['dates'] = dates
	d['temps'] = np.array(temps)
	d['txgains'] = np.array(txgains)
	d['vdcs'] = np.array(vdcs)
	d['asds'] = np.array(asds)
	d['gains'] = np.array(gains)
	d['fs'] = np.array(fs)

	fnpkl = fn+'.pkl'
	pickle.dump(d,open(fnpkl,'w'))

def main():
	fn = sys.argv[1]
	freq = float(sys.argv[2])/1e6
	fnpkl = fn+'.pkl'
	rawreduce(fn)
	d = pickle.load(open(fnpkl,'r'))
	dates = d['dates']
	temps = d['temps']
	txgains = d['txgains']
	vdcs = d['vdcs']
	asds = d['asds']
	fs = d['fs']
	gains = d['gains']
	vdclist = np.array(list(set(vdcs)))

	nacq = len(dates)
	for i in range(nacq):
		for j in range(len(vdclist)):
			vdc = vdclist[j]
			if vdcs[i] != vdc: continue
			date,temp,txgain,vdc = dates[i],temps[i],txgains[i],vdcs[i]
			gain = gains[i]
			asd = asds[i]
			pl.figure(100+j)
			pl.plot(fs,asd,label='%.1f'%txgain)
			# We don't want to plot the NEP when there was no dc offset
			if gain != np.inf:
				pl.figure(200+j)
				pl.plot(fs,1e18*asd/gain,label='%.1f'%txgain)


	rh = gaincal_rh
	rb = gaincal_rb

	for j in range(len(vdclist)):
		vdc = vdclist[j]
		p0 = (vdc/rb)**2 * rh*1e12
		pl.figure(100+j)
		pl.gca().set_yscale('log')
		pl.gca().set_xscale('log')
		pl.legend(loc='best')
		pl.grid()
		pl.xlabel('Frequency (Hz)')
		pl.ylabel('NEF (Hz/rtHz)')
		pl.title('Resonator %3.1f MHz'%freq)
		#pl.title('NEF\nPheater = %.2fpW'%p0)
		pl.savefig('fig/reso_%3.1fMHz_nefasdonfloor.png'%freq)
		
		if gains[j] != np.inf:
			pl.figure(200+j)
			pl.gca().set_yscale('log')
			pl.gca().set_xscale('log')
			pl.legend(loc='best')
			pl.grid()
			pl.xlabel('Frequency (Hz)')
			pl.ylabel('NEP (aW/rtHz)')
			pl.title('Resonator %3.1f MHz'%freq)
			#pl.title('NEP\nPheater = %.2fpW'%p0)
			pl.savefig('fig/reso_%3.1fMHz_nepasd.png'%freq)

	fmin = 2.
	fmax = 6.
	ok = (fmin < fs) & (fs < fmax)
	meanasd = np.sqrt(np.mean(asds[:,ok]**2,axis=1))
	meannep = 1e18*meanasd / gains
	meannef = meanasd

	fmin = 380.
	fmax = 400.
	ok = (fmin < fs) & (fs < fmax)
	meanasd = np.sqrt(np.mean(asds[:,ok]**2,axis=1))
	meannepread = 1e18*meanasd / gains
	meannefread = meanasd

	colors = 'blue green red orange black'.split()

	for j in range(len(vdclist)):
		vdc = vdclist[j]
		p0 = (vdc/rb)**2 * rh*1e12
		ok = vdcs == vdc
		if gains[j] != np.inf:
			pl.figure(3)
			pl.scatter(txgains[ok],meannep[ok],marker='o',label='2-6Hz pheat=%.2fpW'%p0,color=colors[j])
			pl.scatter(txgains[ok],meannepread[ok],marker='x',label='380-400Hz pheat=%.2fpW'%p0,\
			color=colors[j])
			pl.xlabel('TXGain (dB)')
			pl.ylabel('NEP (aW/rtHz)')
			pl.grid()
			pl.ylim(ymin=0)
			pl.legend(loc='best')
			pl.title('Resonator %3.1f MHz'%freq)
			pl.savefig('fig/reso_%3.1fMHz_nep.png'%freq)
	
	pl.figure(4)
	for j in range(len(vdclist)):
		vdc = vdclist[j]
		p0 = (vdc/rb)**2 * rh*1e12
		ok = vdcs == vdc
		#pl.scatter(txgains[ok],meannef[ok],marker='o',label='2-6Hz pheat=%.2fpW'%p0,color=colors[j])
		#pl.scatter(txgains[ok],meannefread[ok],marker='x',label='380-400Hz pheat=%.2fpW'%p0,color=colors[j])
		pl.scatter(txgains[ok],meannef[ok],marker='o',label='2-6Hz',color=colors[j])
		pl.scatter(txgains[ok],meannefread[ok],marker='x',label='380-400Hz',color=colors[j])
		pl.xlabel('TXGain (dB)')
		pl.ylabel('NEF (Hz/rtHz)')
		pl.grid()
		pl.ylim(ymin=0)
		pl.title('Resonator %3.1f MHz'%freq)
		pl.legend(loc='best')
		pl.savefig('fig/reso_%3.1fMHz_nef.png'%freq)

	pl.show()

if __name__=='__main__':
	main()

