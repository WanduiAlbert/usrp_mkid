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
chirp_rate = 1.0e6/ns 

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
		return np.load(fndisk)

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

	ratio[:,:] = np.fft.fftshift(fftpack.fft(ratio,axis=1),axes=1)

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

def analyze_gain(date,power,delta_power,chopfreq):
	fn = os.path.join('raw',date+'gaincal')
	y = reduce_slowchirp(fn)
	gain = chirpanal.gaincalsquarenotime(y,power,delta_power,chopfreq,chirp_rate)
	gain *= 1e12		# Hz/pW -> Hz/W
	print "gain: ",gain,"Hz/W"
	return gain

def rawreduce(fn):
	dates = []
	temps = []
	txgains = []
	powers = []
	delta_powers = []
	gains = []
	asds = []
	chopfreqs = []
	for line in open(fn):
		_,date,temp,txgain,power,delta_power,chopfreq = line.split()
		dates.append(date)
		temps.append(float(temp))
		txgains.append(float(txgain))
		powers.append(float(power))
		delta_powers.append(float(delta_power))
		chopfreqs.append(float(chopfreq))

	nacq = len(dates)
	for i in range(nacq):
		date,power,delta_power,chopfreq = dates[i],powers[i],delta_powers[i],chopfreqs[i]
		gain = analyze_gain(date,power,delta_power,chopfreq)
		gains.append(gain)
	
	for i in range(nacq):
		date = dates[i]
		fs,asd = analyze_noise(date)
		asds.append(asd)
	
	d = {}
	d['dates'] = dates
	d['temps'] = np.array(temps)
	d['txgains'] = np.array(txgains)
	d['powers'] = np.array(powers)
	d['delta_powers'] = np.array(delta_powers)
	d['chopfreqs'] = np.array(chopfreqs)
	d['asds'] = np.array(asds)
	d['gains'] = np.array(gains)
	d['fs'] = np.array(fs)

	fnpkl = fn+'.pkl'
	pickle.dump(d,open(fnpkl,'w'))

def main():
	fn = sys.argv[1]
	fnpkl = fn+'.pkl'
	rawreduce(fn)
	d = pickle.load(open(fnpkl,'r'))
	dates = d['dates']
	temps = d['temps']
	txgains = d['txgains']
	powers = d['powers']
	delta_powers = d['delta_powers']
	chopfreqs = d['chopfreqs']
	asds = d['asds']
	fs = d['fs']
	gains = d['gains']

	powerlist = np.array(list(set(powers)))

	nacq = len(dates)
	for i in range(nacq):
		for j in range(len(powerlist)):
			power = powerlist[j]
			if powers[i] != power: continue
			date,temp,txgain,power = dates[i],temps[i],txgains[i],powers[i]
			delta_power,chopfreq = delta_powers[i],chopfreqs[i]
			gain = gains[i]
			asd = asds[i]
			pl.figure(100+j)
			pl.plot(fs,asd,label='%.1f'%txgain)
			# We don't want to plot the NEP when there was no dc offset
			if gain != np.inf:
				pl.figure(200+j)
				pl.plot(fs,1e18*asd/gain,label='%.1f'%txgain)


	for j in range(len(powerlist)):
		power = powerlist[j]
		pl.figure(100+j)
		pl.gca().set_yscale('log')
		pl.gca().set_xscale('log')
		pl.legend(loc='best')
		pl.grid()
		pl.xlabel('Frequency (Hz)')
		pl.ylabel('NEF (Hz/rtHz)')
		pl.title('NEF\nPheater = %.2fpW'%power)
		pl.ylim(0.1,1e3)
		pl.savefig('fig/nefasdonfloor%.1fpW.png'%power)
		
		if gains[j] != np.inf:
			pl.figure(200+j)
			pl.gca().set_yscale('log')
			pl.gca().set_xscale('log')
			pl.legend(loc='best')
			pl.grid()
			pl.xlabel('Frequency (Hz)')
			pl.ylabel('NEP (aW/rtHz)')
			pl.title('NEP\nPheater = %.1fpW'%power)
			pl.ylim(10,10000)
			pl.savefig('fig/nepasd%.1fpW.png'%power)

	fmin = 5.
	fmax = 10.
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

	colors = 'blue green red orange black purple magenta cyan'.split()

	pl.figure(3)
	for j in range(len(powerlist)):
		power = powerlist[j]
		ok = powers == power
		if gains[j] != np.inf:
			pl.scatter(txgains[ok],meannep[ok],marker='o',label='5-10Hz pheat=%.2fpW'%power,color=colors[j])
			pl.scatter(txgains[ok],meannepread[ok],marker='x',label='380-400Hz pheat=%.2fpW'%power,color=colors[j])
	pl.xlabel('TXGain (dB)')
	pl.ylabel('NEP (aW/rtHz)')
	pl.grid()
	pl.legend(loc='best')
	pl.xlim(xmax=40)
	pl.ylim(10,10000)
	pl.gca().set_yscale('log')
	pl.savefig('fig/nep.png')
	
	pl.figure(4)
	for j in range(len(powerlist)):
		power = powerlist[j]
		ok = powers == power
		pl.scatter(txgains[ok],meannef[ok],marker='o',label='5-10Hz pheat=%.2fpW'%power,color=colors[j])
		pl.scatter(txgains[ok],meannefread[ok],marker='x',label='380-400Hz pheat=%.2fpW'%power,color=colors[j])
	pl.xlabel('TXGain (dB)')
	pl.ylabel('NEF (Hz/rtHz)')
	pl.grid()
	pl.ylim(ymin=0)
	pl.xlim(xmax=40)
	pl.legend(loc='best')
	pl.savefig('fig/nef.png')

	#pl.show()

if __name__=='__main__':
	main()

