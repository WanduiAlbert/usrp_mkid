#!/usr/bin/env python3

import sys, os
import numpy as np
import matplotlib.pyplot as pl
from scipy import optimize, signal, fftpack
from math import pi
import time
import chirpanal

resfreqs = [251.64, 252.58, 252.85, 282.03, 282.38, 284.3]
#resfreqs = [273.70,288.3,301.4,315.1,329.7]

fn = sys.argv[1]
sample_rate = 100e6
navg = 100
ntotal = 8192
nchirp = 1700

chirp_rate = sample_rate / (0.+navg * ntotal)
chirp_period = 1e3/chirp_rate
print("chirp_rate: ",chirp_rate,"Hz")
print("chirp_period: ",chirp_period,"ms")

t0 = time.time()
#d = np.fromfile(fn,dtype=np.int16,count=int(1e7))
#d = np.fromfile(fn,dtype=np.int16)
z,conf = chirpanal.loadmeas(fn)
t1 = time.time()
print("data loading time: ",t1-t0)
print("n sample: %e"%(z.size))

plot_basic = True
#plot_basic = False

#flo = 300.0e6
flo = float(conf['rx_freq'])
Q = 100000.
tau = Q / (flo * pi)
print("tau: ",tau)

z = z[80:110000000]
nchunk = z.size // ntotal
nt = nchunk * ntotal

z = z[:nt].reshape((nchunk,ntotal))

z = z[30:,:]

if not os.path.exists("fig"):
	os.mkdir("fig")

if plot_basic:
	pl.figure()
	pl.plot(z[0].real,label='z[0].real')
	pl.plot(z[0].imag,label='z[0].imag')
	pl.plot(z[-1].real,label='z[-1].real')
	pl.plot(z[-1].imag,label='z[-1].imag')
	pl.grid()
	pl.xlabel('Sample (10ns)')
	pl.ylabel('Signal (ADC)')
	pl.title('Timestream of first and last chirps')
	pl.legend()
	pl.savefig('fig/ts.png')

z -= np.mean(z,axis=1)[:,np.newaxis]

if plot_basic:
	pl.figure()
	pl.imshow(10*np.log10(np.abs(z)),interpolation='nearest',aspect='auto')
	pl.title('Waterfall of timestreams, folded by chirp period')
	pl.xlabel('Sample in chirp (10ns)')
	pl.ylabel('Chirp number (%.1f ms)'%chirp_period)
	pl.colorbar()
	pl.savefig('fig/tswater.png')

#z[:,:420] = np.mean(z[:,420:],axis=1)[:,np.newaxis]
z[:,nchirp:] -= np.mean(z[:,nchirp:],axis=1)[:,np.newaxis]
z[:,:nchirp] = 0
#z[:,cut:] -= np.mean(z[:,cut:],axis=1)[:,np.newaxis]
#z = z[:,420:]
#z = z[:1024,:]
#z = z.reshape((8,128,34))
#z = np.mean(z,axis=1)
#print z.shape
#z2 = np.zeros((z.shape[0],16384),dtype=np.complex)
#z2[:,:z.shape[1]] = z
#z = z2

zm = np.mean(z,axis=0)
if plot_basic:
	pl.figure()
	pl.plot(zm.real-np.mean(zm.real))
	pl.plot(zm.imag-np.mean(zm.imag))
	pl.grid()
	pl.title('Average timestream of chirp listen phase')
	pl.xlabel('Sample (10ns)')
	pl.ylabel('Signal (ADC)')
	pl.savefig('fig/tsavg.png')

print("FFTing")
def db(x):
	return 10*np.log10(x)
fs = np.fft.fftshift(np.fft.fftfreq(zm.size,d=1.0/sample_rate)) + flo
fs *= 1e-6
#zf = np.fft.fftshift(np.fft.fft(z,axis=1)/np.sqrt(z.size))
t0 = time.time()
zf = np.fft.fftshift(fftpack.fft(z,axis=1)/np.sqrt(z.shape[1]),axes=(1,))
t1 = time.time()
print("FFT time: ",(t1 - t0))

zmf = np.abs(np.mean(zf,axis=0))**2
zfm = np.mean(np.abs(zf)**2,axis=0)

if plot_basic:
	pl.figure()
	pl.imshow(db(np.abs(zf)**2),interpolation='nearest',aspect='auto',extent=(fs[0],fs[-1],0,zf.shape[0]))
	pl.colorbar()
	pl.xlabel('Frequency (MHz)')
	pl.ylabel('Chirp number (%.1f ms)'%chirp_period)
	pl.title('Waterfall plot of spectra')
	pl.savefig('fig/fftwaterfall.png')

	pl.figure()
	pl.plot(fs,db(zmf),label='psd(mean(timestream))')
	pl.plot(fs,db(zfm),label='mean(psd(timestream))')
	pl.grid()
	pl.xlabel('Frequency (MHz)')
	pl.ylabel('Signal PSD (dB)')
	pl.title('Power spectra of chirp listen phases')
	pl.legend(loc='lower left')
	pl.savefig('fig/psdrg.png')
	pl.savefig(fn+'.png')
	#pl.show()
	exit()

	pl.show()
	exit()


''''
pl.figure()
#pl.plot(fs,db(np.abs(zf[0])**2),label='single trace')
pl.plot(fs,db(zmf),label='psd(mean(timestream))')
pl.plot(fs,db(zfm),label='mean(psd(timestream))')
#pl.gca().set_yscale('log')
pl.grid()
pl.xlabel('Frequency (MHz)')
pl.ylabel('Signal PSD (dB)')
pl.title('Power spectra of chirp listen phases')
pl.xlim(330.0,365.)
pl.ylim(-30,45)
pl.legend(loc='lower left')
pl.savefig('fig/psdrgz1.png')
'''
'''
pl.figure()
#pl.plot(fs,db(np.abs(zf[0])**2),label='single trace')
pl.plot(fs,db(zmf),label='psd(mean(timestream))')
pl.plot(fs,db(zfm),label='mean(psd(timestream))')
#pl.gca().set_yscale('log')
pl.grid()
pl.xlabel('Frequency (MHz)')
pl.ylabel('Signal PSD (dB)')
pl.title('Power spectra of chirp listen phases')
pl.xlim(347.5,351.)
pl.ylim(-20,45)
pl.legend(loc='lower left')
pl.savefig('fig/psdrgz2.png')
'''

def extract_timestream(fres):
	fs = np.fft.fftshift(np.fft.fftfreq(zm.size,d=1.0/sample_rate)) + flo
	fs *= 1e-6
	fwidth = 0.50
	ok = (fres + fwidth > fs) & (fs > fres - fwidth)
	ratio = zf[:,ok]
	fs = fs[ok]
	df = fs[1]-fs[0]

	pl.figure()
	pl.imshow(db(np.abs(ratio)**2),interpolation='nearest',aspect='auto',extent=(fs[0],fs[-1],0,ratio.shape[0]))
	pl.colorbar()
	pl.xlabel('Frequency (MHz)')
	pl.ylabel('Chirp number (%.1f ms)'%chirp_period)
	pl.title('Waterfall plot of spectra of %dMHz resonator'%(fres))
	pl.savefig('fig/waterfallresonator%d.png'%fres)
	pl.close()

	ts = chirpanal.peakfind_convolve(ratio,reim=True)

	ts *= df
	ts = ts - np.mean(ts)

	fmean = np.mean(ts)
	ts -= fmean

	return ts

signals = []
for fres in resfreqs:
	y = 1e6*extract_timestream(fres)
	signals.append(y)
signals = np.array(signals)

N = len(signals[2])
pl.figure()
pl.psd(signals[2], NFFT=256, Fs=chirp_rate, label='full')
#pl.psd(signals[2][:N/2], NFFT=256, Fs=chirp_rate, label='first')
#pl.psd(signals[2][N/2:], NFFT=256, Fs=chirp_rate, label='last' )
#pl.psd(signals[4][:N/2], NFFT=256, Fs=chirp_rate)
#pl.ylim([30, 50])
#pl.yticks([30, 35, 40,45,50])
pl.legend()
pl.show()
exit()

def make_cm(chans,bw):
	src = np.zeros((len(chans),signals.shape[1]))
	for i in range(len(chans)):
		src[i,:] = signals[chans[i],:]

#src = signals + 0.
	b,a = signal.butter(2,bw)
	lpsig = signal.filtfilt(b,a,src,axis=1)
#lpsig = src

	print(lpsig.shape)
	cov = np.dot(lpsig,lpsig.T)/lpsig.shape[1]
	w,v = np.linalg.eigh(cov)
	w = w[::-1]
	v = v[:,::-1]

	cm = np.dot(v[:,0],lpsig)
	print("cm.shape: ",cm.shape)
	print("w: ",w)
	return cm

#cm0 = make_cm([1,3],0.2)
#cm1 = make_cm([2,4],0.2)
#cm2 = signals[0]

'''
vdc = 6.0
vchop = 0.02
chopfreq = 5.0
rh = 0.106
rb = 0.854e6
#gain4 = chirpanal.gaincal(signals[4],vdc,vchop,chopfreq,rh,rb,chirp_rate)
#print "gain4: ",gain4,"Hz/W"
gain4 = 4.05e16		# 12/17/2017 chirp04, 329MHz resonator

y = 1e18*signals[4] / gain4

z = signals[2]
z = signal.filtfilt(b,a,z)

y = y - z*np.dot(y,z)/np.dot(z,z)
y = y - cm*np.dot(y,cm)/np.dot(cm,cm)

fs,psd = signal.welch(y,fs=chirp_rate,detrend='constant',scaling='density',nperseg=1024)
asd = np.sqrt(psd)
pl.plot(fs,asd)
pl.xlabel('Frequency (Hz)')
pl.ylabel('NEP (Hz/rtHz)')
pl.gca().set_xscale('log')
pl.gca().set_yscale('log')
pl.grid()
pl.show()
exit()
'''

def project(y,template):
	y -= template * np.dot(template,y)/np.dot(template,template)

t = np.arange(signals.shape[1]) / chirp_rate
colors = 'blue green red purple orange'.split()
ns = [4]
nperseg = 1024
meanpsdbefore = np.zeros(1+nperseg/2)
meanpsdafter = np.zeros(1+nperseg/2)
for i in range(len(ns)):
	ch = ns[i]
	fs,psd = signal.welch(signals[ch],fs=chirp_rate,detrend='linear',scaling='density',nperseg=nperseg)
	asd = np.sqrt(psd)
	meanpsdbefore += psd
	pl.figure(1)
	#pl.plot(fs,asd,color='black',alpha=0.5)
	#pl.plot(fs,asd,color=colors[i])
	pl.plot(fs,asd,color='blue',label='raw 329MHz')
	pl.figure(2)
	lpsig = signal.decimate(signals[ch],8)
	pl.plot(t[::8],lpsig,color='black')

	project(signals[ch],signals[2])
	#project(signals[i],cm2)
	#project(signals[i],cm0)

	fs,psd = signal.welch(signals[ch],fs=chirp_rate,detrend='constant',scaling='density',nperseg=nperseg)
	asd = np.sqrt(psd)
	meanpsdafter += psd
	pl.figure(1)
	#pl.plot(fs,asd,color=colors[i],linestyle='--')
	pl.plot(fs,asd,color='green',label='clean 329MHz - 301MHz')
	#pl.figure(2)
	#lpsig = signal.decimate(signals[i],8)
	#pl.plot(t[::8],lpsig,color='red')

pl.figure(1)
meanpsdbefore /= len(ns)
meanpsdafter /= len(ns)
#pl.plot(fs,np.sqrt(meanpsdbefore),color='black',linewidth=2,label='raw')
#pl.plot(fs,np.sqrt(meanpsdafter),color='red',linewidth=2,label='cleaned')

pl.legend()
pl.xlabel('Frequency (Hz)')
pl.ylabel('NEF (Hz/rtHz)')
pl.gca().set_xscale('log')
pl.gca().set_yscale('log')
pl.grid()
pl.title('Power spectra of 329MHz bolometer\n301MHz bolometer timestream projected out')
pl.savefig('cmsub.png')

pl.figure(2)

pl.legend()
pl.xlabel('Time (s)')
pl.ylabel('Resonator frequency (Hz)')
pl.title('Timestreams of released bolometers at zero power')
pl.grid()
pl.savefig('cmsubts.png')

pl.xlim(45,65)
pl.savefig('cmsubtszoom.png')
pl.close()

pl.show()
exit()

print(w)
pl.imshow(cov,interpolation='nearest',vmin=-0.3,vmax=0.3)
pl.colorbar()
pl.show()

exit()


pl.figure(100)
pl.ylabel('NEF (Hz/rtHz)')
pl.xlabel('Frequency (Hz)')
pl.gca().set_yscale('log')
pl.gca().set_xscale('log')
pl.grid()
pl.legend()
title = 'p=%d f=%.2f'%(ntotal,nchirp/float(ntotal))
pl.title(title)
pl.savefig('fig/nef01.png')

pl.figure(101)
pl.ylabel('NEP (aW/rtHz)')
pl.xlabel('Frequency (Hz)')
pl.gca().set_yscale('log')
pl.gca().set_xscale('log')
pl.grid()
pl.legend()
pl.title(title)
pl.savefig('fig/nep01.png')
pl.show()

