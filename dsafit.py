#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as pl
import reso_fit

#sample_rate = 1.0e6
sample_rate = 0.5e6
sourcefn = sys.argv[1]
measfn = sys.argv[2]

d = np.fromfile(measfn,dtype=np.int16)
mre,mim = d[::2],d[1::2]
print "std(re): ",np.std(mre)
print "std(im): ",np.std(mim)
print "max(re),min(re): ",np.max(mre),np.min(mre)
print "max(im),min(im): ",np.max(mim),np.min(mim)

sre,sim = np.loadtxt(sourcefn,dtype=np.int16,skiprows=1).T

ns = sre.size

sz = sre+sim*1.0j
mz = mre+mim*1.0j

nt = mz.size
nchunk = (nt / ns) - 10
nt2 = nchunk * ns
mz = mz[nt-nt2:]
mz = mz.reshape((nchunk,ns))

fs = np.fft.fftshift(np.fft.fftfreq(ns,d=1.0/sample_rate))
szf = np.fft.fftshift(np.fft.fft(sz))

mzf = np.fft.fftshift(np.fft.fft(mz,axis=1),axes=1)

ratio = mzf/szf[np.newaxis,:]

cal = np.load('cal.npy')
print "cal.shape: ",cal.shape
print "ratio.shape: ",ratio.shape
ratio /= cal[np.newaxis,:]

clip = 64
fs += 258.037e6
fs = fs[clip:-clip]
ratio = ratio[:,clip:-clip]

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

nsolve = nchunk
results = np.zeros((nsolve,7))
ratio_fit = np.zeros_like(ratio)
p0 = None
for i in range(nsolve):
	try:
		f0,Qr,A,B,Qe_re,Qe_im,D,ratio_fit[i],p0 = reso_fit.do_fit(fs,ratio[i].real,ratio[i].imag,p0)
		results[i,:] = f0,Qr,A,B,Qe_re,Qe_im,D
		print "%d"%i,
	except RuntimeError:
		print "fail",
		exit(1)
		results[i,:] = np.nan
		ratio_fit[i] = np.nan
	sys.stdout.flush()

A = results[:,2]
B = results[:,3]
readout_gain = A*A+B*B
readout_phase = np.unwrap(np.arctan2(B,A))
results[:,2] = readout_gain
results[:,3] = readout_phase

np.savetxt(measfn+'.fit',results)

phi = np.unwrap(np.angle(ratio_mean))
p = np.polyfit(fs,phi,1)
phi -= np.polyval(p,fs)

corr = np.exp(-1j*np.polyval(p,fs))
ratio *= corr[np.newaxis,:]

phase2 = np.angle(np.mean(ratio,axis=1))
ii = np.arange(phase2.size)
p = np.polyfit(ii,phase2,1)
corr2 = np.exp(-1j*np.polyval(p,ii))
ratio *= corr2[:,np.newaxis]


fs *= 1e-6
'''
pl.figure()
pl.plot(fs,phi*180/np.pi)
pl.xlabel('Freq (MHz)')
pl.ylabel('Phase (deg)')
pl.title('Mean angle(S21)')
pl.grid()
'''

pl.figure()
extent = (1e-3*np.min(fs),1e-3*np.max(fs),nchunk,0)
pl.xlabel('Frequency (kHz)')
pl.ylabel('Sweep')
pl.title('Waterfall |S21|')
pl.imshow(np.abs(ratio),interpolation='nearest',extent=extent)
pl.colorbar().set_label('|S21|')
pl.gca().set_aspect('auto')
pl.ylim(0,500)
pl.title('abs')
pl.savefig('waterfall_abs.png')

pl.figure()
pl.xlabel('Frequency (kHz)')
pl.ylabel('Sweep')
pl.title('Waterfall |S21|')
pl.imshow(np.abs(ratio-ratio_fit),interpolation='nearest',extent=extent)
pl.colorbar().set_label('|S21|')
pl.gca().set_aspect('auto')
pl.ylim(0,500)
pl.title('residual abs')
pl.savefig('waterfall_abs_resid.png')

'''
pl.figure()
extent = (1e-3*np.min(fs),1e-3*np.max(fs),nchunk,0)
pl.xlabel('Frequency (kHz)')
pl.ylabel('Sweep')
pl.title('Waterfall angle(S21)')
pl.imshow(np.unwrap(np.angle(ratio),axis=1)*180/np.pi,interpolation='nearest',extent=extent)
pl.colorbar().set_label('angle(S21) (Deg)')
pl.gca().set_aspect('auto')
pl.ylim(0,500)
pl.savefig('waterfall_angle.png')
'''

results -= np.mean(results,axis=0)
results /= np.std(results,axis=0)
nfft = 1024
pl.figure()
pl.psd(results[:,0],Fs=sample_rate/1024,NFFT=nfft,label='f0')
pl.psd(results[:,1],Fs=sample_rate/1024,NFFT=nfft,label='Qr')
pl.psd(results[:,2],Fs=sample_rate/1024,NFFT=nfft,label='readout_gain')
pl.psd(results[:,3],Fs=sample_rate/1024,NFFT=nfft,label='readout_phase')
pl.psd(results[:,4],Fs=sample_rate/1024,NFFT=nfft,label='Qe_re')
pl.psd(results[:,5],Fs=sample_rate/1024,NFFT=nfft,label='Qe_im')
pl.psd(results[:,6],Fs=sample_rate/1024,NFFT=nfft,label='D')
pl.gca().set_xscale('log')
pl.legend()
pl.savefig('fit_psd01.png')

pl.figure()
nshow = 200
t = np.arange(nshow)*1024./sample_rate
pl.plot(t,results[:nshow,0],label='f0')
pl.plot(t,results[:nshow,1],label='Qr')
pl.plot(t,results[:nshow,2],label='readout_gain')
pl.plot(t,results[:nshow,3],label='readout_phase')
pl.plot(t,results[:nshow,4],label='Qe_re')
pl.plot(t,results[:nshow,5],label='Qe_im')
pl.plot(t,results[:nshow,6],label='D')
pl.title('Fit timestreams')
pl.grid()
pl.xlabel('Time (seconds)')
pl.ylabel('Fit timestream')
pl.legend()
pl.savefig('fit_ts01.png')

pl.show()
