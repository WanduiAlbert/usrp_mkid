#! /usr/bin/env python

import sys, os
import pickle
import numpy as np
import matplotlib.pyplot as pl
from scipy import fftpack
pi = np.pi

showPlots = False 
sample_rate = 1.0e6
resfreq = 253.305e6

chirpfile = "chirp1024.txt"
savedir = "fig/diagnostics/"	

if not os.path.exists(savedir):
	os.mkdir(savedir)

sre,sim = np.loadtxt(chirpfile,dtype=np.int16,skiprows=1).T
ns = sre.size
chirp_rate = sample_rate / ns
sz = sre+sim*1.0j
fs = np.fft.fftshift(np.fft.fftfreq(ns,d=1.0/sample_rate))
szf = np.fft.fftshift(np.fft.fft(sz))

def loadmeas(measfn):
	d = np.fromfile(measfn,dtype=np.int16)
	mre,mim = d[::2],d[1::2]
	print "std(re): ",np.std(mre)
	print "std(im): ",np.std(mim)
	print "max(re),min(re): ",np.max(mre),np.min(mre)
	print "max(im),min(im): ",np.max(mim),np.min(mim)

	ratio = np.zeros(mre.shape,dtype=np.complex64)
	ratio.real = mre
	ratio.imag = mim
	return ratio

def adjustphase(x, ratio, axis):
	ratio_mean = np.mean(ratio,axis=axis)
	angle_mean = np.unwrap(np.angle(ratio_mean))
	phase = np.polyval(np.polyfit(x,angle_mean,1),x)
	phasecorr = np.exp(-1j*phase)
	if axis == 1:
		ratio *= phasecorr[:,np.newaxis]
	elif axis == 0: 
		ratio *= phasecorr
	return ratio


def waterfallplotter(measfile, txgain, vdc, savename):
	plotTitle = True
	if txgain == None or vdc == None:
		plotTitle = False

	ratio = loadmeas(measfile)
	nt = ratio.size
	nchunk = (nt / ns) - 10
	nt2 = nchunk * ns
	ratio = ratio[nt-nt2:]
	ratio = ratio.reshape((nchunk,ns))
	
	# Save a representative time stream of the data to see 
	# the noise characteristic
	if savename:
		savemeasfile = savename	
	else:
		savemeasfile = os.path.split(measfile)[-1]
	
	# Plot a representative view through the timestreams
	fig, ax = pl.subplots()
	period = nchunk/10
	data = np.abs(ratio[nchunk/2, :])
	pivot = np.where(data == np.min(data))[0]
	ax.plot(np.roll(data, -pivot))
	#for i, row in enumerate(ratio[::period, :]):
	#	ax.plot(fs, row, label="%d"%i)
	ax.grid()
	if plotTitle:
		ax.set_title("Timestream with Vdc = {0:1.1f} V, txgain = {1:1.1f}dB".format(vdc, txgain))
	#ax.legend(loc="best")
	tsfilename = savedir + "ts_" + savemeasfile + ".png"
	pl.savefig(tsfilename)
	
	ratio[:,:] = np.fft.fftshift(fftpack.fft(ratio,axis=1),axes=1)
	ratio[:,1:] /= szf[np.newaxis,1:]
	ratio[:,0] = 0
	clip = 64
	ratio[:,-clip:] = 0
	ratio[:,:clip] = 0

	
	# Generate a section through the ratio array to see the shape of the resonance
	fig, ax = pl.subplots()
	period = nchunk/10
	ax.plot((fs + resfreq)/1e6, np.abs(ratio[nchunk/2, :]))
	#for i, row in enumerate(ratio[::period, :]):
	#	ax.plot(fs, row, label="%d"%i)
	ax.set_xlabel(r'Frequency [MHz]')
	ax.set_ylabel(r'S21')
	ax.grid()
	if plotTitle:
		ax.set_title("Vdc = {0:1.1f} V, txgain = {1:1.1f}dB".format(vdc, txgain))
	#ax.legend(loc="best")
	secfilename = savedir + "section_" + savemeasfile + ".png"
	pl.savefig(secfilename)
	
	ii = np.arange(ratio.shape[0])
	ratio = adjustphase(ii, ratio, axis=1)

	ratio = adjustphase(fs, ratio, axis=0)
	ratio = adjustphase(ii, ratio, axis=1)

	# Generate the waterfall plot for the data
	pl.figure(200)
	pl.imshow(np.abs(ratio[::10,:]),aspect='auto',interpolation='nearest')
	if plotTitle:
		pl.title("Vdc = {0:1.1f} V, txgain = {1:1.1f}dB".format(vdc, txgain))
	xticks = (fs + resfreq)/1e6
	xticklabels = ["{0:3.1f}".format(_) for _ in xticks]
	pl.xticks(np.arange(1024)[::146],xticklabels[::146] )
	pl.xlabel(r'Frequency [MHz]')
	yticks = np.arange(0, 1000, 200)
	yticklabels = ["{0:2.1f}".format(_) for _ in (yticks/ chirp_rate * 10)]
	pl.yticks(yticks, yticklabels)
	pl.ylabel(r'Time [s]')
	savefilename = savedir + "waterfall_" + savemeasfile + ".png"
	pl.savefig(savefilename)
	if showPlots:
		pl.show()

def responsivityplotter(fnpkl):
	d = pickle.load(open(fnpkl,'r'))
	dates = d['dates']
	txgains = d['txgains']
	vdcs = d['vdcs']
	gains = np.array(d['gains'])
	nacq = len(vdcs)
	pl.figure(500)
	pl.plot(txgains, gains * 1e-15, 'bs', markersize=10)
	pl.xlabel(r"Txgain [dB]")
	pl.ylabel(r"Responsivity [kHz/pW]")
	savefilename = savedir + fnpkl.split(".")[0] + "_responsivity_vs_dcoffset.png"
	pl.savefig(savefilename)
	if showPlots:
		pl.show()

def singleresponsivityplotter(fnpkl, freq):
	freq = str(int(np.trunc(freq/1e6)))
	d = pickle.load(open(fnpkl,'r'))
	dates = d['dates']
	txgains = d['txgains']
	vdcs = d['vdcs']
	gains = d['gains']
	nacq = len(vdcs)
	vdc = vdcs[0]
	dc = ""
	if vdc == 8.4:
		dc = "8p4"
	else:
		dc = str(int(np.trunc(vdc)))
	pl.figure(500)
	pl.plot(txgains, gains, 'bs', markersize=10)
	pl.xlabel(r"Txgain [dB]")
	pl.ylabel(r"Responsivity [Hz/W]")
	savefilename = savedir +  freq + "MHz_" + dc + "Vdc_responsivity_vs_txgain.png"
	pl.savefig(savefilename)
	if showPlots:
		pl.show()

def getsavename(freq, txgain, vdc):
	freq = "%3.1f"%(freq/1e6)
	#freq = str(int(np.trunc(freq/1e6)))
	dc = ""
	if vdc == 8.4:
		dc = "8p4"
	else:
		dc = str(int(np.trunc(vdc)))
	tx = str(int(np.trunc(txgain)))
	return freq + "MHz_" + dc + "Vdc_" + tx + "dB"


if __name__ == "__main__":
	fn = sys.argv[1]
	resfreq = float(sys.argv[2])
	# If the filename is a txt file, we want the diagnostic plots for all the txgains listed in the file.
	# If not, then we have a raw data file and we simply call waterfallplotter on the file
	if os.path.split(fn)[-1].split(".")[-1] != "txt":
		waterfallplotter(fn, None, None, None)
	else:
		for line in open(fn):
			_, date, temp, txgain, vdc = line.split()
			vdc = float(vdc)
			txgain = float(txgain)
			gaincalfn = os.path.join("raw", date + "gaincal")
			if os.path.exists(gaincalfn):
				savename = getsavename(resfreq, txgain, vdc)
				waterfallplotter(gaincalfn, txgain, vdc, savename)
		# Check to see if a pickle file exists. If not, we haven't run the analysis yet. 
		# If it does, generate the responsivities
		fnpkl = fn + ".pkl"
		if (os.path.exists(fnpkl)):
			singleresponsivityplotter(fnpkl, resfreq)
