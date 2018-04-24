
import argparse
import numpy as np
import matplotlib.pyplot as pl
from scipy import signal

# Half band filter generator

def make_fir(ntap,decimate):
	fir = signal.firwin(ntap,1.0/decimate,window='hanning')
	return fir

def compress_fir(fir):
	n = fir.size
	nout = (n-1)/4
	out = np.zeros(nout)
	fir = np.fft.fftshift(fir)[::-1]
	out[0] = fir[0]
	for i in range(nout-1):
		out[i+1] = fir[2*i+1]
	return out

def getargs():
	parser = argparse.ArgumentParser()
	parser.add_argument('-d','--decimate',type=int,required=True,help='Decimate factor')
	parser.add_argument('-n','--ntap',type=int,required=True,help='ntaps')
	args = parser.parse_args()
	return args

def main():
	args = getargs()
	ntap = args.ntap
	decimate = args.decimate

	name = 'fir_coeff_%d_%d'%(ntap,decimate)
	coeff = make_fir(ntap,decimate)

	coeff[abs(coeff) < 1e-16] = 0.0
	print coeff

	coeff_compress = compress_fir(coeff)

	fntxt = name+'.txt'
	f = open(fntxt,'w')
	print>>f,"%d %d"%(ntap,decimate)
	for i in range(coeff.size):
		print>>f, coeff[i]
	
	fnpng = name+'.png'


	w,h = signal.freqz(coeff)
	w /= np.pi
	h = np.abs(h)
	pl.subplot(211)
	pl.plot(coeff,marker='o')
	pl.axhline(0)
	pl.subplot(212)
	pl.semilogy(w,h)
	pl.savefig(fnpng)


if __name__=='__main__':
	main()
