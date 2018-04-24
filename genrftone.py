
import argparse
import numpy as np
from numpy import pi
import matplotlib.pyplot as pl

def getargs():
	parser = argparse.ArgumentParser()
	parser.add_argument('fn')
	parser.add_argument('-n',default=16384,type=int)
	args = parser.parse_args()
	return args

def main():
	args = getargs()

	n = args.n
	fn = args.fn

	phases = 2.0*pi*np.random.uniform(size=n)
	zfs = np.exp(1.0j*phases)
	zs = np.fft.ifft(zfs)
	
	zs = 0.3*zs / np.max([np.max(zs.real),np.max(zs.imag)])

	zs *= 32767.0;
	zsr = np.round(zs.real).astype(np.int16)
	zsi = np.round(zs.imag).astype(np.int16)

	zsri = np.vstack((zsr,zsi)).T
	np.savetxt(fn,zsri,header=str(n),fmt='%d')

	rms = np.std(zsr)/32767.0
	print rms


if __name__=='__main__':
	main()
