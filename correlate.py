
import argparse

import numpy as np
import matplotlib.pyplot as pl

def getargs():
	parser = argparse.ArgumentParser()
	parser.add_argument('--tx',required=True)
	parser.add_argument('--rx',required=True)
	args = parser.parse_args()
	return args

def main():
	args = getargs()
	
	rtx,itx = np.loadtxt(args.tx,skiprows=1).T
	ztx = rtx + 1.0j*itx
	nb = len(ztx)

	nc = 128
	offset = nb*128
	nt = nb*nc

	rx = np.fromfile(args.rx,dtype=np.int16)
	rx = rx[offset:offset+2*nt]
	rrx,irx = rx[::2],rx[1::2]
	zrx = rrx + 1.0j*irx

	ztxf = np.fft.fft(ztx)
	ztxpsd = ztxf * np.conj(ztxf)

	accum = np.zeros(nb,dtype=np.complex128)
	for i in range(nc):
		zrxb = zrx[i*nb:(i+1)*nb]
		zrxbf = np.fft.fft(zrxb)
		accum += ztxf*zrxbf.conj()
	
	accum = np.fft.fftshift(accum)
	accum /= nc
	accum /= ztxpsd

	ndec = 256
	kernel = np.ones(ndec)/ndec
	accum = np.convolve(accum,kernel,mode='valid')
	accum = accum[::ndec]

	pl.subplot(211)
	pl.plot(np.abs(accum))
	pl.subplot(212)
	pl.plot(np.angle(accum))
	pl.show()


if __name__=='__main__':
	main()
