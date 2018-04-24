
import numpy as np
import pylab as pl
from math import pi


def main():
	ntone = 64
	buflen = 65536

	v = np.zeros(buflen,dtype=np.complex)
	phase = 2*pi*np.random.uniform(size=ntone)

	for i in range(ntone):
		j = i * buflen/ntone
		u = np.exp(1.j*phase[i])
		v[j] = u

	ts = np.fft.ifft(v)
	re,im = ts.real,ts.imag

	maxval = np.max(np.abs([re,im]))
	re *= 0.2*32767./maxval
	im *= 0.2*32767./maxval

	re = np.round(re).astype(dtype=np.int16)
	im = np.round(im).astype(dtype=np.int16)

	ts = re + 1.0j*im

	d = np.vstack((re,im)).T
	print d.shape

	np.savetxt('rf_in.dat',d,fmt='%d',header='%d'%buflen)

	pl.psd(ts)
	pl.show()
	

if __name__=='__main__':
	main()
