
import numpy as np
from scipy import optimize
from scipy import weave
from math import pi
import matplotlib.pyplot as pl


def real_of_complex(z):
	''' flatten n-dim complex vector to 2n-dim real vector for fitting '''
	r = np.hstack((z.real,z.imag))
	return r
def complex_of_real(r):
	assert len(r.shape) == 1
	nt = r.size
	assert nt % 2 == 0
	no = nt/2
	z = r[:no] + 1j*r[no:]
	return z

def model(f,f0,A,B,D,Qr,Qe_re,Qe_im):
	cable_z = np.exp(2.j*pi*(1e-6*D*f))
	Qe = Qe_re + 1.j*Qe_im
	x = (f - f0)/(f0+300.510e6)
	s21 = (A+1.0j*B)*cable_z*(1. - (Qr/Qe)/(1. + 2.j*Qr*x))
	return real_of_complex(s21)

def do_fit(freq,re,im,p0=None):
	nt = len(freq)

	mag = np.sqrt(re*re+im*im)
	phase = np.unwrap(np.arctan2(im,re))

	#p = np.polyfit(freq,phase,1)
	#phase -= np.polyval(p,freq)

	z = mag*np.exp(1.j*phase)
	re,im = z.real,z.imag

	if p0 is None:
		f0 = freq[np.argmin(mag)]*1e-6
		scale = np.max(mag)
		phi = 0.0
		A = scale*np.cos(phi)
		B = scale*np.sin(phi)
		D = 0.0
		Qr = 50000
		Qe_re = 100000
		Qe_im = 0
		p0 = (f0,A,B,D,Qr,Qe_re,Qe_im)

	ydata = np.hstack((re,im))

	popt,pcov = optimize.curve_fit(model,freq,ydata,p0=p0)
	f0,A,B,D,Qr,Qe_re,Qe_im = popt
	yfit = model(freq,*popt)
	zfit = complex_of_real(yfit)
	#print p0
	#print popt
	#exit()

	'''
	pl.clf()
	pl.plot(freq,re*re+im*im)
	pl.plot(freq,np.abs(zfit)**2)
	pl.plot(freq,np.unwrap(np.arctan2(im,re)))
	pl.plot(freq,np.unwrap(np.angle(zfit)))

	#pl.clf()
	zm = re + 1.j*im
	resid = zfit - zm
	#pl.plot(freq,resid)
	#pl.hist(resid,bins=30)
	pl.show()
	exit()
	'''

	Qi = 1.0/ (1./Qr - 1./Qe_re)

	return f0,Qr,A,B,Qe_re,Qe_im,D,zfit,popt

