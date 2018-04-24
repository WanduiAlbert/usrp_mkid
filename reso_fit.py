
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

def model_python(f,f0,A,B,D,Qr,Qe_re,Qe_im):
	f0 = f0 * 1e6
	cable_z = np.exp(2.j*pi*(1e-6*D*(f-f0)))
	Qe = Qe_re + 1.j*Qe_im
	x = (f - f0)/f0
	s21 = (A+1.0j*B)*cable_z*(1. - (Qr/Qe)/(1. + 2.j*Qr*x))
	return real_of_complex(s21)

def model_fast(f,f0,A,B,D,Qr,Qe_re,Qe_im):
	f0 = f0*1e6
	nf = f.size
	out = np.zeros(2*nf)
	Qe = Qe_re + 1.j*Qe_im
	Ac = A + 1.0j*B

	c_code = '''
	int i;
	double cable_phase;
	std::complex<double> cable_z;
	std::complex<double> s21;
	std::complex<double> I2(0,2);
	double x;
	for(i=0;i<nf;i++) {
		cable_phase = 2.*M_PI*D*(f(i)-f0);
		cable_z = exp(cable_phase);
		x = (f(i)-f0)/f0;
		s21 = Ac*cable_z*(1.-(Qr/Qe)/(1.+I2*Qr*x));
		out(i) = s21.real();
		out(i+nf) = s21.imag();
	}
	'''

	f0 = float(f0)
	Qe = complex(Qe)
	Ac = complex(Ac)
	Qr = float(Qr)
	D = float(D)
	varlst = 'Ac D Qr Qe nf out f f0'.split()
	weave.inline(c_code,varlst,type_converters=weave.converters.blitz)
	return out

model = model_python

def do_fit_quick(freq,re,im):
	nt = len(freq)

	D = -0.0087
	Qe_re = 31500.
	Qe_im = 12011.
	def quick_model(f,f0,A,B,Qr):
		return model(f,f0,A,B,D,Qr,Qe_re,Qe_im)

	mag = np.sqrt(re*re+im*im)
	phase = np.unwrap(np.arctan2(im,re))

	p = np.polyfit(freq,phase,1)
	phase -= np.polyval(p,freq)

	z = mag*np.exp(1.j*phase)
	re,im = z.real,z.imag

	f0 = freq[np.argmin(mag)]*1e-6
	scale = np.max(mag)
	phi = 0.0
	A = scale*np.cos(phi)
	B = scale*np.sin(phi)
	Qr = 12000
	p0 = (f0,A,B,Qr)

	ydata = np.hstack((re,im))

	popt,pcov = optimize.curve_fit(quick_model,freq,ydata,p0=p0)
	f0,A,B,Qr = popt

	yfit = quick_model(freq,*popt)
	zfit = complex_of_real(yfit)

	Qi = 1.0/ (1./Qr - 1./Qe_re)

	return f0,Qr,A,B,zfit

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
		Qr = 25000
		Qe_re = 50000
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

def main():
	import sys
	fn=sys.argv[1]
	data = np.loadtxt(fn,skiprows=5).T
	freq = data[0]
	res = data[1::2]
	ims = data[2::2]
	nm = res.shape[0]

	print freq.shape
	print res.shape
	print ims.shape

	f0s = []
	Qis = []
	Qrs = []
	powers = np.linspace(-10,16,nm) - 90
	for i in range(nm):
		f0,Qi,Qr = do_fit(freq,res[i],ims[i])
		f0s.append(f0)
		Qis.append(Qi)
		Qrs.append(Qr)
	f0s = np.array(f0s)
	Qis = np.array(Qis)
	Qrs = np.array(Qrs)
	#pl.plot(powers,f0s*1e6)
	pl.plot(powers,Qrs,label='Qr')
	pl.plot(powers,Qis,label='Qi')
	pl.xlabel('Power (dBm)')
	pl.ylabel('Q')
	pl.grid()
	pl.title('Q vs. power at %d MHz'%(f0*1e6))
	pl.legend()
	pl.ylim(0,100e3)
	pl.savefig('qpower_%dMHz.png'%(f0*1e6))
	pl.show()

if __name__=='__main__':
	main()
