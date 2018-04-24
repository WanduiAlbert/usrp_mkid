
import numpy as np

class VNAData():
	''' Read and analyze a multi-power vna file from vna.cpp '''
	def __init__(self,fn):
		self.fn = fn

		f = open(fn,'r')
		self.samp_per_step = None
		self.rate = None
		self.tx_gains = None
		self.rx_gain = None
		self.frequencies = None
		skiprows = 0
		while True:
			line = f.readline()
			try:
				name,val = line.split(':')
			except ValueError:
				break

			name = name.strip()
			val = val.strip()
			if name == 'samp_per_step':
				self.samp_per_step = int(val)
			elif name == 'rate':
				self.rate = float(val)
			elif name == 'rx_gain':
				self.rx_gain = float(val)
			elif name == 'tx_gains':
				self.tx_gains = np.fromstring(val,sep=' ')
			elif name == 'frequencies':
				self.frequencies = np.fromstring(val,sep=' ')
			else:
				print "Unknown name in vna file",name
				exit(1)
			skiprows += 1

		assert self.samp_per_step is not None
		assert self.rate is not None
		assert self.tx_gains is not None
		assert self.rx_gain is not None
		assert self.frequencies is not None

		print skiprows
		data = np.loadtxt(fn,skiprows=skiprows)
		if len(data.shape) == 1:
			data = np.array([data])
		assert data.shape[0] == self.tx_gains.size
		assert data.shape[1] == 2*self.frequencies.size
		re,im = data[:,::2],data[:,1::2]
		z = re + 1j*im
		self.fs = self.frequencies
		self.z = z

		#resonators = self.extract_resonators(frequencies,z)
	
	def extract_resonators(self,fs,z):
		''' Estimate locations of resonators from y '''

		# Linearly flatten transfer function
		# Removes slow gain drifts and time delay
		ii = np.arange(z.size)
		p = np.polyfit(ii,z,1)
		fit = np.polyval(p,ii)
		z = z / fit
		mag_scale = np.median(np.abs(z))
		z = z / mag_scale
		mag = np.abs(z)

		# Build a list of regions that go below threshold
		threshold = 0.8
		below_threshold = mag < threshold
		edges = np.diff(below_threshold)
		begins = list(np.where(edges > 0))
		ends = list(np.where(edges < 0))

		# End with out begin
		if ends[0] < begins[0]:
			ends.pop(0)
		# Begin with out end
		if begins[-1] > ends[-1]:
			begins.pop(-1)

		assert len(begins) == len(ends)
		segments = zip(begins,ends)
		resonators = []

		for begin,end in segments:
			imin = np.argmin(mag[begin:end]) + begin
			f0 = fs[imin]
			s21 = mag[imin]
			df = f[end] - f[begin]
			Qr = f0/df
			Qi = Qr / s21
			Qc = Qr / (1-s21)
			resonators.append(f0)

		return resonators
