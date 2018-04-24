
import numpy as np
import matplotlib.pyplot as pl

def cic_filter(y,order):
	integ = np.zeros(order,dtype=np.int64)
	ncomb = 32
	comb_buffers = np.zeros((order,ncomb))
	out = []

	for i in range(y.size):
		integ[0] += y[i]
		for j in range(order-1):
			integ[j+1] += integ[j]

		v = integ[order-1]
		for j in range(order-1):
			comb_buffers[j,1:] = comb_buffers[j,:-1]
			comb_buffers[j,0] = v
			v = comb_buffers[j,0] - comb_buffers[j,-1]
		out.append(v)
	out = np.array(out)
	
	scale = 2.0**order
	out = out / scale

	return out

	
nt = 16384
y = np.random.normal(size=nt)
y *= 1024
y = np.array(y,dtype=np.int64)

order = 5
z = cic_filter(y,order)

pl.psd(y)
pl.psd(z)
pl.show()
