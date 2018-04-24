#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.animation as animation

def main():
	fn = sys.argv[1]
	d = np.loadtxt(fn)

	r = d[:,::2]
	i = d[:,1::2]
	z = r + 1.0j*i

	line, = pl.gca().plot(np.abs(z[0]))
	pl.ylim(ymin=0)

	def animate(i):
		print "up"
		line.set_ydata(np.abs(z[i]))
		return line,

	def init():
		line.set_ydata(np.abs(z[0]))
		return line,

	ani = animation.FuncAnimation(pl.gcf(),animate,np.arange(z.shape[0]),init_func=init,interval=25,blit=True)

	pl.show()

if __name__=='__main__':
	main()
