#!/usr/bin/env python

import argparse
import sys
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.animation as animation
import time

def getargs():
	parser = argparse.ArgumentParser()
	parser.add_argument('fn')
	parser.add_argument('--dbmin',default=-80.)
	parser.add_argument('--dbmax',default=10.)
	args = parser.parse_args()
	return args

def main():
	args = getargs()
	f = open(args.fn,'r')

	pl.ion()


	line = f.readline()
	fs = 1e-6*np.fromstring(line,sep=' ')
	
	line = f.readline()
	line = f.readline()
	line = f.readline()
	d = np.fromstring(line,sep=' ')
	d = d[::2] + 1.0j*(d[1::2])
	mag = 20*np.log10(np.abs(d))
	phase = np.angle(d) *180./np.pi
	ref = d

	pl.subplot(211)
	mag_line, = pl.plot(fs,mag)
	pl.ylim(-80,10)
	pl.ylabel('|S12| (dB)')
	pl.grid()
	pl.subplot(212)
	phase_line, = pl.plot(fs,phase)
	pl.xlabel('Frequency (MHz)')
	pl.ylabel('angle(S12) (degrees)')
	pl.ylim(-200,200)
	pl.grid()

	for line in f:
		d = np.fromstring(line,sep=' ')
		d = d[::2] + 1.0j*(d[1::2])
		d /= ref
		mag = 20*np.log10(np.abs(d))
		phase = np.angle(d)* 180./np.pi

		mag_line.set_ydata(mag)
		phase_line.set_ydata(phase)
		pl.pause(0.01)


if __name__=='__main__':
	main()
