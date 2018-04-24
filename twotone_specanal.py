#!/usr/bin/env python

import argparse
import numpy as np
import matplotlib.pyplot as pl

def getargs():
	parser = argparse.ArgumentParser()
	fn = parser.add_argument('fn')
	args = parser.parse_args()
	return args
	
def main():
	args = getargs()
	ntone = np.fromfile(args.fn,dtype=np.uint32,count=1)[0]
	print ntone
	words = np.fromfile(args.fn,dtype=np.float32,count=1+2*ntone)[1:]
	amps = words[::2]
	words = np.fromfile(args.fn,dtype=np.uint32,count=1+2*ntone)[2:]
	steps = words[::2]
	print amps
	print steps
	data = np.fromfile(args.fn,dtype=np.int16)[2*(1+2*ntone):]
	re,im = data[::2],data[1::2]
	z = re + 1.j*im
	pl.psd(z,NFFT=65536,Fs=1e6)
	pl.show()
		
	
if __name__=='__main__':
	main()
