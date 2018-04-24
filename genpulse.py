#!/usr/bin/env python

import argparse
import numpy as np
from numpy import pi
import matplotlib.pyplot as pl

def getargs():
	parser = argparse.ArgumentParser()
	parser.add_argument('fn')
	parser.add_argument('-n',default=64,type=int,help='number of samples')
	args = parser.parse_args()
	return args

def main():
	args = getargs()

	ntotal = args.n
	fn = args.fn

	x = np.zeros(args.n,dtype=np.int16)
	y = np.zeros(args.n,dtype=np.int16)
	x[0] = 32700
	y[0] = -32700
	
	zsri = np.vstack((x,y)).T
	np.savetxt(fn,zsri,header=str(ntotal),fmt='%d',comments='')

if __name__=='__main__':
	main()
