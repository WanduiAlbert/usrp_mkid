#!/usr/bin/env python

import os,sys
import numpy as np
import chirpanal

def dochirp(fn,f):
	cmd = '''~/code/usrp_mkid/rfloop \
		--tx-args "type=x300,addr=192.168.40.2" \
		--rx-args "type=x300,addr=192.168.40.2" \
		--infn fastchirp_8192_0p2.txt \
		--outfn %s \
		--settling 1.0 \
		--spb 1000000 \
		--tx-rate 100.0e6 \
		--rx-rate 100.0e6 \
		--tx-freq %d.0e6 \
		--rx-freq %d.0e6\
		--lo-off 0.000e6 \
		--tx-gain 30 \
		--rx-gain 0 \
		--nsamp 1000000\
		--ref external
		'''%(fn,f,f)
	print cmd
	assert os.system(cmd)==0

fmin = 50
fmax = 2250
fstep = 70
fs = np.arange(fmin,fmax,fstep)

for f in fs:
	fn = 'chirp%04d'%f
	if not os.path.exists(fn):
		dochirp(fn,f)

	os.system('showfastchirp.py %s'%fn)

