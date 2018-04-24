#!/usr/bin/env python

import os, sys
import numpy as np
import datetime
import time

sys.path.append('/home/lebicep_admin/code/agilent33220/')
import agilent33220
sys.path.append('/home/lebicep_admin/pyhk')
from pyhkcmd import adr

if not os.path.exists('raw'):
	os.mkdir('raw')

rffreq = 265.0e6

lo_off = 0.0

chopamp = 0.04
chopfreq = 5.0
rate = 100.0e6
rflooppath = '~/code/usrp_mkid/rfloop'
usrpargs = '"type=x300,addr=192.168.40.2"'
x300ip = '192.168.40.2'
chirpfn = 'fastchirp_8192_0p2.txt'
vdcs = np.array([0.0])
txgains = np.array([10, 15, 20, 25, 30])
nsampgain = int(10*rate)
nsampnoise = int(20*rate)
rxgain = 0

rh = 0.107
#rb = 1.030e6
rb = 0.854e6
ps = np.zeros(1)

def get_datestr():
	now = datetime.datetime.now()
	datestr = now.strftime('%Y%m%d_%H%M%S')
	return datestr

def acquireRF(outfn,txgain,nsamp):
	cmd = '~/code/usrp_mkid/rfloopdecim'
	cmd += ' --tx-args '+usrpargs
	cmd += ' --rx-args '+usrpargs
	cmd += ' --infn '+chirpfn
	cmd += ' --navg 100'
	cmd += ' --scaler 20'
	cmd += ' --outfn '+outfn
	cmd += ' --settling 1.0'
	cmd += ' --spb 10000000'
	cmd += ' --tx-rate %e'%rate
	cmd += ' --rx-rate %e'%rate
	cmd += ' --tx-freq %e'%rffreq
	cmd += ' --rx-freq %e'%rffreq
	cmd += ' --lo-off %e'%lo_off
	cmd += ' --tx-gain %f'%txgain
	cmd += ' --rx-gain %f'%rxgain
	cmd += ' --nsamp %d'%nsamp
	cmd += ' --ref external'
	cmd += ' --tx-int-n --rx-int-n'
	cmd += ' > %s.log 2>&1'%outfn
	print cmd
	assert(os.system(cmd)==0)
	time.sleep(10.0)

def acquireboth(name,vdc,txgain):
	#agilent33220.gaincal(chopfreq,vdc,chopamp)
	path = os.path.join('raw',name+'gaincal')
	acquireRF(path,txgain,nsampgain)
	#agilent33220.dc(vdc)
	path = os.path.join('raw',name+'noise')
	acquireRF(path,txgain,nsampnoise)

def main():
	datestr = get_datestr()
	logfn = 'autosweepslowchirp_%s.txt'%datestr
	logfile = open(logfn,'w')
	for vdc in vdcs:
		for txgain in txgains:
			datestr = get_datestr()
			acquireboth(datestr,vdc,txgain)
			temp = adr.get_temp()
			logentry = 'acquire %s %f %f %f'%(datestr,temp,txgain,vdc)
			print logentry
			print>>logfile,logentry
			logfile.flush()

if __name__=='__main__':
	main()

