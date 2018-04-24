#!/usr/bin/env python

import os, sys
import numpy as np
import datetime
import time

sys.path.append('/home/lebicep_admin/pyhk')
from pyhkcmd import adr
sys.path.append('/home/lebicep_admin/code/ls121/')
import ls121

if not os.path.exists('raw'):
	os.mkdir('raw')

rffreq = 300.0e6

lo_off = 0.0

chopfreq = 0.5
rate = 100.0e6
rflooppath = '~/code/usrp_mkid/rfloop'
usrpargs = '"type=x300,addr=192.168.40.2"'
x300ip = '192.168.40.2'
chirpfn = 'fastchirp_8192_0p2.txt'
powers = np.array([7])
txgains = np.array([ 20])
nsampgain = int(10*rate)
nsampnoise = int(120*rate)
nsampfastnoise = int(1*rate)
rxgain = 0

rh = 0.106
delta_power = 0.05

def get_datestr():
	now = datetime.datetime.now()
	datestr = now.strftime('%Y%m%d_%H%M%S')
	return datestr

def acquireRFfast(outfn,txgain,nsamp):
	cmd = '~/code/usrp_mkid/rfloop'
	cmd += ' --tx-args '+usrpargs
	cmd += ' --rx-args '+usrpargs
	cmd += ' --infn '+chirpfn
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

def setupgaincal(power):
	i0 = np.sqrt(power/rh)*1e-6
	i1 = np.sqrt((power+delta_power)/rh)*1e-6
	ls121.start_squarewave(i0,i1)
	print "setupgaincal p=%e pW\ti0=%e A\ti1=%e A"%(power,i0,i1)

def setupdc(power):
	rh = 0.106
	i0 = np.sqrt(power/rh)*1e-6
	ls121.dc_current(i0)
	print "setupdc p=%e pW\ti0=%e A"%(power,i0)

def acquireboth(name,power,txgain):
	setupgaincal(power)
	path = os.path.join('raw',name+'gaincal')
	acquireRF(path,txgain,nsampgain)
	setupdc(power)
	path = os.path.join('raw',name+'fastnoise')
	acquireRFfast(path,txgain,nsampfastnoise)
	path = os.path.join('raw',name+'noise')
	acquireRF(path,txgain,nsampnoise)

def main():
	datestr = get_datestr()
	logfn = 'autosweepslowchirp_%s.txt'%datestr
	logfile = open(logfn,'w')
	for power in powers:
		for txgain in txgains:
			datestr = get_datestr()
			acquireboth(datestr,power,txgain)
			temp = 0.0 #adr.get_temp()
			logentry = 'acquire %s %f %f %f %f'%(datestr,temp,txgain,power,delta_power)
			print logentry
			print>>logfile,logentry
			logfile.flush()

if __name__=='__main__':
	main()

