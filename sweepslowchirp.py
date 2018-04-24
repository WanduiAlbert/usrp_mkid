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

resfreq = 251.6e6#284.3e6
rffreq = 265.0e6
duration = 120 
lo_off = resfreq - rffreq

chopamp = 0.04
chopfreq = 5.0
rate = 1.0e6
rflooppath = '~/code/usrp_mkid/rfloop'
usrpargs = '"type=x300,addr=192.168.40.2"'
x300ip = '192.168.40.2'
chirpfn = 'chirp1024.txt'
vdcs = np.zeros(1)
#vdcs = np.array([0.0, 5.0, 6.0, 7.0])
txgains = np.array([0, 5, 10, 15, 20, 25, 30])
#txgains = np.array([25])
nsampgain = int(10*rate)
nsampnoise = int(duration*rate)
rxgain = 0

rh = 0.107
rb = 1.030e6
ps = np.zeros(1)

def get_datestr():
	now = datetime.datetime.now()
	datestr = now.strftime('%Y%m%d_%H%M%S')
	return datestr

def acquireRF(outfn,txgain,nsamp):
	cmd = '~/code/usrp_mkid/rfloop'
	cmd += ' --tx-args '+usrpargs
	cmd += ' --rx-args '+usrpargs
	cmd += ' --infn '+chirpfn
	cmd += ' --outfn '+outfn
	cmd += ' --settling 1.0'
	cmd += ' --spb 0'
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
	agilent33220.gaincal(chopfreq,vdc,chopamp)
	path = os.path.join('raw',name+'gaincal')
	acquireRF(path,txgain,nsampgain)
	agilent33220.dc(vdc)
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
	vdc = float(sys.argv[1])
	resfreq = float(sys.argv[2])
	lo_off = resfreq - rffreq
	duration = float(sys.argv[3])
	rb = float(sys.argv[4])
	#vdcs[0] = vdc
	#ps = (vdcs/rb)**2 * rh*1e12
	#print "heater powers: ",ps,"pW"
	#print "Resonant frequency", resfreq
	main()

