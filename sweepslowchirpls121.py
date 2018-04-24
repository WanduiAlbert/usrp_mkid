#!/usr/bin/env python

import os, sys
import numpy as np
import datetime
import time

sys.path.append('/home/lebicep_admin/code/ls121/')
import ls121
sys.path.append('/home/lebicep_admin/pyhk')
from pyhkcmd import adr

if not os.path.exists('raw'):
	os.mkdir('raw')

chop_freq = 0.5
resfreq = 301.5e6  #315.08e6#273.7e6#288.1e6#301.2e6 #329.8e6
rffreq = 300.0e6
duration = 120
lo_off = resfreq - rffreq
rh = 0.106
delta_power = 0.05

def setupgaincal(power):
	i0 = np.sqrt(power/rh)*1e-6
	i1 = np.sqrt((power+delta_power)/rh)*1e-6
	#k2450.square_wave(i0,i1,chop_freq)
	ls121.start_squarewave(i0,i1)
	print "setupgaincal p=%e pW\ti0=%e A\ti1=%e A"%(power,i0,i1)

def setupdc(power):
	rh = 0.106
	i0 = np.sqrt(power/rh)*1e-6
	ls121.dc_current(i0)
	print "setupdc p=%e pW\ti0=%e A"%(power,i0)

rate = 1.0e6
rflooppath = '~/code/usrp_mkid/rfloop'
usrpargs = '"type=x300,addr=192.168.40.2"'
x300ip = '192.168.40.2'
chirpfn = 'chirp1024.txt'
#powers = np.array([0,1,2,3,4,5,6,7])
#txgains = np.array([0, 5, 10, 15, 20, 25])
#txgains = np.array([10,15,20])
powers = np.array([5])
txgains = np.array([15])
nsampgain = int(10*rate)
nsampnoise = int(duration*rate)
rxgain = 0

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
	cmd += ' --spb 20000'
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

def acquireboth(name,power,txgain):
	setupgaincal(power)
	path = os.path.join('raw',name+'gaincal')
	acquireRF(path,txgain,nsampgain)
	setupdc(power)
	path = os.path.join('raw',name+'noise')
	acquireRF(path,txgain,nsampnoise)

def main():
	datestr = get_datestr()
	logfn = 'autosweepslowchirp_%s.txt'%datestr
	logfile = open(logfn,'w')
	for power in powers:
		for txgain in txgains:
			datestr = get_datestr()
			temp = adr.get_temp()
			logentry = 'acquire %s %f %f %f %f %f'%(datestr,temp,txgain,power,delta_power,chop_freq)
			acquireboth(datestr,power,txgain)
			print logentry
			print>>logfile,logentry
			logfile.flush()

if __name__=='__main__':
	main()

