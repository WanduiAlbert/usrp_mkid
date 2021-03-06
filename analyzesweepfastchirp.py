#!/usr/bin/env python3

import sys, os
import numpy as np
import matplotlib.pyplot as pl
from scipy import optimize, signal, fftpack
import time
from math import pi

import chirpanal

#plotfolder = 'lowtemp'
plotfolder = 'hightemp'
if not os.path.exists(plotfolder):
    os.mkdir(plotfolder)

clean_cache = False
resfreqs = [251.328, 252.26, 252.565, 281.684, 281.929, 283.995]
#resfreqs = [251.64, 252.58, 252.85, 282.03, 282.38, 284.3]
#resfreqs = [253.305, 254.248, 254.514, 283.687, 283.950, 285.974]

fn = sys.argv[1]
sample_rate = 100e6
navg = 100
ntotal = 8192
nchirp = 1700

delta_power = 0.05
fast_chirp_rate = sample_rate / ntotal
chirp_rate = fast_chirp_rate / navg
chirp_period = 1e3/chirp_rate
print("chirp_rate: ",chirp_rate,"Hz")
print("chirp_period: ",chirp_period,"ms")

flo = 265.0e6

def reduce_fastchirp(fn,desc):
    fn2 = fn + '.npy'
    if os.path.exists(fn2):
        print("found cached "+fn2)
        if clean_cache:
            os.remove(fn2)
            print ("cache ", fn2, "removed")
        else:
            return np.load(fn2)
    z,_ = chirpanal.loadmeas(fn)
    z = z[80:]
    nchunk = z.size // ntotal
    nt = nchunk * ntotal
    print (nt)
    z = z[:nt].reshape((nchunk,ntotal))
    z = z[100:,:]
    z[:,nchirp:] -= np.mean(z[:,nchirp:],axis=1)[:,np.newaxis]
    z[:,:nchirp] = 0
    fs = np.fft.fftshift(np.fft.fftfreq(ntotal,d=1.0/sample_rate)) + flo
    fs *= 1e-6
    zf = np.fft.fftshift(fftpack.fft(z,axis=1)/np.sqrt(z.shape[1]),axes=(1,))

    plotpath = os.path.join(plotfolder, 'rawfig')

    muzf = np.mean(zf,axis=0)
    pl.clf()
    pl.plot(fs,np.abs(muzf))
    pl.grid()
    pl.xlabel('Frequency (MHz)')
    pl.ylabel('PSD (DAC^2/Hz)')
    pl.title(desc)
    pl.gca().set_yscale('log')
    pl.ylim(1e-1,1e5)
    pl.savefig(plotpath + '/%s_all_meanpsd.png'%desc)
    pl.clf()

    nresfreq = len(resfreqs)
    nt = zf.shape[0]
    ys = np.zeros((nresfreq,nt))
    df = fs[1] - fs[0]
    for i in range(nresfreq):
        resfreq = resfreqs[i]
        #fw = 5.0
        fw = 0.15
        ok = (resfreq - fw < fs) & (fs < resfreq + fw)
        zfc = zf[:,ok]

        pl.plot(fs[ok],np.abs(np.mean(zfc,axis=0)))
        pl.xlabel('Frequency (MHz)')
        pl.ylabel('PSD (DAC^2/Hz)')
        pl.title('%s ch %02d %.3f'%(desc,i,resfreq))
        pl.grid()
        pl.savefig(plotpath + '/%s_ch%02d_meanpsd.png'%(desc,i))
        pl.clf()

        N = zfc.shape[0]
        pl.figure()
        for ind in range(0, N, N//2):
            pl.plot(fs[ok], np.abs(zfc[ind]))
        pl.xlabel('Frequency (MHz)')
        pl.ylabel('PSD (DAC^2/Hz)')
        pl.title('%s ch %02d %.3f'%(desc,i,resfreq))
        pl.grid()
        pl.savefig(plotpath + '/%s_ch%02d_sampledpsd.png'%(desc,i))
        pl.close()

        try:
            ys[i] = df*chirpanal.peakfind_convolve(zfc)
        except TypeError:
            print("peakfind_convolve failed for %s resonator %d %.3fMHz"%(fn,i,resfreq))
            ys[i] = np.nan


    np.save(fn2,ys)
    return ys

def analyze_gain(date,power,desc):
    fn = os.path.join('raw',date+'gaincal')
    ys = 1e6*reduce_fastchirp(fn,desc)

    t = np.arange(ys.shape[1]) / chirp_rate
    gains = []

    plotfn = os.path.join(plotfolder ,'rawfig/%s_ts.png'%desc)
    doplot = not os.path.exists(plotfn)

    if doplot: pl.clf()
    for ch in range(ys.shape[0]):
        y = ys[ch,:]
        gain = chirpanal.gaincalsquarenotime(y,power*1e-12,delta_power*1e-12)
        gains.append(gain)

        if doplot: pl.plot(t,y,label='ch%02d'%ch)

        print("gain %s %.2fpW ch%02d %e Hz/W"%(date,power,ch,gain))

    if doplot:
        pl.grid()
        pl.xlabel('Time (s)')
        pl.ylabel('Frequency (Hz)')
        pl.title('%s'%(desc))
        pl.legend()
        pl.savefig(plotfn)

    return gains

def analyze_noise(date,desc,fn,chirp_rate):
    ys = 1e6*reduce_fastchirp(fn,desc)
    nresfreq = len(resfreqs)
    nperseg = 1024

    plotfn = os.path.join(plotfolder ,'rawfig/%s_ts.png'%desc)
    doplot = not os.path.exists(plotfn)

    t = np.arange(ys.shape[1]) / chirp_rate
    asds = np.zeros((nresfreq,nperseg//2+1))
    if doplot: pl.clf()
    for ch in range(len(resfreqs)):
        fs,psd = signal.welch(ys[ch],fs=chirp_rate,detrend='linear',scaling='density',nperseg=nperseg)
        asds[ch] = np.sqrt(psd)
        if doplot: pl.plot(t,ys[ch],label='ch%02d'%ch)
        print("welch %s"%desc)
        #except ValueError:
        #   print "welch failed for %s %d %.3fMHz"%(date,ch,resfreqs[ch])
        #   asds[ch] = np.nan
    if doplot:
        pl.grid()
        pl.xlabel('Time (s)')
        pl.ylabel('Frequency (Hz)')
        pl.title("%s"%(desc))
        pl.legend()
        pl.savefig(plotfn)

    #if doplot and chirp_rate < fast_chirp_rate:
    common_mode(ys,desc)
    return fs,asds

def common_mode(ys,desc):
    channel_list = list(range(6))
    cross_terms = [[a,b] for a in channel_list\
            for b in channel_list if a > b]
    plotpath = os.path.join(plotfolder, 'cross_spectra')
    if not os.path.exists(plotpath):
        os.mkdir(plotpath)

    nperseg = 256 #1024
    for cterm in cross_terms:
        ch0, ch1 = cterm
        y = ys[[ch0,ch1],:]

        pl.close()
        pl.figure()
        fs,csd = signal.csd(y[0],y[1],fs=chirp_rate,\
                detrend='linear',scaling='density',nperseg=nperseg)
        fs,psd0 = signal.welch(y[0],fs=chirp_rate,\
                detrend='linear',scaling='density',nperseg=nperseg)
        fs,psd1 = signal.welch(y[1],fs=chirp_rate,\
                detrend='linear',scaling='density',nperseg=nperseg)
        print(desc)
        pl.clf()
        pl.plot(fs,np.abs(psd0),label='%3.1fMHz'%resfreqs[ch0])
        pl.plot(fs,np.abs(psd1),label='%3.1fMHz'%resfreqs[ch1])
        pl.plot(fs,np.abs(csd),\
                label='%3.1fMHz x %3.1fMHz'%(resfreqs[ch0], resfreqs[ch1]))
        pl.gca().set_xscale('log')
        pl.gca().set_yscale('log')
        pl.grid()
        pl.xlabel('Frequency (Hz)')
        pl.ylabel('ASD (Hz/rtHz)')
        pl.title('cm_%s'%desc)
        pl.legend()
        pl.savefig(plotpath + '/cm_%s_ch%02d_ch%02d.png'%(desc, ch0, ch1))
        pl.clf()

        # Correlation coefficient
        rho = np.abs(csd)/(np.abs(psd0)*np.abs(psd1))**0.5
        pl.figure()
        pl.plot(fs, rho)
        pl.gca().set_xscale('log')
        pl.gca().set_yscale('log')
        pl.grid()
        pl.ylim(ymax=1,ymin=0)
        pl.xlabel('Frequency (Hz)')
        pl.ylabel('Correlation Coefficient')
        pl.title('%3.1fMHz x %3.1fMHz %s'%(resfreqs[ch0], resfreqs[ch1],
            desc))
        #pl.legend()
        pl.savefig(plotpath +\
                '/cm_coeff_%s_ch%02d_ch%02d.png'%(desc, ch0, ch1))
        pl.close()

def rawreduce(fn):
    dates = []
    temps = []
    txgains = []
    powers = []
    #gains = []
    asds = []
    fastasds = []

    for line in open(fn):
        _,date,temp,txgain,power = line.split()
        #_,date,temp,txgain,power,dpin = line.split()
        dates.append(date)
        temps.append(float(temp))
        txgains.append(float(txgain))
        powers.append(float(power))
        #np.testing.assert_almost_equal(delta_power,float(dpin))

    nacq = len(dates)
    for i in range(nacq):
        date,temp,txgain,power = dates[i],temps[i],txgains[i],powers[i]
        #date,temp,txgain = dates[i],temps[i],txgains[i]

        #desc = "gain_%ddB_%.2fpW"%(txgain,power)
        #gain = analyze_gain(date,power,desc)
        #gains.append(gain)

        desc = "noise_%ddB"%(txgain)
        fn = os.path.join('raw',date+'noise')
        fs,asd = analyze_noise(date,desc,fn,chirp_rate)
        asds.append(asd)

    d = {}
    d['dates'] = dates
    d['temps'] = np.array(temps)
    d['txgains'] = np.array(txgains)
    d['powers'] = np.array(powers)
    d['asds'] = np.array(asds)
    #d['gains'] = np.array(gains)
    d['fs'] = np.array(fs)
    return d

def main():
    plotpath = os.path.join(plotfolder, 'rawfig')
    if not os.path.exists(plotpath):
        os.mkdir(plotpath)

    plotpath_fig = os.path.join(plotfolder, 'fig')
    if not os.path.exists(plotpath_fig):
      os.mkdir(plotpath_fig)
    fn = sys.argv[1]
    d = rawreduce(fn)
    dates = d['dates']
    temps = d['temps']
    txgains = d['txgains']
    fs = d['fs']
    asds = d['asds']
    plotch = [0,1,2,3,4,5]

    nacq = len(dates)
    for i in range(nacq):
          date,temp,txgain = dates[i],temps[i],txgains[i]
          asd = asds[i]
          for ch in plotch:
              pl.figure(100+ch)
              pl.plot(fs,asd[ch, :],label='%.1f'%(txgain))

    for ch in plotch:
        pl.figure(100+ch)
        pl.gca().set_yscale('log')
        pl.gca().set_xscale('log')
        pl.legend(loc='best', title='txgains')
        pl.grid()
        pl.xlabel('Frequency (Hz)')
        pl.ylabel('NEF (Hz/rtHz)')
        pl.title('NEF %3.1f MHz'%resfreqs[ch])
        pl.savefig(os.path.join(plotfolder,\
          'fig/nefasd_%3.1fMHz.png'%resfreqs[ch]))
        pl.close()


    fminl = 5.
    fmaxl = 10.
    ok = (fminl < fs) & (fs < fmaxl)
    meanasd = np.sqrt(np.mean(asds[:,:,ok]**2,axis=2))
    meannef = meanasd

    # Fitting the noise to the readout power model
    fitgains = np.r_[0:30:100j]
    p = np.polyfit(txgains, 10*np.log10(meannef), 1)
    print (p[0, :])
    fminh = 40.
    fmaxh = 50.
    ok = (fminh < fs) & (fs < fmaxh)
    meanasd = np.sqrt(np.mean(asds[:,:,ok]**2,axis=2))
    meannefread = meanasd

    colors = 'blue green red purple orange magenta cyan black'.split()

    for j, ch in enumerate(plotch):
        print("Summary plot ch%02d"%ch)
        pl.figure(400+ch)
        pl.scatter(txgains,meannefread[:,ch],marker='x',\
            label='%d-%dHz'%(fminh, fmaxh), color='b')
        pl.scatter(txgains,meannef[:,ch],marker='o',\
            label='%d-%dHz'%(fminl,fmaxl),color='b')
        pl.plot(fitgains, 10**(np.polyval(p[:, ch], fitgains)/10),\
            label='fit to the slope: %1.3f'%(p[0, ch]),color='r')
        pl.xlabel('TXGain (dB)')
        pl.ylabel('NEF (Hz/rtHz)')
        pl.grid(which='both')
        pl.xlim(xmax=30)
        pl.legend(loc='best')
        pl.title('NEF %3.1f MHz'%resfreqs[ch])
        pl.gca().set_yscale('log')
        pl.ylim(1,1e3)
        pl.savefig(os.path.join(plotfolder,\
          'fig/nef_%3.1fMHz.png'%resfreqs[ch]))
        pl.clf()

if __name__=='__main__':
    main()
