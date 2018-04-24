
import configparser
import numpy as np
from scipy import optimize, signal, fftpack
pi = np.pi
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys

def loadmeas(measfn):
    ints = np.memmap(measfn,dtype=np.int16,mode='r')
    floats = ints.astype(np.float32)
    complexs = floats.view(np.complex64)

    runfn = measfn+'.run'
    parser = configparser.RawConfigParser()
    d = parser.read(runfn)
    conf = parser._sections['RFLOOP']
    return complexs,conf

def gaincalsquarenotime(y,power,delta_power):
    y -= np.mean(y)
    highs = y > 0
    lows = y < 0
    lowmean = np.mean(y[lows])
    highmean = np.mean(y[highs])
    delta_y = highmean - lowmean
    gain = delta_y / delta_power

    '''
    t = np.arange(y.size) / rate
    import matplotlib.pyplot as pl
    pl.plot(y)
    pl.axhline(lowmean)
    pl.axhline(highmean)
    pl.show()
    exit()
    '''

    return gain

def gaincalsquare(y,power,delta_power,chopfreq,rate):
    t = np.arange(y.size) / rate

    # Power is a square wave from power to power + delta_power
    # sine_rms is the RMS of the fundamental
    sine_rms = delta_power * np.sqrt(2) / pi

    c = np.cos(2*pi*t*chopfreq)
    s = np.sin(2*pi*t*chopfreq)

    c0 = 2*np.mean(c*y)
    s0 = 2*np.mean(s*y)
    p0 = [c0,s0,chopfreq,0.0]

    def model(t,A,B,f,dc):
        phi = 2.0*pi*t*f
        return A * np.cos(phi) + B * np.sin(phi) + dc

    popt,pcov = optimize.curve_fit(model,t[:1000],y[:1000],p0)
    popt,pcov = optimize.curve_fit(model,t,y,popt)

    A,B,_,_ = popt
    rmssig = np.sqrt(A*A+B*B)

    gain = rmssig/sine_rms
    return gain

def gaincal(y,vdc,vchop,chopfreq,rh,rb,rate):
    t = np.arange(y.size) / rate

    c = np.cos(2*pi*t*chopfreq)
    s = np.sin(2*pi*t*chopfreq)

    c0 = 2*np.mean(c*y)
    s0 = 2*np.mean(s*y)
    p0 = [c0,s0,chopfreq,0.0]

    def model(t,A,B,f,dc):
        phi = 2.0*pi*t*f
        return A * np.cos(phi) + B * np.sin(phi) + dc

    popt,pcov = optimize.curve_fit(model,t[:1000],y[:1000],p0)
    popt,pcov = optimize.curve_fit(model,t,y,popt)

    A,B,_,_ = popt
    rmssig = np.sqrt(A*A+B*B)

    p0 = (vdc/rb)**2 * rh
    pmax = ((vdc+vchop/2.)/rb)**2 * rh
    pmin = ((vdc-vchop/2.)/rb)**2 * rh
    ppp = pmax - pmin
    prms = ppp / (2.*np.sqrt(2.))
    gain = rmssig/prms
    return gain

#import matplotlib.pyplot as pl
def peakfind_convolve(zf,reim=False):
    ''' Convolutional peak finding '''

    zfm = np.mean(zf,axis=0)
    tre = zfm.real + 0.
    tim = zfm.imag + 0.
  
    if reim:
        rf = fftpack.fft(zf,axis=1)
        tf = np.fft.fft(tre + tim*1.0j)
    else:
        rf = fftpack.fft(np.abs(zf),axis=1)
        tf = np.fft.fft(np.abs(tre+tim*1.0j))
    
    
    rf *= np.conjugate(tf)[np.newaxis,:]
    nt = tf.size

    zf[:,:] = fftpack.ifft(rf,axis=1)
    corr = np.fft.fftshift(np.abs(zf),axes=(1,))
    ts = np.argmax(corr,axis=1)
    ts2 = np.zeros(ts.size)

    # Poly fit to the peak of the convolution 
    ii = np.arange(5) - 2
    A = np.vander(ii,N=3)
    AtA = np.dot(A.T,A)
    AtAi = np.linalg.inv(AtA)
    solv = np.dot(AtAi,A.T)
    for i in range(corr.shape[0]):
        t = ts[i]
        cslice = corr[i,t-2:t+3]
        if cslice.size < ii.size:
          continue
        #a,b,c = np.polyfit(ii,cslice,2)
        a,b,c = np.dot(solv,cslice)
        ts2[i] = -b/(2*a) + t
    
    ts2 = ts2 - np.mean(ts2)
    plt.plot(ts2)
    plt.show()
    return ts2

