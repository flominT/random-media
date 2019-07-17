#!/usr/bin/env python3
"""
@Author  :: Flomin TCHAWE
@Date    :: March 24, 2018

Generate 2D random fractional fluctuations describing random media following:
Goff & Jordan, 1988.
"""

import sys
sys.path.append('/Users/flomin/Desktop/thesis/MyScripts/python/modules')
import numpy as np
import ipdb as db
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy import special
import sys
import os
import pandas as pd
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from util_sys import *

class random2D():
  def __init__(self,input_dir='./',n_real=1):
    """
    inputs :
      - input_dir : directory containing inputfile 'input.inp'
      - n_real    : Number of realizations per random media.
                   If n_real > 1, creates array self.m_ACF with shape
                   (n_real,Nx,Nz) containing the random space ACF per realization

    outputs :
      - creates von karman or gaussian autocorrelation function (acf).
      - plots the random fluctuations
      - saves the 2d fluctuations in a text file.
        note !! : writing is done using the bottom left corner as origin.
    """
    self.n_real = int(n_real)
    if input_dir[-1] != '/':
      input_dir += '/'
    self.input_dir = input_dir
    self.run()

  def run(self):
    self.read_input()
    self.__spectral_var()
    self.__acf_psd()
    if self.n_real > 1 :
      self.m_ACF = np.zeros((self.n_real,self._Nz,self._Nx))
      for i in range(self.n_real):
        self.m_ACF[i,:,:] = self.__space_ACF()
    else :
      self.__space_ACF()
    self.theoretical_acf()
    #self.plot_spectrum()
    #self.plot_acf()

  def read_input(self):
    """
    Reads the input file for ACF parameters.
    The format of the input file is described below :

    - 2nd line  : ACF type (VK/GS)
    - 4th line  : Correlation length (in meters)
    - 6th line  : Hurst exponent (0-1)
    - 8th line  : Grid spacing (in meters)
    - 10th line : Number of points in the space directions (integers)
    - 12th line : Option to taper spectrum (taper/filter/None)

    An example of input file is as flows :

    ** ACF type 'VK' or 'GS'
    VK
    ** Correlation length ax az
    50 50
    ** Variance of fluctuations in percentage
    0.05
    ** Hurst exponent 0-1
    0.1
    ** Grid spacing dx dz
    1 1
    ** Number of points in x  and z directions
    1000 1000
    ** Taper spectrum above maximum wavenumber
    filter

    """
    filename = self.input_dir + 'input.inp'

    try :
      with open(filename,'r') as f:
        lines = f.readlines()
    except :
      if not os.path.isfile(filename):
        msg = 'OpenError : Can not open input file input.inp'
        raise Exception(msg)

    # Broadcast variables
    self._ACF   = lines[1].strip()
    self._ax,self._az = list(map(float,lines[3].split(' ')))
    self._eps   = float(lines[5])*1e-2
    self._H     = float(lines[7])
    self._dx,self._dz = list(map(float,lines[9].split(' ')))
    self._Nx,self._Nz  = list(map(int,lines[11].split(' ')))
    self._taper = lines[13].strip().lower()
    if lines[15].strip() == 'None':
      self._kc = None
    else : 
      self._kc = float(lines[15])

    # Check generation conditions
    self.__check_input()

    # Print broadcasted variables
    print("---------------------------------------------")
    print("--          INPUT VARIABLES                --")
    print("---------------------------------------------\n")
    print("- Autocorrelation function type          : {}".format(self._ACF))
    print("- Correlation lengths in meters (ax,az)  : {} {}".format(self._ax,self._az))
    print("- Standard deviation (eps)               : {}".format(self._eps))
    print("- Hurst exponent                         : {}".format(self._H))
    print("- Space step in meters (dx,dz)           : {} {}".format(self._dx,self._dz))
    print("- Domain dimensions in meters (x,z)      : {} {}".format(self._Nx,self._Nz))
    print("- Tapering                               : {}".format(self._taper))


  def __check_input(self):
    """
    Check input data

    """

    # Minimum wave number must be smaller than corner wave number
    if (self._dx*self._Nx < 2.*np.pi*self._ax) or (self._dz*self._Nz < 2.*np.pi*self._az):
      msg = 'Minimum wave number must be smaller than corner wavenumber\n'\
            'Either decrease the correlation or increase the n of grid pts'
      raise Exception(msg)

    # Check grid spacing
    if (self._dx > self._ax/4.) or (self._dz > self._az/4.):
      msg = 'Grid spacing must be atleast 4 times smaller than correlation length \n'\
            'Either increase a or decrease dx'
      raise Exception(msg)

  def __spectral_var(self):
    """ Create spectral domain variables from input data """

    # Space dimensions
    xmax = self._Nx * self._dx
    zmax = self._Nz * self._dz

    self._dkx   = (2.0*np.pi) / xmax
    self._dkz   = (2.0*np.pi) / zmax
    self._kxnyq = np.pi / self._dx
    self._kznyq = np.pi / self._dz
    self._kxmax = self._kxnyq/2     # Maximum wave number is chosen to be half the nyquist wavenumber
    self._kzmax = self._kznyq/2
    self._kmax   = np.sqrt( (self._kxmax * self._kxmax) + (self._kzmax * self._kzmax) )
    self._a = np.sqrt( (self._ax * self._ax) + (self._az * self._az) ) # radial correlation length

    # Wave number vectors
    Nx2, Nz2 = int(self._Nx/2), int(self._Nz/2)

    kx = np.zeros(self._Nx)
    kz = np.arange(Nz2+1) * self._dkz

    if self._Nx%2 :  # if odd
      kx[0:Nx2+1]  = np.linspace(0,self._kxnyq,Nx2+1)
      kx[Nx2+1:]   = np.flip(kx[1:Nx2+1],axis=0)

    else:  # if even
      kx[0:Nx2+1]  = np.linspace(0,self._kxnyq,Nx2+1)
      kx[Nx2+1:]   = np.flip(kx[1:Nx2],axis=0)
      kx[Nx2] = self._kxnyq

    self._kx = kx
    self._kz = kz

    # Print spectral variable parameters
    print("\n")
    print("---------------------------------------------")
    print("--         SPECTRAL VARIABLES              --")
    print("---------------------------------------------\n")
    print("- Nyquist wave numbers (knyqx,knyqz)     : {0:8.5f} {1:8.5f}".format(self._kxnyq,self._kznyq))
    print("- Maximum wave numbers (kxmax,kzmax)     : {0:8.5f} {1:8.5f}".format(self._kxmax,self._kzmax))
    print("- Sampling wave numbers (dkx,dkz)        : {0:8.5f} {1:8.5f}".format(self._dkx,self._dkz))
    print("- Radial correlation length (a)          : {0:8.3f} ".format(self._a))
    print("- Radial wave number (kmax)              : {0:8.3f} ".format(self._kmax))
    print("- Product of a and kmax (ka)             : {0:8.3f} ".format(self._kmax * self._a))


  def butter_response(self,npole=4,plot=False):
    """
     Frequency response of a Butterworth filter :
         butter = 1/sqrt( 1 + (k/kc)^(npole*2) )
     with
         k = sqrt(kx*kx + ky*ky)  (radial wavenumber)
         kc = 2 * Π / a  : corner wave number
    """

    m = ( self._kx - int(self._Nx/2) ) / self._Nx
    n = ( self._kz - int(self._Nz/2) ) / self._Nz
    
    kx = (m[:,np.newaxis] * m[:,np.newaxis])
    kz = (n[np.newaxis,:] * n[np.newaxis,:])
    
    k = np.sqrt( kx + kz )
    
    R = np.max([-m[0] , m[-1] , -n[0] , n[-1]])
    
    kc = self._kc or 2.0 * np.pi / self._a
    
    butter = 1. /  (1 + (k/kc)**(npole*2) ) 
    
    if plot:
      X,Y = np.meshgrid(m,n)
      k_b = np.sqrt(X**2 + Y**2)
      butter_b = 1. /  (1 + (k_b/kc)**(npole*2) )
      fig = plt.figure()
      ax  = plt.axes(projection='3d')
      ax.plot_surface(X, Y, butter_b, cmap=plt.cm.jet, rstride=1, cstride=1, linewidth=0)
      plt.show()
    
    return butter


  def __acf_psd(self):
    """
     Create's 2D Von Karman or Gaussian PSDF
    """

    # Broadcast wavenumber arrays (for vectorization operation (double for loop))
    kx = self._kx[:,np.newaxis]
    kz = self._kz[np.newaxis,:]

    # 2D Radial wave number
    m = ( (kx**2) * (self._ax**2 ) +  (kz**2) * (self._az**2)  )

    # Initialize PSD array
    PSD = np.zeros(m.shape)

    if self._ACF == 'VK':
      # Equation 1 of Carpentier et al. 2007 (c.f. Goff et al. 1988)
      num = 4.0 * np.pi * self._H * self._ax * self._az
      den_fac = 2**(self._H - 1) * special.gamma(self._H)
      PSD = num / ( den_fac * (1 + m)**(self._H + 1) )

    elif self._ACF == 'GS':
      fac = self._eps * self._eps * np.sqrt( np.pi**3 ) * self._a * self._a
      exp = np.exp( -1.0 * 0.25 * m * m * self._a * self._a )
      PSD = fac * exp

    # Filter or taper spectrum
    if self._taper == 'filter' :
      # Low pass filter above the corner wave number k * a ≈ 1
      PSD = PSD * self.butter_response()
    elif self._taper == 'taper':
      # Taper above the maximum wave number
      X,Z = self.cos_taper2d()
      PSD = PSD * X * Z

    # Divide by the spatial frequency step to obtain the power spectral density
    PSD = PSD / (self._dkx * self._dkz)

    self._PSD = np.sqrt(PSD)

    return self._PSD

  @staticmethod
  def random_phase(shape):
    """
    Create random uniform distribution to superimpose on PSD

    Note : Numpy automatically updates the random seed of the random state
    """
    rng = np.random.RandomState()  # Container object for np.random methods
    uni_dist = rng.uniform(size=shape)
    rand_phase = np.cos( 2.0 * np.pi * uni_dist ) + 1j * np.sin( 2.0 * np.pi * uni_dist)
    #toto = np.exp( 2.0 * np.pi * uni_dist * 1j )
    #print(np.allclose(rand_phase,toto))
    return rand_phase

  @staticmethod
  def stand_norm(dist):
    # standardize a distribution
    shape = dist.shape
    mu = np.mean(dist.flatten())
    std = np.std(dist.flatten())
    dist_stand = (dist - mu)/std
    dist_stand = dist_stand.reshape(shape)
    return dist_stand

  @staticmethod
  def fft2d_sym(A,B):
    """
    2d symmetry of ifft2d
            |
         A  |   B
            |
     ---------------
            |
    conj(B) | conj(A)
            |
    """
    Nx,Nz = A.shape
    Nx2,Nz2 = int(Nx/2), int(Nz/2)

    if ( (Nx % 2) != 1 ) and ( (Nz % 2) != 1 ) : # if both are even
      A[:,:Nz2+1] = B
      A[Nx2+1:,Nz2+1:] = np.conj( np.flip(np.flip(A[1:Nx2,1:Nz2],axis=0),axis=1) )
      A[:Nx2+1,Nz2+1:] = np.conj( np.flip(A[:Nx2+1,1:Nz2],axis=1) )
      A[1:Nx2,Nz2+1:]  = np.conj( np.flip(np.flip(A[Nx2+1:,1:Nz2],axis=1),axis=0) )

    elif ( (Nx % 2) == 1 ) and ( (Nz % 2) == 1 ) : # if both are odd
      A[:,:Nz2+1] = B
      A[Nx2+1:,Nz2+1:] = np.conj( np.flip(np.flip(A[1:Nx2+1,1:Nz2+1],axis=0),axis=1) )
      A[:Nx2+1,Nz2+1:] = np.conj( np.flip(A[:Nx2+1,1:Nz2+1],axis=1) )
      A[1:Nx2+1,Nz2+1:]  = np.conj( np.flip(np.flip(A[Nx2+1:,1:Nz2+1],axis=1),axis=0) )

    elif ( (Nx % 2) == 1 ) and ( (Nz % 2) != 1 ) : # if x is odd
      A[:,:Nz2+1] = B
      A[Nx2+1:,Nz2+1:] = np.conj( np.flip(np.flip(A[1:Nx2+1,1:Nz2],axis=0),axis=1) )
      A[:Nx2+1,Nz2+1:] = np.conj( np.flip(A[:Nx2+1,1:Nz2],axis=1) )
      A[1:Nx2+1,Nz2+1:]  = np.conj( np.flip(np.flip(A[Nx2+1:,1:Nz2],axis=1),axis=0) )

    elif ( (Nx % 2) != 1 ) and ( (Nz % 2) == 1 ) : # if z is odd
      A[:,:Nz2+1] = B
      A[Nx2+1:,Nz2+1:] = np.conj( np.flip(np.flip(A[1:Nx2,1:Nz2+1],axis=0),axis=1) )
      A[:Nx2+1,Nz2+1:] = np.conj( np.flip(A[:Nx2+1,1:Nz2+1],axis=1) )
      A[1:Nx2,Nz2+1:]  = np.conj( np.flip(np.flip(A[Nx2+1:,1:Nz2+1],axis=1),axis=0) )

    return A

  def __space_ACF(self):
    """
    A note on numpy.fft.ifft(ifft2) and fortran fftw(fftw2d):

    Numpy ifft2 is normalized by the size of the 2d input array while
    Fortran's FFTW_BACKWARD algorithm is not. This means that the ratio between
    fortran and numpy ifft is equal to size of the 2d input array ( i.e Nx*Nz);

        → fortran_ifft2/python_ifft2 = Nx*Nz

    So to get a consistent python ifft2 value with fotran's FFTW_BACKWARD,
    you need to multiply by the size of the input array;

    i.e  python_ifft2 * (Nx*Nz) = fortran_ifft2
    """

    # Create random complex phase
    rand_phase = self.random_phase( self._PSD.shape )

    # Superimpose PSDF with random phase
    PSD_filt  = rand_phase * self._PSD

    # Define symmetry condition for 2d IFFT
    # Note : This follows python convention of FFT2 computation
    #(see fft2d_sym funcion above)
    
    PWR = np.zeros((self._Nx,self._Nz),dtype=np.complex_) # initialize complex array
    PWR = self.fft2d_sym(PWR,PSD_filt)
    
    # Inverse 2D Fourier Transform and scale
    PSD_fft = np.fft.ifft2( PWR ) * self._Nx * self._Nz  # for consistency with Fortan
    scale = self._dkx * self._dkz
    PSD_fft = np.real( PSD_fft ) * scale

    # Standardize the distribution
    self._rand_acf = self.stand_norm(PSD_fft)

    print("")
    print("-------------------------------")
    print("---  Random fluctuations    ---")
    print("-------------------------------")
    print("Minimum fluctuations       :{}".format( np.min(self._rand_acf) ))
    print("Maximum fluctuations       :{}".format( np.max(self._rand_acf) ))
    print("Standard deviation of fluctuations : {}".format(np.std(self._rand_acf)))

    return self._rand_acf.T

  def plot_acf(self,cmap='gray'):
    #from matplotlib import colors
    # Plot parameters
    label_param  = set_plot_param(option='label',fontsize=16)
    title_param  = set_plot_param(option='label',fontsize=18)
    tick_param   = set_plot_param(option='tick', fontsize=14)
    c_tick_param = set_plot_param(option='c_tick',fontsize=14)

    #cmap = colors.ListedColormap(['lightgreen'])
    #vs = (self._rand_acf.T * 18) + 180
    #vs = np.ones(self._rand_acf.shape) * 180

    title = 'Von Karman ACF \n $a_{x}$ = ' + str(self._ax) + 'm $a_{z}$ = ' + str(self._az) \
            + 'm $\kappa$ = ' + str(self._H) #+ ' $\sigma$ = 10%'
    #title = 'S-wave velocity = 180 $ms^{-1}$'
    fig,ax = plt.subplots(figsize=(10,6))
    cax = ax.imshow(self._rand_acf.T,origin='lower',cmap=cmap, \
             extent=[0,self._Nx*self._dx, -1 * self._Nz*self._dz, 0],aspect='auto')
    #ax.plot(250,-150,'r*',markersize=20)
    #ax.invert_yaxis()
    ax.set_xlabel('Length [m]', **label_param)
    ax.set_ylabel('Depth [m]', **label_param)
    ax.set_title(title,**title_param)
    c = fig.colorbar(cax,fraction=0.1,pad=0.08,shrink=0.8)
    c.set_label('Random Fluctuations',**label_param)
    c.ax.yaxis.set_tick_params(**c_tick_param)
    c.ax.set_yticks(fontname='serif')
    plt.xticks(fontname='serif',fontsize=14)
    plt.yticks(fontname='serif',fontsize=14)
    plt.savefig('/Users/flomin/Desktop/thesis/figures/rand_media/rand50.png')
    plt.show()


  def theoretical_acf(self):
    """
    Compares the 2D random fluctuations with the 1d theoretical PSDF.
    !! Note 2D medium must be isotropic i.e ax=az and Nx = Nz
    """
    set_rcParams()
    ka = self._kx * self._ax

    # 1D psdf
    num = 2.0 * np.sqrt( np.pi ) * special.gamma( self._H + 0.5 ) * self._ax
    den = special.gamma(self._H) * ( 1 + (self._ax * self._ax * self._kx * self._kx ) )**(self._H + 0.5)
    psd1d = num/den
    psd1d = np.sqrt(psd1d/self._dkx)

    # Average along z-direction
    #avg_acf = np.mean( self._rand_acf, axis=0)
    avg_acf  = self._rand_acf[int(self._Nx/2),:]
    rand_psdf = np.abs(np.fft.fft(avg_acf))

    # normalize the spectrum
    rand_psdf = (rand_psdf - rand_psdf.min()) / rand_psdf.max()
    psd1d     = (psd1d - psd1d.min()) / psd1d.max()


    # plot
    plt.figure()
    plt.semilogx(ka,psd1d,'--r',label='Theoretical PSDF')
    plt.semilogx(ka,rand_psdf,'k',label='Random PSDF')
    plt.xlabel('ka')
    plt.ylabel('Normalized PSDF')
    plt.title('1D Von Karman PSDF a = {} m , H = {}'.format(int(self._ax),self._H))
    plt.ylim([0,1])
    plt.xlim([0.1,np.max(ka)])
    plt.legend()
    plt.grid()
    #plt.savefig('/Users/flomin/Desktop/thesis/report/year2/CST/figures/theoretical_1D_a50.png')
    plt.show()

  def plot_spectrum(self):

    Nx2, Nz2 = int(self._Nx/2), int(self._Nz/2)
    
    X = np.linspace(0, self._kxnyq , Nx2) * self._ax * np.pi

    psd1d = np.abs(np.fft.fft(self._rand_acf[2300,:]))
    plt.plot(X,psd1d[:Nx2])
    plt.show()
    db.set_trace()
   
    Z  = psdf2d[:,5]
    Z /= np.max(np.abs(Z))
    fig = plt.figure()
    ax  = fig.gca()
    ax.semilogx(X,Z)
    plt.show()
    db.set_trace()

  def cos_taper2d(self,):
    """
    Taper spectrum above maximum wave number which is half the Nyquist wave number
    """
    hanx = np.ones(len(self._kx))
    hanz = np.ones(len(self._kz))
    kc = 2.0 * np.pi / self._a
    indx = np.where(np.abs(self._kx) > self._kxmax)[0]
    indz = np.where(np.abs(self._kz) > self._kzmax)[0]
    hanx[indx] = 0.5 * (1. - np.cos(2. * np.pi * (self._kx[indx]/self._kxnyq)))
    hanz[indz] = 0.5 * (1. - np.cos(2. * np.pi * (self._kz[indz]/self._kznyq)))
    X,Z = np.meshgrid(hanz,hanx)
    return X,Z





def make_vfile(Z,vsi,vpi,x0,z0,Nx,Nz,dx,dz,ax,az,eps,n=1,dirname='./'):
  def add_veloc(z,eps,v0):
    vperturb = v0*(1+eps*z.flatten(order='C'))
    return vperturb

  def remove_perc(v):
    """
      Keep random values only between μ-3σ and μ+3σ
    """
    mu = np.mean(v)
    std = np.std(v)
    muplus3sigma = mu + (3*std)
    muminus3sigma = mu - (3*std)
    ind1 = np.where(v > muplus3sigma)[0]
    ind2 = np.where(v < muminus3sigma)[0]
    v[ind1] = muplus3sigma
    v[ind2] = muminus3sigma
    return v

  ncol = 2
  for i in range(len(vsi)):
    vp = add_veloc(Z,eps,vpi[i])
    vs = add_veloc(Z,eps,vsi[i])
    vs = remove_perc(vs)
    vp = remove_perc(vp)

    #filename = dirname+'mat'+str(n)+'_ax_'+str(int(ax))+'_az_'+str(int(az))+'_vs_'+str(int(vsi[i]))+'.tab'
    filename = dirname + 'mat{0:d}{1:d}_ax_{2:d}_az_{3:d}_vs_{4:d}.tab'.format(i+1,n,int(ax),int(az),int(vsi[i]))
    f = open(filename,'w')
    f.write(str(ncol)+ ' ' + str(Nx) + ' ' +str(Nz) + ' ' +str(x0) + ' ' +str(z0) + ' ' +str(dx) + ' ' +str(dz)+ '\n')
    #np.savetxt(f,np.column_stack((vs[:,np.newaxis],vp[:,np.newaxis])),fmt='%9.4f   %9.4f',newline='\n')
    stack = pd.DataFrame(np.column_stack((vs[:,np.newaxis],vp[:,np.newaxis])))
    stack.to_csv(f,sep="\t",index=False,header=False,float_format='%9.4f')  # hack around numpy savetxt bug
    f.close()
  return vs.reshape((Nx,Nz))

if __name__ ==  '__main__':
  dirname = "/Users/flomin/Desktop/thesis/MyScripts/bash/randomfield/rand_code"
  obj = random2D(input_dir=dirname,n_real=1)
  db.set_trace()



