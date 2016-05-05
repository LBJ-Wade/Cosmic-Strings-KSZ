# Name: Class_CosmicString.py
#
# CosmicStrings Software I
#
# Type: Python Class
#
# Content: 1 class, 1 constructor, 2 methods
#
# Description:
#
#


__author__ = "Guadalupe Canas Herrera"
__copyright__ = "Copyright (C) 2015 G. Canas Herrera"
__license__ = "Public Domain GNU"
__version__ = "1.0.0"
__maintainer__ = "Guadalupe Canas Herrera"
__email__ = "gch24@alumnos.unican.es"


import numpy as np #Maths arrays and more, matlab-type vectors/arrays
import matplotlib.pyplot as plt #Plot Libraries
import math #mathematical functions
from astropy.modeling import models, fitting #Package for fitting functions with a astronomical character
import random #pseudo-random generator
from astropy.io import fits #Open and Reading FITS Files usign astropy
from scipy import fftpack
import pylab as py


"""
    General Class that contents several methods destinated to plot CosmicStrings
    
"""

class CosmicStrings(object):
    
    """
        Constructor that save into attributes main values of the simulation
        
    """
    
    def __init__(self, L=12.8, G_mu=1, n=15, gamma=1, v_bar=15):
        self.G_mu=1
        self.n=n
        self.gamma=gamma
        self.v_bar=v_bar
        self.L=L
        self.d_0=1.8
        self.N=10 #input parameter of strings
        self.N_hubble=[]
        self.N_strings=[]
        self.M_field_expanded=[]
        self.d_0_pixels=72
        self.L_pixels=512
        self.pixel_size=0.025
        self.M_field_view=np.zeros((self.L_pixels, self.L_pixels))
        self.simulation_picture=''

    """
        Method that defines temperature fluctuation of the string
    
    """
    def temperature_fluctuation(self, T=2.726):
        r=np.random.rand(1) #flag r ---> to consider different velocities
        return r*self.v_bar*4*np.pi*self.G_mu


    """
        Method that defines the hubble radius for different time intervals as a function of n
    
    """

    def hubble_radius(self, m):
        d=self.d_0_pixels*np.exp(m/3) # m is the index of n
        return d

    """
        Method that calculates the Hubble Volume and the Number of String to be thrown
    
    """

    def hubble_volume(self, m):
        N_h_double = (self.L_pixels/self.hubble_radius(m) + 2.0)*(self.L_pixels/self.hubble_radius(m) + 2.0)
        N_h=int(N_h_double)
        N_s=self.N*N_h
        return N_h, N_s



    """
        Method that calculates the proyection of the string length
    
    """
        
    def string_proyection(self, d):
        alpha=np.random.rand(1)*(np.pi/2)
        string_end=self.gamma*np.cos(alpha)*d
        return string_end

    
    """
        Methods that PLOTS the rhombuses according to the angles
    
    """
    
    def romboide_positive(self, x_m, y_m, t, ll, d, l):
        
        lup=np.array([ll[0]+l*np.cos(t), ll[1]+l*np.sin(t)])
        t_prime=(np.pi/2-t)
        ldown=np.array([ll[0]+d*np.cos(t_prime), ll[1]-d*np.sin(t_prime)])
        lr=np.array([ldown[0]+l*np.cos(t), ldown[1]+l*np.sin(t)])
        
        m_1=(lup[1]-ll[1])/(lup[0]-ll[0])
        m_2=(lr[1]-lup[1])/(lr[0]-lup[0])
        m_3=(ldown[1]-ll[1])/(ldown[0]-ll[0])
        m_4=(lr[1]-ldown[1])/(lr[0]-ldown[0])
        
        ver = np.logical_and(y_m < m_1*(x_m-ll[0])+ll[1], y_m < m_2*(x_m-lup[0])+lup[1])
        hor = np.logical_and(y_m > m_3*(x_m-ll[0])+ll[1], y_m > m_4*(x_m-ldown[0])+ldown[1])
        return ver, hor

    def romboide_negative(self, x_m, y_m, t, ll, d, l):
        t_change=-t
        t_prime=(np.pi/2-t_change)
        lup=np.array([ll[0]+d*np.cos(t_prime), ll[1]+d*np.sin(t_prime)])
        ldown=np.array([ll[0]+l*np.cos(t_change), ll[1]-l*np.sin(t_change)])
        lr=np.array([ldown[0]+d*np.cos(t_prime), ldown[1]+d*np.sin(t_prime)])
        
        m_1=(lup[1]-ll[1])/(lup[0]-ll[0])
        m_2=(lr[1]-lup[1])/(lr[0]-lup[0])
        m_3=(ldown[1]-ll[1])/(ldown[0]-ll[0])
        m_4=(lr[1]-ldown[1])/(lr[0]-ldown[0])
        
        ver = np.logical_and(y_m < m_1*(x_m-ll[0])+ll[1], y_m < m_2*(x_m-lup[0])+lup[1])
        hor = np.logical_and(y_m > m_3*(x_m-ll[0])+ll[1], y_m > m_4*(x_m-ldown[0])+ldown[1])
        return ver, hor
    
    def romboide_angle_0(self, x_m, y_m, t, ll, d, l):
        lr = ll+np.array([l,0])
        lleft = ll+np.array([l,0])
        hor = np.logical_and(y_m < ll[1], y_m > ll[1]-d )
        ver = np.logical_and(x_m > ll[0], x_m < ll[0]+l)
        hor_prime = np.logical_and(y_m > ll[1], y_m < ll[1]+d )
        ver_prime = np.logical_and(x_m > ll[0], x_m < ll[0]+l)
        return ver, hor, ver_prime, hor_prime
    
    def romboide_angle_pi_2(self, x_m, y_m, t, ll, d, l):
        lr = ll+np.array([d,0])
        lleft = ll-np.array([d,0])
        hor = np.logical_and(y_m > ll[1], y_m < ll[1]+l*np.sin(t) )
        ver = np.logical_and(y_m > m*(x_m-lr[0])+lr[1], y_m < m*(x_m-ll[0])+ll[1])
        hor_prime = np.logical_and(y_m > ll[1], y_m < ll[1]+l*np.sin(t) )
        ver_prime = np.logical_and(y_m >m*(x_m-ll[0])+ll[1], y_m < m*(x_m-lleft[0])+lleft[1])
        return ver, hor, ver_prime, hor_prime
    

    """
        Method that evaluates which rombus we should plot according to the angle
        
    """
    
    def angle_evaluation(self, x_m, y_m, t, ll, d, l):
        
        #t is the angle, ll is the left vertex, d is the hubble radius and l is the length of the string (string proyection)
        
        if t==0:
            ver, hor, ver_prime, hor_prime=self.romboide_angle_0(x_m, y_m, t, ll, d, l)
            return ver, hor, ver_prime, hor_prime

        elif t==np.pi/2:
            ver, hor, ver_prime, hor_prime=self.romboide_angle_pi_2(x_m, y_m, t, ll, d, l)
            return ver, hor, ver_prime, hor_prime

        elif t>0 and t<np.pi/2:
            ver, hor=self.romboide_positive(x_m, y_m, t, ll, d, l)
            phi=(np.pi/2-t)
            ver_prime, hor_prime=self.romboide_positive(x_m, y_m, t, np.array([ll[0]-d*np.cos(phi), ll[1]+d*np.sin(phi)]), d, l)
            return ver, hor, ver_prime, hor_prime
        
        elif t<0 and t>-np.pi/2:
            ver, hor=self.romboide_negative(x_m, y_m, t, ll, d, l)
            t_change=-t
            phi=(np.pi/2-t_change)
            ver_prime, hor_prime=self.romboide_negative(x_m, y_m, t, np.array([ll[0]-d*np.cos(phi), ll[1]-d*np.sin(phi)]), d, l)
            return ver, hor, ver_prime, hor_prime
    
    
    """
        Method that writes and generates the new .fits picture that contains the new celestials objects using Astropy library at http://astropy.readthedocs.org/en/latest/io/fits/index.html?highlight=fits#module-astropy.io.fits
        
    """
    def get_simulation_picture(self, ID_number, PICTURE, tension):
        self.simulation_picture = '{}_simulation_{}_{}.fits'.format(PICTURE, ID_number, tension)
        fits.writeto(self.simulation_picture, self.M_field_view)
    
    
    """
        Method that throws the strings corresponding to only one t
        
    """
        
    def cosmic_strings_creator(self, k):
        # Calculate number of strings for this n=k
        N_h, N_s = self.hubble_volume(k)
        
        self.M_field_expanded=0
        
        #print 'Number of Strings: %d' % N_s
        
        #self.N_hubble[k]=N_h
        #self.N_strings[k]=N_s

        # Calculate d hubble_radius
        d=self.hubble_radius(k)
        
        
        # Create matrix with positions x_m and y_m
        
        x = np.linspace(0, self.L_pixels+2*int(d), self.L_pixels+2*int(d)+1)
        y = np.linspace(0, self.L_pixels+2*int(d), self.L_pixels+2*int(d)+1)
        x_m, y_m = np.meshgrid(x,y)
            
        # Create random matrix with zeros of size x_m and y_m
            
        mat=np.zeros_like(x_m)
        #self.M_field_view=np.zeros_like(x_m)
        self.M_field_expanded=np.zeros_like(x_m)

        #Throw strings according to the number of strings N_s
        for string in range (0, N_s):
            
            #print 'number of string  in loop: %d' % string
            
            # Calculate the coordenates of (x,y) randomly at which the string will be plotted
            ll = np.random.random_integers(0, self.L_pixels+2*int(d), (2))
            
            # Calculate the length of the string to throw
            
            l=self.string_proyection(d)
            
            #print 'l=%f' % l
            
            # Throw a random orientation to plot
            
            theta=0
            flag_theta = np.random.rand(1) # Easy Montecarlo to accept negative angles
            #print 'flag_theta= %f' %d
            
            if flag_theta < 0.5:
                theta=np.random.rand(1)*(-np.pi/2)
                #print 'theta=%f' % theta
            elif flag_theta > 0.5:
                theta=np.random.rand(1)*(np.pi/2)
                #print 'theta=%f' % theta
            
            #print 'theta FINAL=%f' % theta

            # evaluate theta and return boolean for mat

            ver, hor, ver_prime, hor_prime = self.angle_evaluation(x_m, y_m, theta, ll, d, l)
            rhom = np.logical_and(hor,ver)
            rhom_prime = np.logical_and(hor_prime,ver_prime)

            #Flag for temperature_anisotropy
            temperature_flag=0
            temperature_flag = np.random.rand(1)

            value_temperature=self.temperature_fluctuation()

            if temperature_flag < 0.5:
                mat[rhom] = value_temperature
                mat[rhom_prime] = -value_temperature
            elif flag_theta > 0.5:
                mat[rhom] = -value_temperature
                mat[rhom_prime] = value_temperature


            #print 'I am here after T'



            #Save the value of the string in the matrix
            self.M_field_expanded=self.M_field_expanded+mat
        
        #Save only the central part of the expanded field of view
        central_part=self.M_field_expanded[int(d):self.L_pixels+int(d), int(d):self.L_pixels+int(d)]
        
        return central_part
        
            

    """
        Method that loop-for all periods of time
    """
        
    def cosmic_looper(self):
        # Make the loop to go for go to n=1 until n=n
        for time in range(1, self.n+1):
            
            print 'Time: %d' % time
            
            central_part=self.cosmic_strings_creator(time)
            self.M_field_view=self.M_field_view+central_part
        
        self.get_simulation_picture(self.n, 'CosmicStringsMap', self.G_mu)

        plt.figure(figsize=(10,10))
        plt.imshow(self.M_field_view, origin='lower')
        plt.colorbar()
        plt.show()




class PowerSpectrum(object):
    
    """
        Constructor that save into attributes main values for obtaining the power spectrum of the map
        
    """
    
    def __init__(self, map):
        self.map=map
        self.fourier_map=[]
    
    """
        Method that plots the 2-D Fourier Transform and shift
        
    """
    
    def fourier_transform(self):
        fourier=fftpack.fft2(self.map)
        self.fourier_map(fourier) #to center in zero
    
    
    
    """
        Calculate the azimuthally averaged radial profile.
        
        image - The 2D image
        center - The [x,y] pixel coordinates used as the center. The default is
        None, which then uses the center of the image (including
        fracitonal pixels).
        
    """
    
    
    def azimuthalAverage(self, image, center=None):

        # Calculate the indices from the image
        y, x = np.indices(image.shape)
    
        if not center:
            center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])
    
        r = np.hypot(x - center[0], y - center[1])

        # Get sorted radii
        ind = np.argsort(r.flat)
        r_sorted = r.flat[ind]
        i_sorted = image.flat[ind]
    
        # Get the integer part of the radii (bin size = 1)
        r_int = r_sorted.astype(int)
        
        # Find all pixels that fall within each radial bin.
        deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
        rind = np.where(deltar)[0]       # location of changed radius
        nr = rind[1:] - rind[:-1]        # number of radius bin
        
        # Cumulative sum to figure out sums for each radius bin
        csim = np.cumsum(i_sorted, dtype=float)
        tbin = csim[rind[1:]] - csim[rind[:-1]]
        
        radial_prof = tbin / nr
        
        return radial_prof
    
    
    def power_spectrum(self):
        self.fourier_transform()
        self.ps2D = np.abs(self.fourier_map)**2
        self.ps1D = self.azimuthalAverage(self.psd2D)
    
    
    def plotter_fft(self):
        
        self.power_spectrum()
        
        py.figure(1)
        py.clf()
        py.imshow( np.log10(self.map), cmap=py.cm.Greys)

        py.figure(2)
        py.clf()
        py.imshow( np.log10(self.ps2D))

        py.figure(3)
        py.clf()
        py.semilogy( self.ps1D )
        py.xlabel('Spatial Frequency')
        py.ylabel('Power Spectrum')

        py.show()























