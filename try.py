from Class_CosmicStrings import CosmicStrings, PowerSpectrum
import time

print 'Starting Simulation'




#t = time.time()


#for i in range (0, 50000)
#    simulation=CosmicStrings(L=12.8, G_mu=6.0E-8, n=10, gamma=1, v_bar=15)
#    simulation.cosmic_looper('i')


#elapsed = time.time() - t
#print elapsed




fft=PowerSpectrum(name='CosmicStringsMap_simulation_10_G6Eminus08.fits')
fft.plotter_fft()

