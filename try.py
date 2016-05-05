from Class_CosmicStrings import CosmicStrings, PowerSpectrum
import time

print 'Starting Simulation'


t = time.time()
simulation=CosmicStrings(L=12.8, G_mu=6.0E-8, n=10, gamma=1, v_bar=15)
simulation.cosmic_looper()
#simulation.cosmic_strings_creator(1)
elapsed = time.time() - t
print elapsed


fft=PowerSpectrum(simulation.M_field_view)
fft.plotter_fft()





