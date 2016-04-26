from Class_CosmicStrings import CosmicStrings
import time

t = time.time()
simulation=CosmicStrings(L=12.8, G_mu=1, n=15, gamma=1, v_bar=15)
#simulation.cosmic_looper()
simulation.cosmic_strings_creator(1)
elapsed = time.time() - t
print elapsed



