from Class_CosmicStrings import CosmicStrings, PowerSpectrum
import time
import matplotlib.pyplot as plt

print 'Starting Simulation\n'



t = time.time()



simulation=CosmicStrings(L=12.8, G_mu=6E-08, n=12, gamma=1, v_bar=0.15)
simulation.cosmic_looper('final')


elapsed = time.time() - t
print 'The simulation has lasted %f\n' % elapsed


print 'N_h=%f\n' % simulation.N_hubble[0]
print 'N_s=%f\n' % simulation.N_strings[0]
print 'd_c=%f\n' % simulation.d[0]
print 'L_extended_view=%f\n' % simulation.M_field_expanded.shape[0]



x=[]
y=[]
theta=[]
alpha=[]
flag_r=[]
flag_temperature=[]

for i in range(0, len(simulation.coordinates)):
    x.append(simulation.coordinates[i][0])
    y.append(simulation.coordinates[i][1])
    theta.append(simulation.theta[i][0])
    alpha.append(simulation.alpha[i][0])
    flag_r.append(simulation.r[i][0])
    flag_temperature.append(simulation.temperature_flag[i][0])



plt.figure(1)
plt.hist(alpha, 50, histtype='stepfilled')
plt.xlabel('$\alpha$', labelpad=20, fontsize=36)
plt.ylabel('$Frequency$', fontsize=36)


plt.figure(2)
plt.hist(theta, 50, histtype='stepfilled')
plt.xlabel('$\theta$ (orientation)', labelpad=20, fontsize=36)
plt.ylabel('$Frequency$', fontsize=36)


plt.figure(3)
plt.hist(flag_r, 50, histtype='stepfilled')
plt.xlabel('$flag_r$', labelpad=20, fontsize=36)
plt.ylabel('$Frequency$', fontsize=36)



plt.figure(4)
plt.hist(x, 50, histtype='stepfilled')
plt.xlabel('$coordinate_x$', labelpad=20, fontsize=36)
plt.ylabel('$Frequency$', fontsize=36)


plt.figure(5)
plt.hist(y, 50, histtype='stepfilled')
plt.xlabel('$coordinate_y$', labelpad=20, fontsize=36)
plt.ylabel('$Frequency$', fontsize=36)

plt.figure(6)
plt.hist(flag_temperature, 50, histtype='stepfilled')
plt.xlabel('$flag_{temperature}$', labelpad=20, fontsize=36)
plt.ylabel('$Frequency$', fontsize=36)


plt.show()



fft=PowerSpectrum(name='CosmicStringsMap_simulation_n10_G6e-08_lfinal.fits')
fft.plotter_fft()

