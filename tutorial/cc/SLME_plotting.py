import pymatgen.analysis.solar.slme as slme
import numpy as np
import matplotlib.pyplot as plt 

dir = 0.603 # Direct Bandgap of your material from the band_std
indir= 0.603 # Indirect Bandgap of your material from the band_std (if none given use the direct bandgap twice)

#Get energy and absorption form absorption.dat. Get from sumo-optplot
data = np.loadtxt('absorption.dat', delimiter=' ')
energy = data[:,0]
alpha_cm = data[:,1]

print(energy)

#Test to see if this works for a specific thickness
data = slme.slme(energy, alpha_cm, dir, indir, thickness=1e-05, absorbance_in_inverse_centimeters=True)

print(data)

# Now run for multiple thicknesses 

thickness = np.logspace(-8, -3, 100, endpoint=True)
effSlm = []

for i in thickness:
    eff = data = slme.slme(energy, alpha_cm, dir, indir, thickness=i, absorbance_in_inverse_centimeters=True)
    effSlm.append(eff)

# Plot the SLME results

ax = plt.axes()
plt.plot(thickness, effSlm)
plt.xscale('log')
plt.margins(x=0)
plt.ylim([0, 35])
plt.xlabel('Film Thickness / m', labelpad=5)
plt.ylabel('Max PV Efficiency $(\eta_{Max})$ / %')
ax.set_aspect(0.06)
plt.savefig('slme.pdf', format='pdf')

# You should have ran the Blank_Metric module from github - Chris Savory and have flat_eta_out and lamb_eta_out files

print('Running full Blank SLME plot')

lamb = np.loadtxt('lamb_eta_out', skiprows=0)
thickness_lamb = lamb[:,0]
eff_lamb = [x*100 for x in lamb[:,1]]

flat = np.loadtxt('flat_eta_out', skiprows=0)
thickness_flat = flat[:,0]
eff_flat = [x*100 for x in flat [:,1]]

ax = plt.axes()
plt.plot(thickness, effSlm, label = 'SLME')
plt.plot(thickness_lamb, eff_lamb, label = 'Blank et al, Lambertian Surface')
plt.plot(thickness_flat, eff_flat, label = 'Blank et al, Flat Surface')
plt.xscale('log')
plt.margins(x=0)
plt.ylim([0, 35])
ax.set_aspect(0.06)
plt.legend()
plt.savefig('pveff.pdf', format='pdf')