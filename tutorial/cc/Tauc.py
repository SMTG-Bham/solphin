from sklearn.linear_model import LinearRegression
from scipy.constants import h, c, eV
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

def load_abs_dat(absdatpath):
    absorption = np.loadtxt(absdatpath)
    energies = absorption.transpose()[0]
    alpha = absorption.transpose()[1]
    return (energies, alpha)

absorption = load_abs_dat("absorption.dat")

f, ax = plt.subplots(1,1)#, figsize=(8,6))

alpha = absorption[1]
energy = absorption[0]

ax.set_yticks([0])
ax.set_ylabel(r"($\alpha h \nu$)$^2$")
ax.set_xlabel(r"$h \nu$ (eV)")

ax.plot(energy, (alpha*energy)**2, c = "C0", label="Calculated")

fit_centre = 1.22 # eV
e_range = np.logical_and(energy < fit_centre+0.1, energy > fit_centre-0.1) # Using a +/-0.1 eV range to fit around this point
x = np.array(energy[e_range]).reshape((-1, 1))
model = LinearRegression().fit(x, (alpha[e_range]*energy[e_range])**2)
r_sq = model.score(x, (alpha[e_range]*energy[e_range])**2)
print("Fitting Performance:")
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
xnew = np.linspace(1,2.5, num=1000)
y_pred = model.intercept_ + model.coef_ * xnew
print(f"Approx. point of interception of fit with x-axis (i.e. predicted Tauc gap): {xnew[np.argmin(abs(y_pred))]:.3f} eV")
ax.plot(xnew, y_pred, c="C3", label="Linear Fit")
ax.plot(energy[e_range], (alpha[e_range]*energy[e_range])**2, c="C5", label="Fit Datapoints", marker="o", markersize=5)
ax.set_title("Tauc Plot")
ax.set_xlim(0.0,2.0)
ax.set_ylim(0,5e9)
ax.legend()
f.savefig("tauc.pdf", bbox_inches="tight")