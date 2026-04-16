import matplotlib
from matplotlib import pyplot as plt
from sumo.cli.dosplot import dosplot
import logging
logging.getLogger('matplotlib.font_manager').disabled = True

def plot_dos(filename, xmin=-3, xmax=3, gaussian=0.05, save=False):
    fig, ax = plt.subplots(figsize=(5,3), dpi=150)
    dosplot(filename=filename, xmin=xmin, xmax=xmax, gaussian=gaussian, plt=plt)

    if save:
        plt.savefig("dos.pdf")

    plt.show()
    return
