"""Combines all diferent catalogs for different pointings and quadrants
in the given field. Some targets can have been observed more than
once."""

from astro.io import readtxt
from astro.io import writetxt
import numpy as np

prefix = '/home/ntejos/catalogs/J1005/'

catalog = readtxt(prefix+'catalog_J1005_total.txt', readnames=True)

#clean duplicates
#for i in range(len(ra)):
#    cond = (np.fabs(ra[i]-ra[i+1:]) < ratol) & (np.fabs(dec[i] - dec[i+1:])<dectol)
#    if np.sum(cond)>0:
#        z[i]=-99.


#writetxt('/home/ntejos/catalogs/Q1005/VIMOS.txt',[ra,dec,z,mag])

