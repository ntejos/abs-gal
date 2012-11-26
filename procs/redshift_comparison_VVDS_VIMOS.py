from astro.io import readtxt,writetxt
import numpy as np
import pylab as pl
from barak.coord import unique_radec,match
import pylab as pl
import scipy as sc

VIMOS = readtxt('/home/ntejos/catalogs/J1005/catalog_J1005_total.txt',readnames=True)
VVDS  = readtxt('/home/ntejos/catalogs/VVDS/VVDS_F10.cat',readnames=True,comment='#')

tol = 3.
matches = match(VIMOS.RA_MAPPING,VIMOS.DEC_MAPPING,VVDS.ALPHA_J2000,VVDS.DELTA_J2000,tol)

VIMOS_ids = np.where(matches.ind>-1)[0]
VVDS_ids  = matches.ind[matches.ind>-1]


z_diff = VIMOS.ZGAL[VIMOS_ids] - VVDS.Z[VVDS_ids]
bins = np.arange(-1,1,0.0005)
bins = np.linspace(-0.005,0.005,30)
cond = (VIMOS.ZGAL[VIMOS_ids]>0)

pl.hist(z_diff[cond],bins,histtype='step',label='std='+str(float(format(np.std(z_diff[(np.fabs(z_diff)<0.005)&(np.fabs(z_diff)>0)]),'.5f'))))
pl.legend()
pl.xlabel(r'$\Delta z$',fontsize=18)
pl.ylabel('#',fontsize=18)
pl.title('J1005 VVDS vs VIMOS within '+str(tol)+' arcsec')
pl.show()
#pl.plot(VIMOS.ZGAL[VIMOS_ids],VVDS.Z[VVDS_ids],'bo')
