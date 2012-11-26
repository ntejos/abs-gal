from astro.io import readtxt
from astro.io import writetxt
import numpy as np

quadrants = ['Q1','Q2','Q3','Q4']
pointings = ['J2218_p3']

ratol  = 0.001
dectol = 0.001

ra,dec,z,zf,mag = [],[],[],[],[]

for p in pointings:
    for q in quadrants:
        filename = '/home/ntejos/J2218/reduction/mos/%s/%s/catalog.txt' %(p,q)
        colnames = 'object,ra,dec,z,zf,temp,mag,mag_err' 
        try:
            catalog  = readtxt(filename,names=colnames)
        except:
            continue
        print p,q
        for i,obj in enumerate(catalog.object):
            ra.append(catalog.ra[i])
            dec.append(catalog.dec[i])
            z.append(catalog.z[i])
            zf.append(catalog.zf[i])
            mag.append(catalog.mag[i])
    
ra  = np.array(ra)
dec = np.array(dec)
z   = np.array(z)
zf  = np.array(zf)
mag = np.array(mag)

#clean duplicates
for i in range(len(ra)):
    cond = (np.fabs(ra[i]-ra[i+1:]) < ratol) & (np.fabs(dec[i] - dec[i+1:])<dectol)
    if np.sum(cond)>0:
        z[i]=-99.


writetxt('/home/ntejos/catalogs/Q2218/VIMOS.txt',[ra,dec,z,mag])
