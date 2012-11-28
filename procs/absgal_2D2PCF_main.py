import numpy as np
from astro.cosmology import Cosmology,PC,to_xyz
from barak.io import readtxt
import numpy.lib.recfunctions as rf
from barak.coord import s2dec
from absgal_2D2PCF_util import Field, Survey2D2PCF
from pyntejos.xcorr.xcorr import *
from scipy.ndimage import gaussian_filter as gf


Ckms = 299792.458
wlya = 1215.6701

cosmo = Cosmology(H0=70., Om=0.3, Ol=0.7)

#Randoms
Ngal_rand = 0.
Nabs_rand = 1000.

# lower edge of smallest bin, bin width, and number of bins
# in comoving Mpc. For radial bins
rmin      = 0
rmax      = 20.
rwidth    = 0.5
rbinedges = np.arange(rmin, rmax+0.5*rwidth, rwidth)
rcbins = 0.5*(rbinedges[:-1] + rbinedges[1:])

# for transverse bins
tmin      = 0
tmax      = 30.
twidth    = 0.5
tbinedges = np.arange(tmin, tmax+0.5*twidth, twidth)
tcbins    = 0.5*(tbinedges[:-1] + tbinedges[1:])

#read absorbers
names  = 'ION,ZABS,ZABS_ERR,B,B_ERR,LOGN,LOGN_ERR'
q1005  = readtxt('/home/ntejos/catalogs/Finn_2012/q1005_fuv_HI.sort',names=names)
q0209  = readtxt('/home/ntejos/catalogs/Finn_2012/q0209_fuv_HI.sort',names=names)
q1357  = readtxt('/home/ntejos/catalogs/Finn_2012/q1357_fuv_HI.sort',names=names)
q0107a = readtxt('/home/ntejos/catalogs/Q0107/A_HI.txt',names=names)
q0107b = readtxt('/home/ntejos/catalogs/Q0107/B_HI.txt',names=names)
q0107c = readtxt('/home/ntejos/catalogs/Q0107/C_HI.txt',names=names)
q1022  = readtxt('/home/ntejos/catalogs/J1022/HI.txt',names=names)

#read spectrum
names = 'wa,fl,er,norm'
q1005_fuv = readtxt('/home/ntejos/COS/q1005/FUV/q1005_fuv_norm.txt',names=names)


#append RA and DEC fields to QSO sightlines
radec = s2dec('10 05 35.24','01 34 45.7')
q1005 = rf.rec_append_fields(q1005,'RA' ,radec[0]*np.ones(len(q1005)))
q1005 = rf.rec_append_fields(q1005,'DEC',radec[1]*np.ones(len(q1005)))
radec = s2dec('02 09 30.7','-04 38 26')
q0209 = rf.rec_append_fields(q0209,'RA' ,radec[0]*np.ones(len(q0209)))
q0209 = rf.rec_append_fields(q0209,'DEC',radec[1]*np.ones(len(q0209)))
radec = s2dec('13 57 26.27','04 35 41.4')
q1357 = rf.rec_append_fields(q1357,'RA' ,radec[0]*np.ones(len(q1357)))
q1357 = rf.rec_append_fields(q1357,'DEC',radec[1]*np.ones(len(q1357)))
radec = s2dec('01 10 13.1','-02 19 52')
q0107a = rf.rec_append_fields(q0107a,'RA' ,radec[0]*np.ones(len(q0107a)))
q0107a = rf.rec_append_fields(q0107a,'DEC',radec[1]*np.ones(len(q0107a)))
radec  = s2dec('01 10 16.2','-02 18 50')
q0107b = rf.rec_append_fields(q0107b,'RA' ,radec[0]*np.ones(len(q0107b)))
q0107b = rf.rec_append_fields(q0107b,'DEC',radec[1]*np.ones(len(q0107b)))
radec  = s2dec('01 10 14.52','-02 16 57.5')
q0107c = rf.rec_append_fields(q0107c,'RA' ,radec[0]*np.ones(len(q0107c)))
q0107c = rf.rec_append_fields(q0107c,'DEC',radec[1]*np.ones(len(q0107c)))
radec = s2dec('10 22 18.99','01 32 18.8')
q1022 = rf.rec_append_fields(q1022,'RA' ,radec[0]*np.ones(len(q1022)))
q1022 = rf.rec_append_fields(q1022,'DEC',radec[1]*np.ones(len(q1022)))


#read galaxies
VVDSF10 = readtxt('/home/ntejos/catalogs/VVDS/VVDS_F10.cat',readnames=True)
VVDSF14 = readtxt('/home/ntejos/catalogs/VVDS/VVDS_F14.cat',readnames=True)
VVDSF22 = readtxt('/home/ntejos/catalogs/VVDS/VVDS_F22.cat',readnames=True) 
VVDSF10 = rf.rename_fields(VVDSF10,{'Z':'ZGAL','MAG_AUTO_I':'MAG','ALPHA_J2000':'RA','DELTA_J2000':'DEC'})
VVDSF14 = rf.rename_fields(VVDSF14,{'Z':'ZGAL','MAG_AUTO_I':'MAG','ALPHA_J2000':'RA','DELTA_J2000':'DEC'})
VVDSF22 = rf.rename_fields(VVDSF22,{'Z':'ZGAL','MAG_AUTO_I':'MAG','ALPHA_J2000':'RA','DELTA_J2000':'DEC'})

names    = 'RA,DEC,ZGAL,MAG'
usecols  = [0,1,2,3]
VIMOSQ1  = readtxt('/home/ntejos/catalogs/Q0107/VIMOS.txt',names=names,usecols=usecols,comment='#')
DEIMOSQ1 = readtxt('/home/ntejos/catalogs/Q0107/DEIMOS.txt',names=names,usecols=usecols,comment='#')
CFHTQ1   = readtxt('/home/ntejos/catalogs/Q0107/CFHT.txt',names=names,usecols=usecols,comment='#')
GMOSQ1   = readtxt('/home/ntejos/catalogs/Q0107/GMOS.txt',names=names,usecols=usecols,comment='#')
VIMOSF1005 = readtxt('/home/ntejos/catalogs/J1005/VIMOS.txt',names=names,usecols=usecols,comment='#')
VIMOSF1022 = readtxt('/home/ntejos/catalogs/J1022/VIMOS.txt',names=names,usecols=usecols,comment='#')


#clean galaxy catalog
galaxies=[VVDSF10,VVDSF14,VVDSF22,VIMOSF1005,VIMOSF1022]
for i in range(len(galaxies)):
    gal  = galaxies[i]
    try:
        cond = (gal.ZGAL_FLAG==4)|(gal.ZGAL_FLAG==24)
        cond = cond | ((gal.ZGAL_FLAG==3)|(gal.ZGAL_FLAG==23))
        cond = cond & (gal.ZGAL>0.0)
    except:
        cond = (gal.ZGAL>0.0)
    #cond = (gal.ZGAL>0.0)
    galaxies[i] = gal[np.where(cond)]
VVDSF10 = galaxies[0]
VVDSF14 = galaxies[1]
VVDSF22 = galaxies[2]
VIMOSF1005 = galaxies[3] 
VIMOSF1022 = galaxies[4]   

#define fields
F1005 = Field(q1005,VVDSF10,q1005_fuv.wa,q1005_fuv.fl,q1005_fuv.er,Ngal_rand=Ngal_rand,Nabs_rand=Nabs_rand)


#redefine the samples
fields  = [F1005]
lognmin = 1.
lognmax = 25.0
for field in fields:
    field.redefine_sample(lognmin=lognmin,lognmax=lognmax,tdist_max=tmax)

print 'defining survey'
S1 = Survey2D2PCF(F1005,rbinedges,tbinedges)

s=2
f1,f2,f3       = (1./Ngal_rand/Nabs_rand,1./Nabs_rand,1./Ngal_rand)
Wag,Wag_err    =  S1.xi_ag_LS(s,jacknife=False,f1=f1,f2=f2,f3=f3)

