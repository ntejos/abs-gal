from barak.io import readtabfits
from astro.io import writetxt
import numpy as np

Ckms = 299792.458

def read_gal():
    print 'Reading galaxy info'
    M = readtabfits('/home/ntejos/Q0107/catalogs/master20100921.fits')

    deimos = (M.Z_DEIMOS > 1e-3) & (M.ZCONF_DEIMOS > 2)
    gmos   = (M.Z_GMOS   > 1e-3) & (M.ZCONF_GMOS == 'a')
    vimos  = (M.Z_VIMOS  > 1e-3) & (M.ZCONF_VIMOS > 2)
    cfht   = (M.Z_CFHT   > 1e-3) 
    
    isgal = deimos | gmos | vimos | cfht
    print 'N gmos', gmos.sum()
    print 'N vimos', vimos.sum()
    print 'N deimos', deimos.sum()
    print 'N CFHT',cfht.sum()
    print 'total', isgal.sum()
    # correct the VIMOS redshifts to match DEIMOS (add 150 km/s)
    # new correction from gas-galaxy correlation callibration by eye using 1>W>0.5 A. (add 250 km/s)
    zv = M.Z_VIMOS[vimos]
    M.Z_VIMOS[vimos] = zv + (250/Ckms)*(1 + zv)
    # same for GMOS (add 50 km/s)
    # new correction from gas-galaxy correlation callibration by eye using 1>W>0.5 A. (add 100 km/s)
    zg = M.Z_GMOS[gmos]
    M.Z_GMOS[gmos] = zg + (100/Ckms)*(1 + zg)
    # same for DEIMOS (add 0 km/s)
    # new correction from gas-galaxy correlation callibration by eye using 1>W>0.5 A. (add 50 km/s)
    zg = M.Z_DEIMOS[deimos]
    M.Z_DEIMOS[deimos] = zg + (50/Ckms)*(1 + zg)

    # Now consolidate z list, choose z such that deimos > gmos > vimos >
    # cfht
    zg = []
    isd = np.zeros(len(M), bool)
    isv = np.zeros(len(M), bool)
    isg = np.zeros(len(M), bool)
    isc = np.zeros(len(M), bool)
    for i in range(len(M)):
        z = -1
        if deimos[i]:
            z = M.Z_DEIMOS[i]
            isd[i] = 1
        elif gmos[i]:
            z = M.Z_GMOS[i]
            isg[i] = 1
        elif vimos[i]:
            z = M.Z_VIMOS[i]
            isv[i] = 1
        elif cfht[i]:
            z = M.Z_CFHT[i]
            isc[i] = 1
        zg.append(z)

    zg =  np.array(zg)[isgal]
    isd = isd[isgal]
    isv = isv[isgal]
    isg = isg[isgal]
    isc = isc[isgal]

    ra = M[isgal].ALPHA_J2000
    dec = M[isgal].DELTA_J2000
    # index into the master catalogue
    ind = M[isgal].NUMBER - 1
    mag = M[isgal].MAG_AUTO
    template = M[isgal].VIMOS_TEMPLATE
    rec = np.rec.fromarrays([ra,dec,zg,isd,isv,isg,isc,mag,template],
                            names='ra,dec,z,dei,vim,gmos,cfht,mag,template')
    return rec


gals = read_gal()

vimos = gals[gals.vim==1]
writetxt('/home/ntejos/catalogs/Q0107/VIMOS.txt',[vimos.ra,vimos.dec,vimos.z,vimos.mag,vimos.template])
deimos = gals[gals.dei==1]
writetxt('/home/ntejos/catalogs/Q0107/DEIMOS.txt',[deimos.ra,deimos.dec,deimos.z,deimos.mag])
cfht = gals[gals.cfht==1]
writetxt('/home/ntejos/catalogs/Q0107/CFHT.txt',[cfht.ra,cfht.dec,cfht.z,cfht.mag])
gmos = gals[gals.gmos==1]
writetxt('/home/ntejos/catalogs/Q0107/GMOS.txt',[gmos.ra,gmos.dec,gmos.z,gmos.mag])
