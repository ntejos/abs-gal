import numpy as np
import random
from astro.cosmology import Cosmology,PC,to_xyz
import pylab as pl


def logN_b_to_Wr(logN,b,ion='HI'):
    """I will use the approximation given by Draine book (eq. 9.27),
    whic comes from atomic physics considerations + Rodgers & Williams
    1974 (I couldn't find the reference though)
    
    Input units:
    N is in cm^-2 and b in km/s
    
    Output units:
    Wr in Angstroms

    """
    Ckms  = 299792.458
    logN = float(logN)
    b    = float(b)
    if ion=='HI':
        N     = 10**logN
        t0 = 0.7580 * (N/1e13) * (10/b) # dimensionless (N in cm^-2 and b in km/s)
        if t0 <= 1.25393:
            W = np.sqrt(np.pi) * (b/Ckms) * t0 / (1 + t0/2/np.sqrt(2)) #dimensionless
            return 1215.67 * W  # in angstroms
        else:
            W = (2*b/Ckms)**2 * np.log(t0/np.log(2.)) \
                + (b/Ckms)*(7.616/Ckms) * (t0-1.25393)/np.sqrt(np.pi)
            W = np.sqrt(W)
            return 1215.67 * W  # in angstroms
        
    if ion=='HILyb':
        N     = 10**logN 
        t0 = 0.7580 * (N/1e13) * (10/b) * (0.07912/0.4164) * (1025.72/1215.67) # dimensionless (N in cm^-2 and b in km/s)
        if t0 <= 1.25393:
            W = np.sqrt(np.pi) * (b/Ckms) * t0 / (1 + t0/2/np.sqrt(2)) #dimensionless
            return 1025.72 * W  # in angstroms
        else:
            W = (2*b/Ckms)**2 * np.log(t0/np.log(2.)) \
                + (b/Ckms)*(1.9458/Ckms) * (t0-1.25393)/np.sqrt(np.pi)
            W = np.sqrt(W)
            print 'non-linear part',1215.67 * W, logN,b
            return 1025.72 * W  # in angstroms
        
       
       
    
def compute_Wmin(wa,fl,er,sl=3.,R=20000,FWHM=10,ion='HI'):
    """For a given spectrum and transition, it computes the minimun
    rest-frame equivalent width for that transition to be observed. It
    return a tuple of redshift and Wmin (z,Wmin)"""
    
    from scipy.ndimage import uniform_filter as uf
    
    wa = np.array(wa)
    fl = np.array(fl)
    er = np.array(er)

    if ion=='HI':
        w0 = 1215.67  # HI Lya w0 in angstroms
    if ion=='HILyb':
        w0 = 1025.72 #HI Lyb w0 in angstroms
    z = wa/w0 - 1.   # spectrum in z coordinates
    
    Wmin = sl*w0*er/R/fl  #sl*wa / (1. + z) / R / (S/N)
    Wmin = np.where(Wmin<=0,1e10,Wmin)
    Wmin = np.where(np.isnan(Wmin),1e10,Wmin)
    Wmin = np.where(np.isinf(Wmin),1e10,Wmin)
    Wmin = uf(Wmin.astype(float),FWHM) # smoothed version (uniform prefered over gaussian) 
    return z, Wmin

def random_abs(absreal,Nrand,wa,fl,er,sl=3.,R=20000,FWHM=10.,ion='HI'):
    """From a real absorber catalog it creates a random catalog.  For
    a given real absorber with (z_obs,logN_obs,b_obs) it places it at
    a new z_rand, defined by where the line could have been
    observed. 
    
    Input parameters:
    ---
    absreal: numpy rec array with the absorber catalog.
    Nrand:   number of random lines per real one generated (integer).
    wa:      numpy array of wavelenght covered by the spectrum.
    fl:      numpy array of normalized flux.
    er:      numpy array of error in the normalized flux of the spectrum for 
             a given wavelenght.
    sl:      significance level for the detection of the absorption line.
    R:       resolution of the spectrograph, assumed constant
    FWHM:    Full-width at half maximum in pixels (assumed constant). This 
             parameter defines the smoothing scale for Wmin. 
    ion:     Name of the ion. Function only valid for HI so far.
    
    From the error we calculate the Wmin = sl * wa * er / (1+z) / R,
    where z = wa/w0 - 1 (w0 is the rest frame wavelenght of the
    transition) and R is the resolution of the spectrograp. We then
    smooth Wmin with a boxcar along FWHM pixels. 
    
    For the given absorber we transform (logN_obs,b_obs) to a W_obs assuming 
    linear part of the curve-of-growth. 
    
    We then compute the redshifts where W_obs could have been observed
    according to the given Wmin, and place Nrand new absorbers with
    the same properties as the given one accordingly.
    """
    from astro.sampledist import RanDist
        
    absreal.sort(order='LOGN') #np.recarray.sort() sorted by column density
    Nrand   = int(Nrand)
    absrand = absreal.repeat(Nrand)
    Ckms  = 299792.458
    if ion=='HI':
        z_Lya, Wmin_Lya = compute_Wmin(wa,fl,er,sl=sl,R=R,FWHM=FWHM,ion='HI')
        z_Lyb, Wmin_Lyb = compute_Wmin(wa,fl,er,sl=sl,R=R,FWHM=FWHM,ion='HILyb')
        
    for i in xrange(len(absreal)):
        if absreal.ZABS[i]>np.max(z_Lya): # lines that were observed through Lyb
            Wr   = logN_b_to_Wr(absreal.LOGN[i],absreal.B[i],ion='HILyb')
            z    = z_Lyb
            z    = np.where(z<=z_Lya,-1.,z) #mask out region with Lya coverage
            Wmin = Wmin_Lyb
        else: #lines that were observed through Lya only
            Wr   = logN_b_to_Wr(absreal.LOGN[i],absreal.B[i],ion='HI')
            z    = z_Lya
            Wmin = Wmin_Lya

        zgood  = (Wr > Wmin) & (z>0)
        assert np.sum(zgood)>0, \
            'There are not regions in the spectrum with Wmin<%s A. Addjust significance.' %(Wr)
        
        rand_z = RanDist(z, zgood*1.)
        zrand  = rand_z.random(Nrand)
        absrand.ZABS[i*Nrand:(i+1)*Nrand] = zrand

    return absrand 

      
def random_gal(galreal,Nrand,Nmin=20):
    """ Prefered random galaxy generator. For a given galaxy with a
    given magnitude (and other properties), it calculates the redshift
    sensitivity function from galaxies in a magnitude band around the
    selected one (i.e., including slightly brighter and fainter
    galaxies), and places Nrand new galaxies at a random redshift given
    by a smoothed version of the observed sensitivity function. For
    extremely bright or faint galaxies (rare) the sensitivity function
    is calculated from at least Nmin (=50) galaxies (i.e. the magnitude
    band is increased)."""
    
    from astro.sampledist import RanDist
    from astro.fit import InterpCubicSpline
    from scipy.ndimage import gaussian_filter as gf
    
    Ckms  = 299792.458
    Nmin  = int(Nmin)  #minimum number of galaxys for the fit
    zmin  = np.min(galreal.ZGAL)
    zmax  = np.max(galreal.ZGAL) + 0.1
    DZ    = 0.01     #delta z for the histogram for getting the spline in z
    smooth_scale = 10. #smoothing scale for the histogram (in number
                       #of bins, so depends on DZ)
    galreal.sort(order='MAG') #np.recarray.sort()
    galrand = galreal.repeat(Nrand)
    delta_mag = 0.5 # half of the magnitude bandwidth to generate the z histogram
    
    bins = np.append(np.linspace(0,zmin,20),np.arange(zmin+DZ, zmax, DZ))
    
    for i in xrange(len(galreal)):
        if i < Nmin: 
            vals,bins = np.histogram(galreal.ZGAL[:Nmin], bins)
            vals      = gf(vals.astype(float),smooth_scale) # smooth the histogram
            spl       = InterpCubicSpline(0.5*(bins[:-1] + bins[1:]), vals.astype(float))
            
        else:
            delta_mag2=delta_mag
            while True:
                cond = (galreal.MAG > 0) & (galreal.MAG < 90) 
                cond = cond & (galreal.MAG<=galreal.MAG[i]+delta_mag2)&(galreal.MAG>galreal.MAG[i]-delta_mag2)
                if np.sum(cond)>=Nmin:
                    break
                else:
                    delta_mag2+=0.1
            vals,bins = np.histogram(galreal.ZGAL[cond], bins)
            vals      = gf(vals.astype(float),smooth_scale) # smooth the histogram
            spl       = InterpCubicSpline(0.5*(bins[:-1] + bins[1:]), vals.astype(float))
        
        rvals     = np.linspace(0, zmax, 1e4)
        rand_z    = RanDist(rvals, spl(rvals))
        zrand     = rand_z.random(Nrand)
        galrand.ZGAL[i*Nrand:(i+1)*Nrand] = zrand
    return galrand
        

class Field:
    """The Field class is meant to contain information from a given
    galaxy field (with one or more QSO sightlines containing IGM
    information).
    
    Input parameters:
    ---
    absreal:   catalog of real absorption lines (numpy rec array). It has to have
               dtype.names RA,DEC,ZABS,LOGN,B
    galreal:   catalog of real galaxies (numpy rec array). It has to have
               dtype.names RA, DEC, ZGAL, MAG  
    wa:        numpy array with QSO spectral coverage
    fl:        numpy aray with the QSO spectrum (it can be normalized or not)
    er:        numpy array with the QSO spectrum error (it can be normalized
               or not, but has to be consistent with fl)
    R:         resolution of the QSO spectrum spectrograph
    Ngal_rand: number of random galaxies per real one that will be created.
    Nabs_rand: number of random absorbers per real one that will be created.
    proper:    calculates everything in physical Mpc rather than co-moving (Boolean).

    
    Description of the Class:
    
    It first creates absrand and galrand using random_abs() and
    random_gal() functions. CRA and CDEC are the center position of the
    field, and so will define the coordinate system. Galaxies and
    absorbers (RA,DEC,z) are transformed to (X,Y,Z) co-moving
    coordinates (assuming proper=False). It will then calculate
    cross-pairs DaDg,DaRg,RaDg,RaRg and auto-pairs DgDg, DgRg, RgRg and
    DaDa, DaRa, RaRa where D means 'data' R means 'random', a means
    'absorber' and g means 'galaxy'. The class has also some
    implemented plots that are useful to check for possible problems.

    """
    def __init__(self, absreal,galreal,wa,fl,er,R=20000,Ngal_rand=10,Nabs_rand=1000,proper=False):
        self.absreal   = absreal   #np array with absorber properties (single ion)
        self.galreal   = galreal   #np array with galaxy properties
        self.wa        = wa
        self.fl        = fl
        self.er        = er
        self.Ngal_rand = Ngal_rand #Ngal_rand x len(galreal) = NRANDOM (Gal)
        self.Nabs_rand = Nabs_rand #Nabs_rand x len(absreal) = NRANDOM (Abs)
        self.absrand   = random_abs(self.absreal,self.Nabs_rand,wa,fl,er,R=R)
        self.galrand   = random_gal(self.galreal,self.Ngal_rand)
        self.CRA       = np.mean(self.absreal.RA)
        self.CDEC      = np.mean(self.absreal.DEC)
        self.XYZ()
        self.proper    = proper
        if self.proper:
            self.XYZ_proper()
        
        
    def redefine_sample(self,lognmin=0,lognmax=100.,tdist_max=1000):
        """Redefine the sample getting rid of unwanted
        information. Useful to reduce computational time when
        calculating the pair counts."""
        cond = (self.absreal.LOGN >=lognmin) & (self.absreal.LOGN <lognmax) 
        self.absreal = self.absreal[cond]
        cond = (self.absrand.LOGN >=lognmin) & (self.absrand.LOGN <lognmax) 
        self.absrand = self.absrand[cond]
        cond = np.hypot(self.yg,self.zg)   < tdist_max
        self.galreal = self.galreal[cond]
        cond = np.hypot(self.ygr,self.zgr) < tdist_max
        self.galrand = self.galrand[cond]
        
        self.XYZ()
        if self.proper:
            self.XYZ_proper()

    def apply_redshift_shift(self,lognmin=14.7):
        """Shifts all galaxy redshifts. The shift is determined by the
        difference between redshifts of galaxies close to strong
        absorbers and the redshift of the strong absorption systems."""
        strong_abs = self.absreal[self.absreal.LOGN>=lognmin]
        #TO BE IMPLEMENTED
        

    def XYZ(self):
        """Goes from (RA,DEC,Z) to (X,Y,Z) coordinates"""
        from astro.cosmology import Cosmology,PC
        from barak.coord import radec_to_xyz
        cosmo = Cosmology(H0=70., Om=0.3, Ol=0.7)
        
        Rlos = np.array([cosmo.Dc(self.absreal.ZABS[i])/PC/1e6 for i in xrange(self.absreal.size)])
        xyz  = radec_to_xyz(self.absreal.RA-self.CRA,self.absreal.DEC-self.CDEC).T*Rlos
        self.xa  = xyz[0]
        self.ya  = xyz[1]
        self.za  = xyz[2]
        Rlos = np.array([cosmo.Dc(self.absrand.ZABS[i])/PC/1e6 for i in xrange(self.absrand.size)])
        xyz  = radec_to_xyz(self.absrand.RA-self.CRA,self.absrand.DEC-self.CDEC).T*Rlos
        self.xar = xyz[0]
        self.yar = xyz[1]
        self.zar = xyz[2]
        Rlos = np.array([cosmo.Dc(self.galreal.ZGAL[i])/PC/1e6 for i in xrange(self.galreal.size)])
        xyz  = radec_to_xyz(self.galreal.RA-self.CRA,self.galreal.DEC-self.CDEC).T*Rlos
        self.xg  = xyz[0]
        self.yg  = xyz[1]
        self.zg  = xyz[2]
        Rlos = np.array([cosmo.Dc(self.galrand.ZGAL[i])/PC/1e6 for i in xrange(self.galrand.size)])
        xyz  = radec_to_xyz(self.galrand.RA-self.CRA,self.galrand.DEC-self.CDEC).T*Rlos
        self.xgr = xyz[0]
        self.ygr = xyz[1]
        self.zgr = xyz[2]

    def XYZ_proper(self):
        """From co-moving coordinates goes to physical coordinates
        (dividing by 1+z). It is only meaningful at small distances
        (<100 kpc)"""
        self.yg  = self.yg  / (1. + self.galreal.ZGAL)
        self.zg  = self.zg  / (1. + self.galreal.ZGAL)
        self.ygr = self.ygr / (1. + self.galrand.ZGAL)
        self.zgr = self.zgr / (1. + self.galrand.ZGAL)
        self.ya  = self.ya  / (1. + self.absreal.ZABS)
        self.za  = self.za  / (1. + self.absreal.ZABS)
        self.yar = self.yar / (1. + self.absrand.ZABS)
        self.zar = self.zar / (1. + self.absrand.ZABS)

    def addAbs(self, absnew,wa,fl,er,R=20000):
        """Adds a new absorber catalog. It has to be of the same type
        than the catalog used to initialize the Field Class in the
        first place, i.e., with the same dtype.names than the first
        one."""
        self.absreal = np.append(self.absreal,absnew)
        self.absreal = np.rec.array(self.absreal)
        aux          = random_abs(absnew,wa,fl,er,R,self.Nabs_rand) 
        self.absrand = np.append(self.absrand,aux)
        self.absrand = np.rec.array(self.absrand)
        self.XYZ()
        if self.proper:
            self.XYZ_proper()
        self.CRA     = np.mean(self.absreal.RA)
        self.CDEC    = np.mean(self.absreal.DEC)
        
    def addGal(self, galnew):
        """Adds a new galaxy catalog. It has to be of the same type
        than the catalog used to initialize the Field Class in the
        first place, i.e., with the same dtype.names than the first
        one."""
        self.galreal = np.append(self.galreal,galnew)
        self.galreal = np.rec.array(self.galreal)
        aux          = random_gal(galnew,self.Ngal_rand)
        self.galrand = np.append(self.galrand,aux)
        self.galrand = np.rec.array(self.galrand)
        self.XYZ()
        if self.proper:
            self.XYZ_proper()
        
    def DgDg(self,rbinedges,tbinedges):
        from pyntejos.xcorr.xcorr import auto_pairs_rt
        return auto_pairs_rt(self.xg,self.yg,self.zg,rbinedges,tbinedges)
     
    def DaDa(self,rbinedges,tbinedges):
        from pyntejos.xcorr.xcorr import auto_pairs_rt
        return auto_pairs_rt(self.xa,self.ya,self.za,rbinedges,tbinedges)

    def RgRg(self,rbinedges,tbinedges):
        from pyntejos.xcorr.xcorr import auto_pairs_rt
        return auto_pairs_rt(self.xgr,self.ygr,self.zgr,rbinedges,tbinedges)
    
    def RaRa(self,rbinedges,tbinedges):
        from pyntejos.xcorr.xcorr import auto_pairs_rt
        return auto_pairs_rt(self.xar,self.yar,self.zar,rbinedges,tbinedges)
    
    def DaDg(self,rbinedges,tbinedges):
        from pyntejos.xcorr.xcorr import cross_pairs_rt
        return cross_pairs_rt(self.xa,self.ya,self.za,self.xg,self.yg,self.zg,rbinedges,tbinedges)
    
    def DgRg(self,rbinedges,tbinedges):
        from pyntejos.xcorr.xcorr import cross_pairs_rt
        return cross_pairs_rt(self.xg,self.yg,self.zg,self.xgr,self.ygr,self.zgr,rbinedges,tbinedges)

    def DaRg(self,rbinedges,tbinedges):
        from pyntejos.xcorr.xcorr import cross_pairs_rt
        return cross_pairs_rt(self.xa,self.ya,self.za,self.xgr,self.ygr,self.zgr,rbinedges,tbinedges)
    
    def RaRg(self,rbinedges,tbinedges):
        from pyntejos.xcorr.xcorr import cross_pairs_rt
        return cross_pairs_rt(self.xar,self.yar,self.zar,self.xgr,self.ygr,self.zgr,rbinedges,tbinedges)
    
    def RaDg(self,rbinedges,tbinedges):
        from pyntejos.xcorr.xcorr import cross_pairs_rt
        return cross_pairs_rt(self.xar,self.yar,self.zar,self.xg,self.yg,self.zg,rbinedges,tbinedges)
    
    def plot_zhist_gal(self,mag_min=15,mag_max=25,bs=0.01,normed=True):
        pl.clf()
        bins = np.arange(0,2,bs)
        cond = (self.galreal.MAG <mag_max)&(self.galreal.MAG >=mag_min)
        pl.hist(self.galreal.ZGAL[cond],bins,histtype='step',normed=normed)
        cond = (self.galrand.MAG <mag_max)&(self.galrand.MAG >=mag_min)
        pl.hist(self.galrand.ZGAL[cond],bins,histtype='step',normed=normed)
        pl.show()

    def plot_zhist_abs(self,logn_min=0,logn_max=25,bs=0.01,normed=True):
        pl.clf()
        bins = np.arange(0,1.5,bs)
        cond = (self.absreal.LOGN <logn_max)&(self.absreal.LOGN >=logn_min)
        pl.hist(self.absreal.ZABS[cond],bins,histtype='step',normed=normed)
        cond = (self.absrand.LOGN <logn_max)&(self.absrand.LOGN >=logn_min)
        pl.hist(self.absrand.ZABS[cond],bins,histtype='step',normed=normed)
        pl.show()

    def plot_x(self,bins):
        pl.clf()
        pl.hist(self.xg,bins,histtype='step',normed=True)
        pl.hist(self.xgr,bins,histtype='step',normed=True)
        pl.show()
        
    def plot_yz(self,bins):
        pl.clf()
        pl.hist(np.hypot(self.yg,self.zg),bins,histtype='step',normed=True)
        pl.hist(np.hypot(self.ygr,self.zgr),bins,histtype='step',normed=True)
        pl.show()


class Survey2D2PCF:
    """The Class Survey adds many independent fields together. It
    internally calculates the number of cross- and auto-pairs of data
    and randoms. It also calculates the 2D2PCF using the Landy & Szalay
    estimator, and it estimate the uncertainty with a jacknife
    technique."""

    def __init__(self,field,rbinedges,tbinedges):
        self.fields = [field]
        
        self.DgDg = field.DgDg(rbinedges,tbinedges)
        self.DgRg = field.DgRg(rbinedges,tbinedges)
        self.RgRg = field.RgRg(rbinedges,tbinedges)
        self.DaDg = field.DaDg(rbinedges,tbinedges)
        self.DaRg = field.DaRg(rbinedges,tbinedges)
        self.RaDg = field.RaDg(rbinedges,tbinedges)
        self.RaRg = field.RaRg(rbinedges,tbinedges)
        self.DaDa = field.DaDg(rbinedges,tbinedges)
        self.DaRa = field.DaDg(rbinedges,tbinedges)
        self.RaRa = field.DaDg(rbinedges,tbinedges)
        #self.get_xi_1D()
        
        self.rbinedges = rbinedges
        self.tbinedges = tbinedges

    def addField(self,field):
        self.fields.append(field)
        
        self.DgDg += field.DgDg(self.rbinedges,self.tbinedges)
        self.DgRg += field.DgRg(self.rbinedges,self.tbinedges)
        self.RgRg += field.RgRg(self.rbinedges,self.tbinedges)
        self.DaDg += field.DaDg(self.rbinedges,self.tbinedges)
        self.DaRg += field.DaRg(self.rbinedges,self.tbinedges)
        self.RaDg += field.RaDg(self.rbinedges,self.tbinedges)
        self.RaRg += field.RaRg(self.rbinedges,self.tbinedges)
        self.DaDa += field.DaDa(self.rbinedges,self.tbinedges)
        self.DaRa += field.DaRa(self.rbinedges,self.tbinedges)
        self.RaRg += field.RaRa(self.rbinedges,self.tbinedges)

    def get_xi_1D(self):
        self.DgDg_1D = np.array([np.sum(self.DgDg.T[i]) for i in range(len(self.DgDg[0]))])
        self.DgRg_1D = np.array([np.sum(self.DgRg.T[i]) for i in range(len(self.DgRg[0]))])
        self.RgRg_1D = np.array([np.sum(self.RgRg.T[i]) for i in range(len(self.RgRg[0]))])
        self.DaDg_1D = np.array([np.sum(self.DaDg.T[i]) for i in range(len(self.DaDg[0]))])
        self.DaRg_1D = np.array([np.sum(self.DaRg.T[i]) for i in range(len(self.DaRg[0]))])
        self.RaDg_1D = np.array([np.sum(self.RaDg.T[i]) for i in range(len(self.RaDg[0]))])
        self.RaRg_1D = np.array([np.sum(self.RaRg.T[i]) for i in range(len(self.RaRg[0]))])
        self.DaDa_1D = np.array([np.sum(self.DaDa.T[i]) for i in range(len(self.DaDa[0]))])
        self.DaRa_1D = np.array([np.sum(self.DaRa.T[i]) for i in range(len(self.DaRa[0]))])
        self.RaRa_1D = np.array([np.sum(self.RaRa.T[i]) for i in range(len(self.RaRa[0]))])
        
    def xi_ag_LS(self,sigma=0,jacknife=True,f1=None,f2=None,f3=None):
        """ Returns the abs-gal 2D2PCF using Landy & Szalay
        estimator. It smooths the pair-pair counts with a Gaussian
        kernel with sigma = [rs,ts] (for each direction, see
        scipy.ndimage.gaussian_filter), before computing the
        cross-correlation. It also gives the projected along the LOS
        measurement from DxDy_1D values (not implemented yet)."""
        
        from pyntejos.xcorr.xcorr import W3
        from scipy.ndimage import gaussian_filter as gf
        s = sigma
        Wag,_  = W3(gf(self.DaDg,s),gf(self.RaRg,s),gf(self.DaRg,s),gf(self.RaDg,s),f1=f1,f2=f2,f3=f3)
        
        #jacknife error
        err_Wjk = np.zeros((len(self.rbinedges) - 1, len(self.tbinedges) - 1), float)
        if jacknife:
            for field in self.fields:
                DaDg_aux = self.DaDg - field.DaDg(self.rbinedges,self.tbinedges) 
                RaRg_aux = self.RaRg - field.RaRg(self.rbinedges,self.tbinedges)
                DaRg_aux = self.DaRg - field.DaRg(self.rbinedges,self.tbinedges)
                RaDg_aux = self.RaDg - field.RaDg(self.rbinedges,self.tbinedges)
                Wag_aux,_ = W3(gf(DaDg_aux,s),gf(RaRg_aux,s),gf(DaRg_aux,s),gf(RaDg_aux,s),f1=f1,f2=f2,f3=f3)
                err_Wjk += (Wag - Wag_aux)**2
            N        = len(self.fields)
            err_Wjk = (N - 1.) / N * err_Wjk
            err_Wjk = np.sqrt(err_Wjk)
        return Wag, err_Wjk

    def xi_gg_LS(self,sigma=0,jacknife=True,f1=None,f2=None,f3=None):
        from pyntejos.xcorr.xcorr import W3
        from scipy.ndimage import gaussian_filter as gf

        s = sigma
        W3gg,_ = W3(gf(self.DgDg,s),gf(self.RgRg,s),gf(self.DgRg,s),gf(self.DgRg,s),f1=f1,f2=f2,f3=f3)
        
        #jacknife error
        err_W3jk = np.zeros((len(self.rbinedges) - 1, len(self.tbinedges) - 1), float)
        if jacknife:
            for field in self.fields:
                DgDg_aux = self.DgDg - field.DgDg(self.rbinedges,self.tbinedges) 
                RgRg_aux = self.RgRg - field.RgRg(self.rbinedges,self.tbinedges)
                DgRg_aux = self.DgRg - field.DgRg(self.rbinedges,self.tbinedges)
                W3gg_aux,_ = W3(gf(DgDg_aux,s),gf(RgRg_aux,s),gf(DgRg_aux,s),gf(DgRg_aux,s),f1=f1,f2=f2,f3=f3)
                err_W3jk += (W3gg - W3gg_aux)**2
            N        = len(self.fields)
            err_W3jk = (N - 1.) / N * err_W3jk
            err_W3jk = np.sqrt(err_W3jk)
        return W3gg, err_W3jk
