import numpy as np
import random
from astro.cosmology import Cosmology,PC,to_xyz
import pylab as pl

def extra_dv(sigma,hole,size=1):
    """Is the extra delta velocity chosen randomly from a given
    distribution (Gaussian with a hole).
    
    Input parameters:
    ---
    sigma:   sigma of the Gaussian
    hole:    size of the hole
    size:    number of random points.
    
    """
    from astro.sampledist import RanDist
    #Gaussian with a hole
    mu       = 0
    x        = np.linspace(-4*sigma, +4*sigma, 1e4)
    dist     = (1.0/(sigma*np.sqrt(2*np.pi)))*np.exp(-(x-mu)**2/(2.0*sigma**2))
    dist     = np.where(np.fabs(x)<hole,0,dist) # Gaussian with a hole
    rand_z   = RanDist(x,dist)
    return rand_z.random(size)

def random_abs_old(absreal,Nrand):
    """From a real absorber catalog it creates a random catalog.  For a
    given real absorber with (z_obs,logN_obs,b_obs) it places it at a
    new z_rand uniformly distributed between the

    zmin = minimun z where a line with logN<logN_obs, and,
    zmax = maximun z where a line with logN<logN_obs. 
    
    (the minimum logN absorbers is placed at a random redshift between
    the original redshift and the redshift of the second minimum logN
    absorber).
    
    """
    from astro.sampledist import RanDist
    
    absreal.sort(order='LOGN') #np.recarray.sort()
    absrand = absreal.repeat(Nrand)
    Ckms  = 299792.458
    DVkms = 2500.
    for i in xrange(len(absreal)):
        if i == 0:
            zmin  = np.min([absreal.ZABS[i],absreal.ZABS[i+1]])
            zmax  = np.max([absreal.ZABS[i],absreal.ZABS[i+1]])
        else:
            cond  = absreal.LOGN <= absreal.LOGN[i]
            zmin  = np.min(absreal.ZABS[cond])
            zmax  = np.max(absreal.ZABS[cond])
        zrand = np.random.uniform(zmin,zmax,Nrand)
        absrand.ZABS[i*Nrand:(i+1)*Nrand] = zrand
    return absrand

def logN_b_to_Wr(logN,b,ion='HI'):
    if ion=='HI':
        #assuming linear part of COG
        return (10**logN)/(1.84*10**14) #W in Angstroms.
    


def random_abs(absreal,Nrand,wa,er,sl=5,R=20000,ion='HI'):
    """From a real absorber catalog it creates a random catalog.  For
    a given real absorber with (z_obs,logN_obs,b_obs) it places it at
    a new z_rand, defined by where the line could have been
    observed. 
    
    Input parameters:
    ---
    absreal: numpy rec array with the absorber catalog.
    Nrand:   number of random lines per real one generated (integer).
    wa:      numpy array of wavelenght covered by the spectrum.  
    er:      numpy array of error in the normalized flux of the spectrum for 
             a given wavelenght.
    sl:      significance level for the detection of the absorption line.
    
    From the error we calculate the Wmin = sl * wa * er / (1+z) / R,
    where z = wa/w0 - 1 (w0 is the rest frame wavelenght of the
    transition) and R is the resolution of the spectrograp. We then
    smooth Wmin with a boxcar (sharp edges). 
    
    For the given absorber we transform (logN_obs,b_obs) to a W_obs assuming 
    linear part of the curve-of-growth. 
    
    We then compute the redshifts where W_obs could have been observed
    according to the given Wmin, and place Nrand new absorbers with
    the same properties as the given one accordingly.
    """
    from astro.sampledist import RanDist
    from scipy.ndimage import gaussian_filter as gf
    
    absreal.sort(order='LOGN') #np.recarray.sort() sorted by column density
    Nrand   = int(Nrand)
    absrand = absreal.repeat(Nrand)
    Ckms  = 299792.458
    if ion=='HI':
        w0    = 1215.67  # HI w0 in angstroms
    z     = wa/w0 - 1.   # spectrum in z coordinates
    
    er   = np.where(er==0,1e10,er)
    er   = np.where(np.isnan(er),1e10,er)
    Wmin = sl*w0*er/R  #sl*wa*er / (1. + z) / R
    Wmin = gf(Wmin.astype(float),10) # smoothed version 
    
    for i in xrange(len(absreal)):
        Wr     = logN_b_to_Wr(absreal.LOGN[i],absreal.B[i],ion='HI')
        zgood  = (Wr > Wmin) & (z>0)
        rand_z = RanDist(z, zgood*1.)
        zrand  = rand_z.random(Nrand)
        absrand.ZABS[i*Nrand:(i+1)*Nrand] = zrand

    return absrand 

      
def random_gal2(galreal,Nrand):
    """From a real galaxy catalog, it creates a random catalog. For a
    given real galaxy with zgal,RA,DEC (and other properties), it
    creates Nrand new random galaxies at (RA,DEC) chosen from the rest
    of the galaxies randomly, it also randomizes zgal to be equal to
    zrand = another galaxy z + Dv coming from a P(v) around z (improve
    description)"""
    
    #np.random.seed(24)
    galrand=galreal.repeat(Nrand)
    Ckms   = 299792.458
    DVkms  = 5000.
    for i in xrange(len(galreal)):
        while True:
            integer_random1 = np.random.randint(0,len(galreal),Nrand) #for zrand
            integer_random2 = np.random.randint(0,len(galreal),Nrand) #for RA,DEC
            equal = (integer_random1 == i) | (integer_random2==i)
            if np.sum(equal)==0: #make sure that galaxy i is not taken 
                break 
        zrand   = galreal.ZGAL[integer_random1]
        #Use an extra displacement at the new random redshift
        zrand   = zrand + extra_dv(DVkms,DVkms,size=Nrand)*(1+zrand)/Ckms
        RArand  = galreal.RA[integer_random2]
        DECrand = galreal.DEC[integer_random2]
        galrand.ZGAL[i*Nrand:(i+1)*Nrand] = zrand
        galrand.RA[i*Nrand:(i+1)*Nrand]   = RArand
        galrand.DEC[i*Nrand:(i+1)*Nrand]  = DECrand
    return galrand
        
def random_gal3(galreal,Nrand):
    """From a real galaxy catalog, it creates a random catalog. For a
    given real galaxy with zgal,RA,DEC (and other properties), it
    creates Nrand new random galaxies at (RA,DEC) chosen from the rest
    of the galaxies randomly, it also randomizes zgal to be equal to
    zrand= another galaxy z but excluding a velocity range from the
    original (improve description)"""
    from astro.sampledist import RanDist

    galrand=galreal.repeat(Nrand)
    Ckms   = 299792.458
    DVkms  = 10000. #dv of exclussion
    
    for i in xrange(len(galreal)):
        #cond = np.fabs(galreal.ZGAL-galreal.ZGAL[i]) * Ckms /(1. + galreal.ZGAL[i])>0*DVkms
        cond = (galreal.MAG > 0) & (galreal.MAG < 90) & (galreal.MAG<=galreal.MAG[i])
        
        if np.sum(cond)>0:
            auxz = galreal.ZGAL[cond]
            integer_random1 = np.random.randint(0,len(auxz),Nrand) #for zrand
            zrand   = auxz[integer_random1]
            #zrand   = zrand + extra_dv(DVkms,DVkms,size=Nrand)*(1+zrand)/Ckms
            zrand = zrand + np.random.uniform(-DVkms/2,DVkms/2,size=Nrand)*(1+zrand)/Ckms
        else:
            zrand = galreal.ZGAL[i]
            zrand = zrand + np.random.uniform(-DVkms/2,DVkms/2,size=Nrand)*(1+zrand)/Ckms

        integer_random2 = np.random.randint(0,len(galreal),Nrand) #for RA,DEC  
        RArand  = galreal.RA[integer_random2]
        DECrand = galreal.DEC[integer_random2]
        galrand.ZGAL[i*Nrand:(i+1)*Nrand] = zrand
        galrand.RA[i*Nrand:(i+1)*Nrand]   = RArand
        galrand.DEC[i*Nrand:(i+1)*Nrand]  = DECrand
    return galrand
        
def random_gal(galreal,Nrand,Nmin=10):
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
    DZ    = 0.01       #delta z for the histogram for getting the spline in z
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
                    #print np.sum(cond), delta_mag2, galreal.MAG[i]
            vals,bins = np.histogram(galreal.ZGAL[cond], bins)
            vals      = gf(vals.astype(float),smooth_scale) # smooth the histogram
            spl       = InterpCubicSpline(0.5*(bins[:-1] + bins[1:]), vals.astype(float))
        
        rvals     = np.linspace(zmin, zmax, 1e4)
        rand_z    = RanDist(rvals, spl(rvals))
        zrand     = rand_z.random(Nrand)
        
        integer_random2 = np.random.randint(0,len(galreal),Nrand) #for RA,DEC  
        RArand  = galreal.RA[integer_random2]
        DECrand = galreal.DEC[integer_random2]
        galrand.ZGAL[i*Nrand:(i+1)*Nrand] = zrand
        galrand.RA[i*Nrand:(i+1)*Nrand]   = RArand
        galrand.DEC[i*Nrand:(i+1)*Nrand]  = DECrand
    return galrand
        
def random_gal_fit(galreal,Nrand,Nmin=50):
    """ For a given galaxy with a given magnitude, it calculates the
    redshift sensitivity function from galaxies brighter than the
    selected one, and places Nrand new galaxies at a random redshift
    given by a empirical function (see zgal_function). For extremely
    bright galaxies (rare) the sensitivity function is calculated from
    at least Nmin (=50) galaxies. This Nmin should be set to include
    volume limited galaxies only. [zgal_function still needs to be
    implemented and so the guesses accordingly.]"""
    
    from astro.sampledist import RanDist
    from astro.fit import InterpCubicSpline
    from scipy import optimize
    Ckms  = 299792.458
    Nmin  = int(Nmin) #minimum number of galaxys for the fit
    zmin  = 0
    zmax  = np.max(galreal.ZGAL) + 0.5
    DZ    = 0.2 #delta z for the histogram for getting the spline in z
    galreal.sort(order='MAG') #np.recarray.sort()
    galrand = galreal.repeat(Nrand)
    
    for i in xrange(len(galreal)):
        
        if i < Nmin: 
            #import pdb; pdb.set_trace()
            vals,bins = np.histogram(galreal.ZGAL[:Nmin], bins=np.arange(zmin, zmax, DZ))
            guess = [np.max(vals),0.7,10.,0.5] # guess parameters for the fit
            xdata = 0.5*(bins[:-1] + bins[1:])
            ydata = vals
            print guess
            params, params_covariance = optimize.curve_fit(zgal_function,xdata,ydata,guess)
            print params
            print ''
            spl   = InterpCubicSpline(xdata, zgal_function(xdata,params[0],params[1],params[2],params[3]))
            
        else:
            cond = (galreal.MAG > 0) & (galreal.MAG < 90) & (galreal.MAG<=galreal.MAG[i])
            vals,bins = np.histogram(galreal.ZGAL[cond], bins=np.arange(zmin, zmax, DZ))
            guess = [np.max(vals),0.7,10.,0.5] # guess parameters for the fit
            xdata = 0.5*(bins[:-1] + bins[1:])
            ydata = vals
            params, params_covariance = optimize.curve_fit(zgal_function,xdata,ydata,guess)
            spl   = InterpCubicSpline(xdata, zgal_function(xdata,params[0],params[1],params[2],params[3]))
            
        rvals     = np.linspace(zmin, zmax, 1e4)
        rand_z    = RanDist(rvals, spl(rvals))
        zrand     = rand_z.random(Nrand)
        
        integer_random2 = np.random.randint(0,len(galreal),Nrand) #for RA,DEC  
        RArand  = galreal.RA[integer_random2]
        DECrand = galreal.DEC[integer_random2]
        galrand.ZGAL[i*Nrand:(i+1)*Nrand] = zrand
        galrand.RA[i*Nrand:(i+1)*Nrand]   = RArand
        galrand.DEC[i*Nrand:(i+1)*Nrand]  = DECrand
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
    er:        numpy array with the normalized QSO spectrum error
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
    def __init__(self, absreal,galreal,wa,er,R=20000,Ngal_rand=10,Nabs_rand=1000,proper=False):
        self.absreal   = absreal   #np array with absorber properties (single ion)
        self.galreal   = galreal   #np array with galaxy properties
        self.wa        = wa        
        self.Ngal_rand = Ngal_rand #Ngal_rand x len(galreal) = NRANDOM (Gal)
        self.Nabs_rand = Nabs_rand #Nabs_rand x len(absreal) = NRANDOM (Abs)
        self.absrand   = random_abs(self.absreal,self.Nabs_rand,wa,er,R=R)
        self.galrand   = random_gal(self.galreal,self.Ngal_rand)
        self.CRA       = np.mean(self.absreal.RA)
        self.CDEC      = np.mean(self.absreal.DEC)
        self.XYZ()
        self.proper    = proper
        if self.proper:
            self.XYZ_proper()
        
        
    def redefine_sample(self,lognmin=0,lognmax=100.,tdist_max=1000):
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

    def addAbs(self, absnew):
        """Adds a new absorber catalog. It has to be of the same type,
        i.e., with the same dtype.names than the first ono used to
        initialize the Field Class."""
        self.absreal = np.append(self.absreal,absnew)
        self.absreal = np.rec.array(self.absreal)
        aux          = random_abs(absnew,self.Nabs_rand) 
        self.absrand = np.append(self.absrand,aux)
        self.absrand = np.rec.array(self.absrand)
        self.XYZ()
        if self.proper:
            self.XYZ_proper()
        self.CRA     = np.mean(self.absreal.RA)
        self.CDEC    = np.mean(self.absreal.DEC)
        
    def addGal(self, galnew):
        """Adds a new galaxy catalog. It has to be of the same type,
        i.e., with the same dtype.names than the first ono used to
        initialize the Field Class."""
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
        #pl.clf()
        bins = np.arange(0,1.5,bs)
        cond = (self.absreal.LOGN <logn_max)&(self.absreal.LOGN >=logn_min)
        pl.hist(self.absreal.ZABS[cond],bins,histtype='step',normed=normed)
        cond = (self.absrand.LOGN <logn_max)&(self.absrand.LOGN >=logn_min)
        pl.hist(self.absrand.ZABS[cond],bins,histtype='step',normed=normed)
        #pl.show()

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
    def __init__(self,field,rbinedges,tbinedges):
        self.fields = [field]
        
        self.DgDg = field.DgDg(rbinedges,tbinedges)
        self.DgRg = field.DgRg(rbinedges,tbinedges)
        self.RgRg = field.RgRg(rbinedges,tbinedges)
        self.DaDg = field.DaDg(rbinedges,tbinedges)
        self.DaRg = field.DaRg(rbinedges,tbinedges)
        self.RaDg = field.RaDg(rbinedges,tbinedges)
        self.RaRg = field.RaRg(rbinedges,tbinedges)
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
        
    def get_xi_1D(self):
        self.DgDg_1D = np.array([np.sum(self.DgDg.T[i]) for i in range(len(self.DgDg[0]))])
        self.DgRg_1D = np.array([np.sum(self.DgRg.T[i]) for i in range(len(self.DgRg[0]))])
        self.RgRg_1D = np.array([np.sum(self.RgRg.T[i]) for i in range(len(self.RgRg[0]))])
        self.DaDg_1D = np.array([np.sum(self.DaDg.T[i]) for i in range(len(self.DaDg[0]))])
        self.DaRg_1D = np.array([np.sum(self.DaRg.T[i]) for i in range(len(self.DaRg[0]))])
        self.RaDg_1D = np.array([np.sum(self.RaDg.T[i]) for i in range(len(self.RaDg[0]))])
        self.RaRg_1D = np.array([np.sum(self.RaRg.T[i]) for i in range(len(self.RaRg[0]))])
        


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

    
