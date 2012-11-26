from astro.io import readtxt,writetxt
import numpy as np
import pylab as pl
from barak.coord import unique_radec
import pylab as pl
import scipy as sc
from scipy import optimize

def get_z_diff(catalog,label='ab'):
    """Returns an array with redshift differences between two objects
    observed twice, without considering redshifts failures. It also
    returns arrays for redshift confidence and galaxy template"""
    
    unique, repetition = unique_radec(catalog.RA_MAPPING, catalog.DEC_MAPPING, 0.5)
    z_diff   = np.array([])
    z_mean   = np.array([])

    for r_ind in repetition:
        if len(r_ind)>1:
            #print r_ind
            if (label=='a')|(label=='b'):
                good_redshifts = (catalog.ZGAL_FLAG[r_ind] ==label) & (catalog.ZGAL[r_ind]>0)
            
            elif (label =='ab'):
                good_redshifts = (catalog.ZGAL_FLAG[r_ind] !='c') & (catalog.ZGAL[r_ind]>0)
                
            if np.sum(good_redshifts)==2:#only two measurements
                aux1 = np.diff(catalog.ZGAL[r_ind])
                aux2 = np.mean(catalog.ZGAL[r_ind]) 
                z_diff = np.append(z_diff, aux1[0])
                z_mean = np.append(z_mean, aux2)
                
                    
    return z_diff
    
def get_z_diff_quad(catalog,label='ab'):
    """Returns an array with redshift differences between two objects
    observed twice, without considering redshifts failures. It also
    returns arrays for redshift confidence and galaxy template"""
    
    unique, repetition = unique_radec(catalog.RA_MAPPING, catalog.DEC_MAPPING, 2.0)
    z_diff   = np.array([])
    quad     = np.array([])

    for r_ind in repetition:
        if len(r_ind)>1:
            #print r_ind
            if (label=='a')|(label=='b'):
                good_redshifts = (catalog.ZGAL_FLAG[r_ind] ==label) & (catalog.ZGAL[r_ind]>0)
            
            elif (label =='ab'):
                good_redshifts = (catalog.ZGAL_FLAG[r_ind] !='c') & (catalog.ZGAL[r_ind]>0)
                
            if np.sum(good_redshifts)>=2:
                r_ind2 = np.array(r_ind)
                template = catalog.TEMPLATE[r_ind2[good_redshifts]]
                labels   = catalog.ZGAL_FLAG[r_ind2[good_redshifts]]
                z_gal    = catalog.ZGAL[r_ind2[good_redshifts]]
                for i in range(len(r_ind2[good_redshifts])-1):
                    diff = catalog.ZGAL[r_ind2[good_redshifts]][i]-catalog.ZGAL[r_ind2[good_redshifts]][i+1:]
                    quadrant = catalog.OBJECT[r_ind2[good_redshifts]][i].split('_')[0]
                    for d in diff:
                        z_diff = np.append(z_diff, d)
                        quad   = np.append(quad,quadrant)
    return z_diff,quad
    

def gaussian(x,mean,sigma,amp):
    return amp*np.exp(-(x-mean)**2/2/(sigma**2))
    


fields  = ['J1005']
z_diff_a  = np.array([])
z_diff_b  = np.array([])
z_diff_ab = np.array([])
z_diff_q  = np.array([])
quadrant  = np.array([])

for field in fields:
    print field
    filename = '/home/ntejos/catalogs/%s/catalog_%s_total.txt' %(field,field)
    catalog  = readtxt(filename, readnames=True)
    z_diff_a = np.append(z_diff_a,get_z_diff(catalog,label='a'))
    z_diff_b = np.append(z_diff_b,get_z_diff(catalog,label='b'))
    z_diff_ab = np.append(z_diff_ab,get_z_diff(catalog,label='ab'))
    
    aux1,aux2 = get_z_diff_quad(catalog,label='ab')
    z_diff_q = np.append(z_diff_q,aux1)
    quadrant = np.append(quadrant,aux2)
    


#plots
if 1:
    a_std = float(format(np.std(z_diff_a[np.fabs(z_diff_a)<0.005]),'.5f'))
    b_std = float(format(np.std(z_diff_b[np.fabs(z_diff_b)<0.005]),'.5f'))
    ab_std = float(format(np.std(z_diff_ab[np.fabs(z_diff_ab)<0.005]),'.5f'))


    bins = np.arange(-0.05,0.05,0.0005)
    bins = np.linspace(0,0.005,20)
    pl.hist(np.fabs(z_diff_a),bins,histtype='step',label='a, std='+str(a_std))
    pl.hist(np.fabs(z_diff_b),bins,histtype='step',label='b, std='+str(b_std))
    pl.hist(np.fabs(z_diff_ab),bins,histtype='step',label='a+b, std='+str(ab_std))
    pl.legend()
    pl.xlabel(r'$\Delta z$',fontsize=18)
    pl.ylabel('#',fontsize=18)
    

    #fit gaussian
    guess = [0,0.001,20]
    ydata,xdata = np.histogram(np.fabs(z_diff_ab),bins)
    xdata = xdata[:-1]+(-xdata[0]+xdata[1])/2
    params, params_covariance = sc.optimize.curve_fit(gaussian, xdata, ydata, guess)
    y_err = np.sqrt(ydata+0.75)+1
    params2, params_covariance = sc.optimize.curve_fit(gaussian, xdata, ydata, guess,sigma=y_err)

    #pl.plot(xdata,gaussian(xdata,params[0],params[1],params[2]),label='fit1')
    pl.plot(xdata,gaussian(xdata,params2[0],params2[1],params2[2]),label='Gaussian fit, std='+str(float(format(params2[1],'.5f'))))
    pl.legend()
    pl.show()

if 1: # for quadrants
    std_q1 = float(format(np.std(z_diff_q[(quadrant=='q1')&(np.fabs(z_diff_q)<0.005)]),'.5f'))
    std_q2 = float(format(np.std(z_diff_q[(quadrant=='q2')&(np.fabs(z_diff_q)<0.005)]),'.5f'))
    std_q3 = float(format(np.std(z_diff_q[(quadrant=='q3')&(np.fabs(z_diff_q)<0.005)]),'.5f'))
    std_q4 = float(format(np.std(z_diff_q[(quadrant=='q4')&(np.fabs(z_diff_q)<0.005)]),'.5f'))
    

    bins = np.arange(-0.05,0.05,0.0005)
    bins = np.linspace(-0.005,0.005,30)
    pl.hist(z_diff_q[quadrant=='q1'],bins,histtype='step',label='Q1 std='+str(std_q1))
    pl.hist(z_diff_q[quadrant=='q2'],bins,histtype='step',label='Q2 std='+str(std_q2))
    pl.hist(z_diff_q[quadrant=='q3'],bins,histtype='step',label='Q3 std='+str(std_q3))
    pl.hist(z_diff_q[quadrant=='q4'],bins,histtype='step',label='Q4 std='+str(std_q4))
    pl.legend()
    pl.xlabel(r'$\Delta z$',fontsize=18)
    pl.ylabel('#',fontsize=18)
    pl.title(fields[0])
    pl.show()
