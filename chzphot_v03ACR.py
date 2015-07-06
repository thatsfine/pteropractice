from pylab import *
from scipy import *
from numpy import *
import scipy.ndimage as ndi

from scipy.signal import medfilt
from scipy.optimize import leastsq

import astropy.io.fits as pf

# 8-JUL-2011: EFY, SWRI: changed apPhot to call "zoom" instead of homemade "rebin1" to resample subfrms.
# 11-JUL-2011: EFY, SWRI: added pixwt() routine.
# 13-AUG-2014: EFY, SWR, relaced import pyfits with import astropy.io.fits

#-----------------------------------------------------

def splitkwaj(fn, outPath, outFN, nStart):
    '''Loads a 16-frame FITS file obtained from Kwajalein MIT/LL telescopes - saves a separate FITS files'''
    
    # Basic idea:
    # Kwaj FITS files have one primary header and 15 extensions, each with an
    # associated 1024x1024 image. We load these and save them individually with the
    # associated headers.
    
    hdu1 = pf.open(fn)
    
    # Write the image in the primary HDU to disk
    d = hdu1[0].data
    h = hdu1[0].header
    h.update('EXTEND', 'F')
    
    pf.writeto(outPath + outFN + "_" + str(nStart) + ".fits", d, h)
    
    # Now write the 15 extensions as individual FITS files
    
    fnX = ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15"]
    
    for i0 in range(15):
        i = i0 + 1	# skip the first image
        d = hdu1[i].data
        h = hdu1[i].header
        k = h.keys()
        hList = arange(43)+9	# indices of the keywords we want to keep
        hdu0 = pf.PrimaryHDU(d)	# Make a new stub of a header
        for j in hList:
            hdu0.header.update(k[j], h[k[j]])
        pf.writeto(outPath + outFN + "_" + str(nStart+i) + ".fits", d, hdu0.header)
    
    return
    
#-----------------------------------------------------

def parcntrd(im, subR, xg, yg):
    '''Finds more exact coordinates of a star center, by fitting a parabola
    to the row-wise and col-wise sums, the three points including the max and
    its immediate neighbors. yg,xg are the initial guesses for the center, subR
    is the radius of the subframe and yc,xc are better centroid coords.'''
    
    # Step 1: extract a subframe and build row-wise and col-wise sums
    sf = im[(yg-subR):(yg+subR),(xg-subR):(xg+subR)]
    rS = sum(sf, axis=1)
    cS = sum(sf, axis=0)
    
    # Step 2: Find the maxima
    rP = argmax(rS) # index of the peak in the row-sum
    cP = argmax(cS) # index of the peak in the col-sum
    
    # Step 3: Fit parabolas to the maxima.
    iV = arange(2*subR)
    a_r = ((rS[rP-1]-rS[rP])/(iV[rP-1]-iV[rP]) - (rS[rP]-rS[rP+1])/(iV[rP]-iV[rP+1]))/(iV[rP-1]-iV[rP+1])
    b_r = (rS[rP-1]-rS[rP])/(iV[rP-1]-iV[rP]) - a_r * (iV[rP-1]+iV[rP])
    yc = -0.5 * b_r / a_r
    
    a_c = ((cS[cP-1]-cS[cP])/(iV[cP-1]-iV[cP]) - (cS[cP]-cS[cP+1])/(iV[cP]-iV[cP+1]))/(iV[cP-1]-iV[cP+1])
    b_c = (cS[cP-1]-cS[cP])/(iV[cP-1]-iV[cP]) - a_c * (iV[cP-1]+iV[cP])
    xc = -0.5 * b_c / a_c
    
    return(xc + xg - subR, yc + yg - subR)

#-----------------------------------------------------

def cntrd(im, subR, xg, yg):
    '''Finds more exact coordinates of a star center by calculating signal-weighted x & y moments'''
    
    # Step 1: extract a subframe
    sf = im[(yg-subR):(yg+subR),(xg-subR):(xg+subR)]
    
    # Step 2: Calculate the signal-weighted x and y moments
    (yi,xi) = indices((2*subR,2*subR))
    
    yc = sum(yi*sf)/sum(sf)
    xc = sum(xi*sf)/sum(sf)
    
    return(xc + xg - subR, yc + yg - subR)

#-----------------------------------------------------

def cntrd2(im, subR, xg, yg):
    '''Finds more exact coordinates of a star center by calculating signal-weighted x & y moments'''
    
    # Step 1: extract a subframe
    sf = im[(yg-subR):(yg+subR),(xg-subR):(xg+subR)]
    m,n = sf.shape
    for i in range(m):
    	for j in range(n):
    		sf[i][j] = sf[i][j]**2
    
    # Step 2: Calculate the signal-weighted x and y moments
    (yi,xi) = indices((2*subR,2*subR))
    
    yc = sum(yi*sf)/sum(sf)
    xc = sum(xi*sf)/sum(sf)
    
    return(xc + xg - subR, yc + yg - subR)

#-----------------------------------------------------

def gcntrd(im, subR, xg, yg):
    '''Finds more exact coordinates of a star center, by fitting gaussians
    to the row-wise and col-wise sums. yg,xg are the initial guesses for the center, subR
    is the radius of the subframe and yc,xc are better centroid coords. THIS
    ROUTINE DOES NOT FIT FOR A BIAS LEVEL. You should subtract the estimated backgnd first.'''
    
    #STEP 1: Extract the subimage and take row-wise and col-wise sums
    sf = im[(yg-subR):(yg+subR),(xg-subR):(xg+subR)]
    rS = sum(sf, axis=1)
    cS = sum(sf, axis=0)
    
    rv = arange(2.0 * subR) + yg - subR
    cv = arange(2.0 * subR) + xg - subR
    
    #STEP 2: Get initial estimates for the row-wise and col-wise gaussian parameters.
    rpar = gauss1Dguess(rv, rS)
    cpar = gauss1Dguess(cv, cS) # cv, cS are the x,y points that are part of our Gaussian.
    
    #STEP 3: Get robust fits for the two gaussians
    thresh = 5.0
    rpar2 = robofit(gauss1D, rpar, thresh, rv, rS)
    cpar2 = robofit(gauss1D, cpar, thresh, cv, cS)
    
    return( cpar2[1], rpar2[1])
    
#-----------------------------------------------------

def rebin1(im, x2, y2):
    '''Rescales im to dimensions of y2,x2 using bilinear interpolation'''
    
    sz = im.shape
    
    # Now we have to map y2 points evenly within the interval from 0 to sz[0].
    
    (yV, xV) = indices((y2, x2), dtype="float32")
    
    yV *= (float32(sz[0])/y2)
    xV *= (float32(sz[1])/x2)
    
    d = remap(im, xV, yV)
    
    return(d)
    
#-----------------------------------------------------

def remap(im, xm, ym):
    '''Basic idea: Get weights for the neighboring integer positions for the
    fractional xm,ym pixel coordinates.'''
    
    sz = im.shape
    xMax = sz[1]-1
    yMax = sz[0]-1
    
    xm0 = int32(floor(xm))
    xm1a = xm0 + 1
    xm1 = choose((xm1a > xMax), (xm1a, xMax))
    
    ym0 = int32(floor(ym))
    ym1a = ym0 + 1
    ym1 = choose((ym1a > yMax), (ym1a, yMax))
    
    xw0 = 1.0 - (xm - xm0)
    xw1 = 1.0 - xw0
    
    yw0 = 1.0 - (ym - ym0)
    yw1 = 1.0 - yw0
    
    a = roll(roll(xw0*yw0*im[ym0,xm0]+xw1*yw0*im[ym0,xm1]+xw0*yw1*im[ym1,xm0]+xw1*yw1*im[ym0,xm0] , 1, axis = 0), 1, axis = 1)
    
    return(a)

#-----------------------------------------------------

def apPhot(im, xg, yg, fwhm, apR, inR, outR):
    '''Performs aperture photometry for the object near xg, yg. A more precise center
    is found, then the flux within apR is found and a robust estimate of the backgnd
    for that aperture is found.'''
    
    #STEP 0: Estimate the mean and sdev of the image & subtract the mean
    thresh = 5.0
    mn, sd = robomad(im, thresh)
    im2 = im - mn
    
    # STEP 1: Get a better centroid.
    # xc, yc = gcntrd(im2, fwhm, xg, yg)
    xc, yc = cntrd(im2, fwhm, xg, yg)
    #print "Found centers at x,y = ", xc, yc    
    
    # STEP 2: Now make the aperture mask using pixwt()
    sf = 10.0
    apM = pixwt(im2, xc, yc, apR, sf) # inner aperture
    nApPix = sum(apM) # number of pixels in the aperture mask
    
    # STEP 3: Make a list of pixels that lie between inR and outR. Estimate the backgnd
    # from the robust mean of these pixels. No fractional pixels in the backgnd calculation.
    
    (y,x) = indices(im2.shape)
    r = sqrt((x-xc)**2 + (y-yc)**2)
    bkList = logical_and((r >= inR), (r <= outR))
    
    bkAve, bkSDev = robomad(im2, thresh)
    
    
    # STEP 4: Get flux in apR and return 
    sFlux = sum(im2 * apM)
    
    return(sFlux, bkAve*nApPix)
    
#-----------------------------------------------------

def MAD(im):
    '''Returns the Median and Median Absolute Deviation of an array.'''
    
    m = median(im)
    dif = abs(im - m)
    s = median(dif)

    return(m, s)

#-----------------------------------------------------

def robomad(im, thresh=3):
    '''Assumes that the array im has normally distributed background pixels
    and some sources, like stars bad pixels and cosmic rays. Thresh = no. of sigma
    for rejection (e.g., 5).'''
    
    #STEP 1: Start by getting the median and MAD as robust proxies for the mean and sd.
    m,s = MAD(im)
    sd = 1.4826 * s
    
    if (sd < 1.0e-14):
        return(m,sd)
    else:
    
		#STEP 2: Identify outliers, recompute mean and std with pixels that remain.
		gdPix = where(abs(im - m) < (thresh*sd))
		m1 = mean(im[gdPix])
		sd1 = std(im[gdPix])
		
		#STEP 3: Repeat step 2 with new mean and sdev values.
		gdPix = where(abs(im - m1) < (thresh*sd1))
		m2 = mean(im[gdPix])
		sd2 = std(im[gdPix])
		
		return(m2, sd2)
    
#-----------------------------------------------------

def pixwt(im, xc, yc, apR, sf):
    '''Returns the fraction of a pixel in im that is inside the circle of radius
    apR, centered at xc, yc. sf is the oversampling factor for determining the fractional
    coverage of pixels at the perimeter.'''
    
    y,x = indices(im.shape, dtype= "float64")
    y -= yc
    x -= xc
    
    r = sqrt(y**2 + x**2)
    wt = zeros(im.shape)
    
    inR = r < apR
    wt += int8(inR) # inR is set to boolean ONE for pixels totally inside the circle.

    peri = where(abs(r - apR) <= 1.0)
    
    # Now go through the list of fractionally covered pixels and expand
    # each pixel by sf times, then calculate the covered fraction on the 
    # oversampled grid.
    
    xP = x[peri]
    yP = y[peri]
    
    nP = xP.shape[0]   # Number of points on the perimeter
    fracWt = zeros(nP) # Allocate space for the fractional weights we'll calculate
    
    #yB, xB = (indices((sf,sf),dtype="float64")/sf) - (0.5 * (sf - 1.0)/sf)
    yB, xB = (indices((sf,sf),dtype="float64")/sf)
    
    for i in range(nP):
        rB = sqrt((xB-xP[i]-0.5)**2 + (yB-yP[i]-0.5)**2)
        inRB = float64(rB < apR)
        fracWt[i] = sum(inRB)/(sf**2)
    
    wt[peri] = fracWt
    
    return(wt)
    
#-----------------------------------------------------

def gauss1D(p,x):
    '''Returns a 1-D gaussian with amplitude=p[0], centered at p[1] and width = p[2]'''
    
    return(p[0] * exp(-((x-p[1])**2)/p[2]))
    
#-----------------------------------------------------

def gauss1Dguess(x,y):
    '''returns initial guesses for three fitted parameters (amplitude, center and width)
    of a fitted gaussian.'''
    
    # STEP 1: Estimate the center, amplitude and width of the gaussian
    ctr0 = sum(x*y)/sum(y) # signal-weighted average of the index
    
    ymed = medfilt(y,5) # median filter the points
    amp0 = max(ymed) # approximate amplitude of the gaussian
    
    nhi = ymed > (0.5 * amp0) # mask for the points higher than half-max
    wid0 = sqrt(max(x[nhi]) - min(x[nhi])) # 60% of FWHM
    
    p0 = [amp0, ctr0, wid0] # list of initial guesses
    
    return(p0)

#-----------------------------------------------------

def gauss1Dguess2(y, mfilt=5):
    '''returns initial guesses for three fitted parameters (amplitude, center and width)
    of a fitted gaussian. mfilt is an optional argument for the width of a median filter.'''
    
    # STEP 1: Make a median filtered version of the data with the floor subtracted
    ymed = medfilt(y, mfilt)
    yfloor, ystd = robomad(ymed)
    ymf = ymed - yfloor
    
    # STEP 2: Find the maximum point and its location in the filtered array
    amp0 = ymf.max()
    ctr0 = ymf.argmax()
    
    # STEP 3: Identify pixels that are higher than 50% of the max. Subtract the floor first.
    
    nhi = ymf > (0.5 * amp0) # mask for the points higher than half-max
    wid0 = sum(nhi)
    
    p0 = [amp0, ctr0, wid0] # list of initial guesses
    
    return(p0)

#-----------------------------------------------------

def robofit(func, p, thresh, x, y):
    '''Robust fit of the function *func* solving for unknown parameters
    p that best fit the observed (x,y) points. Initial values for p should
    be good initial guesses for the paarameters.'''
    
    #STEP 1: define the error function, which leastsq() tries to minimize.
    err = lambda p,x,y, func: y - func(p,x)
    
    #STEP 2: Get an initial solution from leastsq()
    p1 = leastsq(err, p, args=(x,y, func))
    
    #STEP 3: Look at the robust mean and sdev of the residuals to look for outliers.
    res = err(p1[0], x, y, func)
    mn, sd = robomad(res, thresh)
    gdpix = abs(res - mn) < (thresh * sd) # Consider changing robustmean to return gdpix.
    
    #STEP 4: Repeat call to leastsq, but without outliers.
    x1 = x[gdpix]
    y1 = y[gdpix]
    p2 = leastsq(err, p1[0], args=(x1, y1, func))
    
    return(p2[0])


#-----------------------------------------------------

def robogauss1D(y, thresh=3, mfilt=5):
    '''
    This routine is a quick estimator of a gaussian fit to a 1-D array.
    - It uses a median-filtered version of the data to estimate the gaussian amp, ctr and width
    - It uses a robust estimator to reject outliers
    - It subtracts an estimate of the mean value (from robomad)
    
    y	- the 1-D array to be fit with a gaussian
    thresh - the optional threshold (in std-devs) for rejecting outliers
    mfilt - the optional width for applying an initial median filter. Use 1 for no filtering.
    	    must be an odd integer
    	    
    returns a three element array [amp, ctr, wid]
    
    EFY, SwRI, 25-JUL-2014
    '''
    m,s = robomad(y, thresh)
    y2 = y - m
    
    p0 = gauss1Dguess2(y2, mfilt)
    print "Preliminary Amp, Ctr, Wid: ", p0
    x = arange(y2.size)
    p = robofit(gauss1D, p0, thresh, x, y2)
    print "Robust Amp, Ctr, Wid: ", p

    
    return(p)

#-----------------------------------------------------

def focusWids(d):
    '''
    This routine estimates the widths of a bright spot along the x and y axes.
    
    d	- A 3-D vector of images (imageNo, y, x)
    
    returns pX, pY, where pX = [peak, xCtr, xWid] and pY = [peak, yCtr, yWid]
    
    Example: pX, pY = focusWid(d)
    
    
    EFY, SwRI, 25-JUL-2014
    '''
    
    nF = (d.shape)[0]
    pX = 1.0 * zeros((nF,3))
    pY = 1.0 * zeros((nF,3))
    
    for i in range(nF):
        pY[i,:] = robogauss1D(sum(d[i,:,:], axis=1))
        pX[i,:] = robogauss1D(sum(d[i,:,:], axis=0))
        
    return(pX, pY)

#-----------------------------------------------------


def minixcorr(a,b,ds):
    '''This routine calculates the cross-correlation function of a and b by shifting
    b by +/- range pixels and summing the product of a * b at each shifted position.
    Returns a 2-D array of 1 + 2*ds pixels on a side.
    '''
    
    ccsize = 2*ds + 1
    cc = zeros((ccsize, ccsize), dtype='float64')
    
    for i1 in range(ccsize):
        i2 = i1 - ds
        for j1 in range(ccsize):
            j2 = j1 - ds
            b2 = 1.0 * roll(roll(b,i2,axis=1), j2, axis=0)
            cc[j1,i1] = sum(a*b2)
            
    return(cc)
    
#-----------------------------------------------------

def robostack(stk, sigthresh, gain):
    '''This routine returns a robust average image from a 3D image cube, 
    where each returned pixel is the robust average of all the pixels at
    that x,y location (i.e., a robust mean calculated along the z direction).
    stk  - the 3D input array
    sigthresh - typically 3-5, the sigma level at which to reject outliers
    gain - electrons per counts
    EFY, SwRI, 11-JUN-2012'''
    
    # First calculate the median of the stack and look at outliers with respect to that
    m0 = median(stk, 0) # calculates the median along the first (z) axis
    
    # Now estimate the noise at each pixel from the median image.
    # We'll remove the background first.
    bg, sd = robomad(m0) #, sd is the estimated sky noise
    m1 = m0 - bg
    
    sd_e = gain * sd # we'll work in electrons
    m1_e = gain * abs(m1)
    
    sig = sqrt(sd_e + m1_e)/gain	# Noise in DN
    
    # Now find outliers, replace temporarily with medians
        
    return(m0)
        
    

#-----------------------------------------------------
