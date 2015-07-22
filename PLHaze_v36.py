
from scipy import *
from pylab import *
import time as tm
#import pyfits as pf
import astropy.io.fits as pf
import os
from scipy.special import *
from scipy.fftpack import *

from scipy.integrate import cumtrapz
from scipy.integrate import trapz
from scipy.interpolate import interp1d
from scipy.ndimage.interpolation import zoom


# 21 MAY 2013 (EFY, SWRI)
# Routines to generate phase screens of occulting atmospheres and generate the observer's plane intensity field.

# 29 NOV 2013 (EFY, SWRI)
# Corrected an (x,y) -> (y,x) bug in raytr() that led to flipped obs plane intensity fields.

# 24 JAN 2014 (EFY, SWRI)
# Code to generate synth lightcurves for the 2007 Mt John event.

# 13 APR 2014 (EFY, SWRI)
# Added a routine to extract the groundstation positions relative to the shadow center from
# an existing Mt John occultation file.

# 15 JUL 2014
# Added a routine to interpolate T(r) profiles using csplines

# 19 JAN 2015
# Fixed calls to gradient(), which were missing a dimension (effectively 1 km grid)


#-----------------------------------------------------

def T2nu(r, Tr, gSurf, rSurf, PSurf, lam):
    '''Calculates the refractivity as a function of r from the T(r) temperature profile.
    Assumes hydrostatic equilibrium and the ideal gas law.
    
    r = radius (km)
    Tr = temperatures at r's (K)
    
    gSurf = acc. due to gravity at the surface (cgs; gSurf = 65.8 cm/sec^2 on Pluto)
    rSurf = radius of Pluto's surface (km)
    PSurf = surface pressure (cgs = microbars; 10.613 microbars for Pluto)
    
    m = molecular weight (g/mol); m = 28 for N2 gas. WE ASSUME N2!!!
    
    lam = wavelength (microns)
    
    EFY, SwRI, 21-MAY-2013
    '''
    
    k_Bolt = 1.3807e-16 # cgs, Boltzmann's constant
    Avo = 6.022e23 # molecules per mole (Avogadro's number)
    
    m = 28.0
    
    r_cm = r * 1.0e5 # radii in cm
    
    gr = gSurf * (rSurf/r)**2 # gr is the acceleration due to grav. at r.
    Hr = (k_Bolt * Tr) / ((m/Avo)*gr) # scale height as a function of r.
    
    # Now integrate to get pressure, P(r). We'll use the relation:
    # ln(P) = integral(-1/H)dr
    
    lnP1 = cumtrapz(-1.0/Hr, r_cm)
    lnP = concatenate([zeros(1), lnP1]) # We want a leading zero at P(rSurf)
    
    Pn = exp(lnP)
    # Pn is normalized. We need to scale it by a factor of PSurf/Pn(rSurf)
    Pf = interp1d(r, Pn, kind='cubic')
    Pr = Pn * (PSurf/Pf(rSurf))
    
    # Now get the refractivity as a function of height
        
    nu_r = nu_ref_N2(Pr, Tr, lam)
    
    return(Pr, nu_r)

#-----------------------------------------------------

def TP2nu(r, Tr, Pr, lam):
    '''
    Like T2nu(), expect that this function TAKES P(r) as an ARGUMENT instead
    of calculating it from the T(r) profile. The refractivity vs. r is returned.
    
    r = radius (km
    Tr = temperatures at r values (K)
    Pr = pressures at r values (cgs: microbars)
    
    lam = wavelength (microns)

    EFY, SwRI, 2-MAY-2015
    
    '''
    
    nu_r = nu_ref_N2(Pr, Tr, lam)
    
    return(nu_r)

#-----------------------------------------------------

def TBeta(T_H, r_H, r, b):
    '''
    22-MAY-2013, EFY, SwRI
    Returns a paramterized thermal profile,
    T(r) = T_H * (r/r_H)**beta, where
    T_H is the temperature at half-light,
    r_H is the half-light radius, and
    beta is the thermal gradient parameter.
    '''
    
    T = T_H * (r/r_H)**b
    return(T)

#-----------------------------------------------------

def nu2alpha(r, rSurf, nu_r, rMax, dr, sMax, ds):
    '''This routine takes the refractivity profile (nu vs. r) as input and returns the 
    integrated line-of-sight refractivity (alpha vs. r) as output.
    
    r = radii (km)
    rSurf = surface radius (km)
    nu_r = refractivities vs. r
    rMax = the highest radius (km, aka the impact parameter) for which we'll calculate alpha.
    dr = the length of step size for sampling the impact parameters (km).
    sMax = the length of the line of sight chord from the midpoint (km)
    ds = the length of an integration step (km) along a chord through the atmosphere. 
    '''
    
    # STEP 1: build a function to give us nu as a function of r
    nu = interp1d(r, nu_r, kind='linear', bounds_error=False, fill_value=0.0)
    
    # STEP 2: 
    sVec = arange(0.0, sMax, ds) 
    # sVec is the vector of positions along the (half) line of sight chord (km). Multiply by 2 later.
    sVec2 = sVec**2 # Get the square for convenience
    
    # STEP 3: loop over desired range of impact parameters
    r_impVec = arange(rSurf, rMax, dr) # calcualte alpha at all of these impact parameters (km)
    r_imp2 = r_impVec**2
    n_imp = r_impVec.shape[0]
    
    alpha0 = zeros(n_imp)
    
    for i in range(n_imp):
        rs = sqrt(r_imp2[i] + sVec2) # Evaluate refractivities along the line of sight
        alpha0[i] = trapz(nu(rs), sVec) * 2.0 # Integrate the refractivities along the line of sight.
        
    return(r_impVec, alpha0) # Return a paired list of impact parameters (km) and alphas.

#-----------------------------------------------------

def occ_lc(ap):
    '''Calculates the E-field in the observers plane for an occultation by an object with a complex aperture (ap).'''
    
    # Basic Idea:
    # This routine uses Trester's modified aperture function to calculate the E-field
    # in the far field. The modified aperture function is simply the complex aperture
    # times exp((ik/(2z0)) * (x0**2 + y0**2)). Since k = 2pi/lam and npts = lam*z0, this
    # exponential term is equivalent to exp((i*pi/npts) * (x0**2 + y0**2)).   
    
    npts = ap.shape[0] # Assume a square aperture
    n2 = npts/2

    (y,x) = indices((npts,npts)) - 1.0*n2
    eTerm = exp(((pi * 1j)/npts)*((y)**2 + (x)**2))
    M = ap*eTerm
        
    E_obs = fft2(M)
    pObs = real(E_obs * conj(E_obs))
    
    return(roll(roll(pObs, n2, 0), n2, 1))

#-----------------------------------------------------

def solid_ap(FOV, npts, a, b):
    '''
    Makes apertures with elliptical obscurations. Platescale (km/pixel) is FOV/npts
    EFY, SwRI, 3-JUN-2013
    
    FOV	- Field of view (km)
    a,b	- Aperture axes (km)
    npts	- number of points across the FOV
    '''
    FOV2 = 0.5 * FOV
    mpts = linspace(-FOV2, FOV2, npts)
    xmesh,ymesh = meshgrid(mpts, mpts)
    rmesh = sqrt(xmesh**2 + ymesh**2)
    
    inAp = where(((ymesh/b)**2 + (xmesh/a)**2) < 1.0)
	
    ap = ones((npts, npts))
    ap[inAp] *= 0.
    return (ap)

#-----------------------------------------------------

def alpha(v_ref, H_ref, r_ref, b, r):
    '''Returns LAY's alpha funtion, the index of refraction-weighted distance along a chord
    through the atmosphere, for each of the impact parameters suppiled by the user.'''
    
    delta = (H_ref/r_ref) * (r/r_ref)**(1+b)
    
    if (b == -1):
        z_alt = r_ref * log(r/r_ref)
    else:
        z_alt = (r_ref/(1+b))*(1 - (r/r_ref)**(-(1+b)))
        
    v = v_ref * (r/r_ref)**(-b) * exp(-z_alt/H_ref)
    
    alpha = v * r * sqrt(2*pi*delta)*(1 + delta*(9.0-b)/(8.0))
    
    return(alpha)

#-----------------------------------------------------

def nu_ref_N2(P, T, lam):
    '''Returns the refractivity of N2 gas (the real index of refraction minus 1) at
    pressure = P (bar), temperature = T (K) and wavelength = lam (microns).'''
    
    nu_STP = 29.06e-5 * (1.0 + 7.7e-3/(lam**2))
    
    # Now calculate the number density (particles per cm^3). We'll ratio this to
    # Loschmidt's number (2.687e19 particles per cm^3 at STP) to get the refractivity
    # at P and T. Use the ideal gas law: PV = NkT, or number density = N/V = P/(kT).
    # Use cgs units: k = 1.380658e-16 erg/K. Remember that one microbar is one dyne/cm^2.
    
    k = 1.380658e-16
    num_den = P/(k*T)
    
    LosNum = 2.687e19 # particles per cm^3
    
    nu_ref = (num_den/LosNum)*nu_STP
    
    # print "num_den, num_den/LosNum, nu: ", num_den, num_den/LosNum, nu_ref
    
    return(nu_ref)
    
#-----------------------------------------------------

def atm_ap(npts, v_ref, H_ref, r_ref, b, r_solid, waveln, D):
    '''Generates an aperture of phase delays given a model atmosphere for a circular object.
    The ap screen is npts x npts pixels, r_ref is a reference radius, v_ref, H_ref are the
    refractivity and scale height at that radius, b is a thermal profile parameter (T is
    proportional to r**b). Expect waveln in MICRONS and D (distance to the screen) in AU.
    v_ref, H_ref and r_solid are all in km.'''
    
    y,x = indices((npts, npts)) - npts/2
    rpix = sqrt(y**2 + x**2) # radii from the center, in pixels
    
    lam_km = waveln / 10.0**9 # convert microns to km
    D_km = D * 149.6e6
    res = sqrt(lam_km * D_km/npts)
    FOV = sqrt(npts * lam_km * D_km)
    
    print "FOV, resolution (km): ", FOV, res
    
    r_km = res*rpix
    
    k = 2 * pi/lam_km
    
    phase = zeros((npts,npts), dtype="float64")
    
    atm = where(r_km >= r_solid)
    a = alpha(v_ref, H_ref, r_ref, b, r_km[atm])
    
    phase[atm] = k * a
        
    phase_ap = cos(phase) + 1j * sin(phase)
    
    solid = where(r_km < r_solid)
    phase_ap[solid] *= 0.0

        
    return (phase_ap)

#-----------------------------------------------------

def ReadTandP():
    '''
    Read the 2009 presure and temperature profiles.
    '''
    
    PFile = "/Volumes/Crawford/Faure/Work13/PlutoHaze13/LAY_AtmProfile/Pluto/2009_oc/table_heat_7_aat.txt"
    PDat = loadtxt(PFile, skiprows = 37)
    
    return(PDat)
    
#-----------------------------------------------------

def AlphaScreen0(OccFOV, npts, aFun, oblateness, NP_Ang, rSurf):
    '''
    This routine populates a screen with alpha values, where alpha is the 
    integrated line-of-sight refractivity. UNLIKE the other versions of AlphaScreen(), 
    this one (a) doesn't have any transition between the surface and the oblate atmosphere - 
    the surface is assumed to be oblate, and (b) the returned screen has the solid
    body zeroed out, where the solid body is ALSO OBLATE.
        
    OccFOV	- The full extent of the field (km)
    npts	- The number of points across OccFOV
    aFun	- A function that returns alpha(r) given r in km.
    oblateness	- fraction by which the equatorial radius exceeds the polar radius
    NP_Ang	- The orientation of the north pole projected onto the sky plane (deg E of N)
    rSurf	- The surface radius (km)
    
    The resolution between grid points is FOV/(npts-1).
    The function aFun(r) is most likely produced by the interp1d function, e.g.:
    aFun = interp1d(r, alpha, kind='cubic', bounds_error=False, fill_value=0.0)
    
    EFY, SwRI, 24-MAY-2015
    '''
    
    FOV2 = 0.5 * OccFOV
    mpts = linspace(-FOV2, FOV2, npts)
    xmesh,ymesh = meshgrid(mpts, mpts)
    rmesh = sqrt(xmesh**2 + ymesh**2)
    
    thetas = arctan2(ymesh, xmesh)
    
    rScl = 1.0 - cos(2*thetas + NP_Ang*(pi/180.))*oblateness
    
    # Now calculate the scaled radii (to simulate oblateness, etc.)
    rmesh2 = rmesh * rScl
    
    # And the line-of-sight refractivities at the scaled radii
    aScreen = aFun(rmesh2)
    inDisk = where(rmesh2 < rSurf)
    aScreen[inDisk] = 0.0
    ap = ones(aScreen.shape)
    ap[inDisk] = 0.0
    
    return(aScreen, ap)

#-----------------------------------------------------

def AlphaScreen(OccFOV, npts, aFun, tFun, rSurf, r1):
    '''
    This routine populates a screen with alpha values, where alpha is the 
    integrated line-of-sight refractivity.
    OccFOV	- The full extent of the field (km)
    npts	- The number of points across OccFOV
    aFun	- A function that returns alpha(r) given r in km.
    tFun	- A function that scales r as a function of latitude
    rSurf	- The surface radius
    r1		- The radius where the theta-dependent r-scaling takes full effect
    
    The resolution between grid points is FOV/(npts-1).
    The function aFun(r) is most likely produced by the interp1d function, e.g.:
    aFun = interp1d(r, alpha, kind='cubic', bounds_error=False, fill_value=0.0)
    
    EFY, SwRI, 24-MAY-2013
    '''
    
    FOV2 = 0.5 * OccFOV
    mpts = linspace(-FOV2, FOV2, npts)
    xmesh,ymesh = meshgrid(mpts, mpts)
    rmesh = sqrt(xmesh**2 + ymesh**2)
    
    thetas = arctan2(ymesh, xmesh) + pi
    wt = alt_scale(rmesh, rSurf, r1)
    
    # Now calculate the scaled radii (to simulate oblateness, etc.)
    rmesh2 = rmesh * (1.0 + wt*(tFun(thetas) - 1.0))
    
    # And the line-of-sight refractivities at the scaled radii
    aScreen = aFun(rmesh2)
    inDisk = (rmesh <= rSurf)
    aScreen[inDisk] = 0.0
    
    return(aScreen)

#-----------------------------------------------------

def AlphaScreen2(OccFOV, npts, aFun, tFun, rSurf, transBot, transTop, NP_Ang):
    '''
    This routine populates a screen with alpha values, where alpha is the 
    integrated line-of-sight refractivity. Unlike AlphaScreen(), *this* routine
    lets the user specify the orientation of the projected oblate atmosphere
    on the sky plane. Earlier versions of AlphaScreen2 (up through v18) included
    the sub-observer latitude as an argument. V19 introduced two new changes:
    - NP_Ang as the sole tilt argument
    - A linear change to oblateness as a function of altitude, as in AlphaScreen().
    
    Use HORIZONS to get the pole orientation: it is given by NP_Ang.
    
    We transform rmesh_fac to a sky plane projection, but be aware that this
    projection will only give approximate alphas as a function of r_impact.
    
    OccFOV	- The full extent of the field (km)
    npts	- The number of points across OccFOV
    aFun	- A function that returns alpha(r) given r in km.
    tFun	- A function that scales r as a function of latitude
    rSurf	- The surface radius
    transBot, transTop: the altitudes (km) marking the bottom and the top 
         of the transition region from circular to oblate radii.
    NP_Ang	- The orientation of the north pole's sky-plane projection, in deg E of N.
    
    The resolution between grid points is FOV/(npts-1).
    The function aFun(r) is most likely produced by the interp1d function, e.g.:
    aFun = interp1d(r, alpha, kind='cubic', bounds_error=False, fill_value=0.0)
    
    EFY, SwRI, 14-APR-2014
    '''

    FOV2 = 0.5 * OccFOV
    mpts = linspace(-FOV2, FOV2, npts)
    xmesh,ymesh = meshgrid(mpts, mpts)
    rmesh = sqrt(xmesh**2 + ymesh**2)
    
    beta = NP_Ang * pi/180.
    thetas = remainder(arctan2(ymesh, xmesh) + 3*pi + beta, 2*pi)
    wt = alt_scale(rmesh, rSurf+transBot, rSurf+transTop) # wt is zero inside Pluto, increasing to one at rObl
    
    rmesh1 = rmesh * tFun(thetas)
    # Now calculate the scaled radii (to simulate oblateness, etc.)
    rmesh2 = (1.0 - wt) * rmesh + wt * rmesh1
    
    # And the line-of-sight refractivities at the scaled radii
    aScreen = aFun(rmesh2)
    inDisk = (rmesh <= rSurf)
    aScreen[inDisk] = 0.0
    
    return(aScreen)

#-----------------------------------------------------

def AlphaScreen3(OccFOV, npts, aFun, tFun, rSurf, transTop, NP_Ang):
    '''
    This routine populates a screen with alpha values, where alpha is the 
    integrated line-of-sight refractivity. Unlike AlphaScreen(), *this* routine
    lets the user specify the orientation of the projected oblate atmosphere
    on the sky plane. Earlier versions of AlphaScreen2 (up through v18) included
    the sub-observer latitude as an argument. V19 introduced two new changes:
    - NP_Ang as the sole tilt argument
    - A linear change to oblateness as a function of altitude, as in AlphaScreen().
    
    Use HORIZONS to get the pole orientation: it is given by NP_Ang.
    
    We transform rmesh_fac to a sky plane projection, but be aware that this
    projection will only give approximate alphas as a function of r_impact.
    
    OccFOV	- The full extent of the field (km)
    npts	- The number of points across OccFOV
    aFun	- A function that returns alpha(r) given r in km.
    tFun	- A function that scales r as a function of latitude
    rSurf	- The surface radius
    transBot, transTop: the altitudes (km) marking the bottom and the top 
         of the transition region from circular to oblate radii.
    NP_Ang	- The orientation of the north pole's sky-plane projection, in deg E of N.
    
    The resolution between grid points is FOV/(npts-1).
    The function aFun(r) is most likely produced by the interp1d function, e.g.:
    aFun = interp1d(r, alpha, kind='cubic', bounds_error=False, fill_value=0.0)
    
    EFY, SwRI, 14-APR-2014
    '''

    FOV2 = 0.5 * OccFOV
    mpts = linspace(-FOV2, FOV2, npts)
    xmesh,ymesh = meshgrid(mpts, mpts)
    rmesh = sqrt(xmesh**2 + ymesh**2)
    
    beta = NP_Ang * pi/180.
    thetas = remainder(arctan2(ymesh, xmesh) + 3*pi + beta, 2*pi)
    
    rTrans = transTop + rSurf
    inTrans = rmesh <= rTrans
    inDisk = rmesh <= rSurf
    
    m = (tFun(thetas[inTrans]) - 1.0)/(rTrans - rSurf)
    b = 1.0 - m * rSurf
    
    rmesh1 = rmesh * tFun(thetas)
    rmesh1[inTrans] = rmesh[inTrans] * (m * rmesh[inTrans] + b) * (tFun(thetas[inTrans]))
    
    # And the line-of-sight refractivities at the scaled radii
    aScreen = aFun(rmesh1)
    aScreen[inDisk] = 0.0
    
    return(aScreen)

#-----------------------------------------------------

def DiskScreen(npts, OccFOV, rSurf, tFun):
    '''
    This routine makes a simple opacity screen, npts x npts, spanning OccFOV km,
    that is all ONES except within the radius of the object's disk (within rSurf km).
    Should give the same array as solid_ap() when a=b=rSurf.
    '''
    FOV2 = 0.5 * OccFOV
    mpts = linspace(-FOV2, FOV2, npts)
    xmesh,ymesh = meshgrid(mpts, mpts)
    rmesh = sqrt(xmesh**2 + ymesh**2)
    
    thetas = arctan2(ymesh, xmesh) + pi
    rmesh_fac = tFun(thetas)

    ap = ones((npts, npts))
    ap[(rmesh) < rSurf] = 0.0
    # ap[(rmesh*rmesh_fac) < rSurf] = 0.0
    
    return(ap)

#-----------------------------------------------------

def HazeScreen(npts, OccFOV, rSurf, tauSurf, H_haze, H_cap, tFun):
    '''
    This routine makes a simple haze opacity screen, npts x npts, spanning OccFOV km,
    that is all ONES except within the radius of the object's disk (within rSurf km).
    The haze distribution is assumed to an exponential with a scale height of H_haze (km),
    an optical depth of tauSurf at the surface, and a cap at H_cap.
    '''
    FOV2 = 0.5 * OccFOV
    mpts = linspace(-FOV2, FOV2, npts)
    xmesh,ymesh = meshgrid(mpts, mpts)
    rmesh = sqrt(xmesh**2 + ymesh**2)
    
    thetas = arctan2(ymesh, xmesh) + pi
    rmesh_fac = tFun(thetas)

    rmesh2 = rmesh*rmesh_fac
    tauChord = tauSurf * exp((rSurf-rmesh2)/H_haze) * sqrt(2*pi*rmesh2*H_haze)
    
    
    
    ap = exp(-tauChord)
    ap[(rmesh2) > H_cap] = 1.0
    ap[(rmesh2) < rSurf] = 0.0
    
    return(ap)

#-----------------------------------------------------

def raytr(aScreen, ap, Z, OccFOV, OutFOV, nOut, rSurf):
    '''
    This routine sends a rectangular array of rays through a refractive atmosphere.
    The bending angle for each ray (in radians) is given by the gradient of the Alpha
    screen (the integrated line-of-sight refractivity).  The offest for each input ray 
    is calculated and the rays are accumulated onto a virtual CCD. This virtual CCD is
    the returned variable.
    
    aScreen	- The alpha screen
    ap		- The opacity screen
    Z		- The distance from the alpha screen to the observer's plane (AU)
    OccFOV	- The width of the aScreen (km)
    OutFOV	- The desired width of the virtual CCD's FOV (km)
    nOut	- The number of points across the virtual CCD
    rSurf	- The radius of the solid surface (km)
    
    EFY, SwRI, 24-May-2013
    '''
    
    # STEP 1: Build the input array of rays
    
    npts = aScreen.shape[0]	# Assume a square screen.
    FOV2 = 0.5 * OccFOV
    mpts = linspace(-FOV2, FOV2, npts)
    xmesh,ymesh = meshgrid(mpts, mpts)
    rmesh = sqrt(xmesh**2 + ymesh**2)
    
    # ap = ones((npts, npts))
    # ap[rmesh < rSurf] = 0.0
    
    # STEP 2: Get the offsets for each ray
    DX, DY = Offsets(aScreen, Z, OccFOV)

    xObs = xmesh + DX
    yObs = ymesh + DY	# New ray locations in the observer's plane
    
    
    # STEP 3: Accumulate rays onto the virtual CCD
    
    xPix = round((nOut-1.0)*xObs/OutFOV + 0.5*(nOut-1.0))
    xPix[xPix >= nOut] = nOut - 1
    xPix[xPix < 0] = 0

    yPix = round((nOut-1.0)*yObs/OutFOV + 0.5*(nOut-1.0))
    yPix[yPix >= nOut] = nOut - 1
    yPix[yPix < 0] = 0

    vCCD = zeros((nOut, nOut), dtype='float64')
    
    for i in range(npts):
        for j in range(npts):
            vCCD[yPix[i,j], xPix[i,j]] += ap[i,j]
    
    vCCD[0,:] = 0
    vCCD[:,0] = 0
    vCCD[nOut-1,:] = 0
    vCCD[:, nOut-1] = 0
    
            
    return(vCCD)


#-----------------------------------------------------

def raytr3(aScreen, ap, Z, OccFOV, OutFOV, nOut, rSurf):
    '''
    This routine sends a rectangular array of rays through a refractive atmosphere.
    The bending angle for each ray (in radians) is given by the gradient of the Alpha
    screen (the integrated line-of-sight refractivity).  The offest for each input ray 
    is calculated and the rays are accumulated onto a virtual CCD. This virtual CCD is
    the returned variable.
    
    Unlike raytr(), this routine does not simply find the nearsest pixel in the virtual
    cdd and put the entire input ray there. Instead this routine estimates the width and height
    of the input ray on the virtual CCD and finds all of the vCCD pixels that overlap with the
    ray's footprint. The ray's energy is distributed equally among all of the overlapped
    vCCD pixels. We don't bother calculating partial overlap areas - maybe in the next version.
    
    aScreen	- The alpha screen
    ap		- The opacity screen
    Z		- The distance from the alpha screen to the observer's plane (AU)
    OccFOV	- The width of the aScreen (km)
    OutFOV	- The desired width of the virtual CCD's FOV (km)
    nOut	- The number of points across the virtual CCD
    rSurf	- The radius of the solid surface (km)
    
    EFY, SwRI, 9-Dec-2013
    MODS:
    EFY, 22-Feb-2015: Added the second argument (scale) to all gradient() calls.
    '''
    
    # STEP 1: Build the input array of rays
    
    npts = aScreen.shape[0]	# Assume a square screen.
    FOV2 = 0.5 * OccFOV
    mpts = linspace(-FOV2, FOV2, npts)
    xmesh,ymesh = meshgrid(mpts, mpts)
    rmesh = sqrt(xmesh**2 + ymesh**2)
    
    # ap = ones((npts, npts))
    # ap[rmesh < rSurf] = 0.0
    
    # STEP 2: Get the offsets for each ray
    DX, DY = Offsets(aScreen, Z, OccFOV)

    xObs = xmesh + DX
    yObs = ymesh + DY	# New ray locations in the observer's plane
    
    # STEP 3: Estimate the footprint of each ray on the observer's plane.
    # NEED TO FIX GRADIENT W. A LENGTH SCALE (EFY, 19-JAN-2015)
    scale = (1.0 * OccFOV)/npts # scale of DX and DY in km per division.
    fpH = (gradient(DY), scale)[0] # height of the footprint (same units as xObs)
    fpW = (gradient(DX), scale)[1] # width of the footprint (same units as xObs)
    
    # The Left, Right, Top and Bottom bounds in virtual CCD indices.
    xLPix = round((nOut-1.0)*(xObs - 0.5 * fpW)/OutFOV + 0.5*(nOut-1.0))
    xRPix = round((nOut-1.0)*(xObs + 0.5 * fpW)/OutFOV + 0.5*(nOut-1.0))
    yTPix = round((nOut-1.0)*(yObs + 0.5 * fpH)/OutFOV + 0.5*(nOut-1.0))
    yBPix = round((nOut-1.0)*(yObs - 0.5 * fpH)/OutFOV + 0.5*(nOut-1.0))
    
    pixArea = (xRPix - xLPix + 1) * (yTPix - yBPix + 1) # No. of pixels in footprint of ea. ray

    # Truncate any out-of-bounds indices.
    xLPix[xLPix >= nOut] = nOut - 1
    xLPix[xLPix < 0] = 0
    xRPix[xRPix >= nOut] = nOut - 1
    xRPix[xRPix < 0] = 0
    
    yTPix[yTPix >= nOut] = nOut - 1
    yTPix[yTPix < 0] = 0
    yBPix[yBPix >= nOut] = nOut - 1
    yBPix[yBPix < 0] = 0

    
    # STEP 4: Accumulate rays onto the virtual CCD pixels
    
    vCCD = zeros((nOut, nOut), dtype='float64')
    
    for i in range(npts):
        for j in range(npts):
            vCCD[(yBPix[i,j]):(yTPix[i,j]+1), (xLPix[i,j]):(xRPix[i,j]+1)] += ap[i,j]/pixArea[i,j]
    
    vCCD[0,:] = 0
    vCCD[:,0] = 0
    vCCD[nOut-1,:] = 0
    vCCD[:, nOut-1] = 0
    
            
    return(vCCD)


#-----------------------------------------------------

def raytr2(aScreen, ap, Z, OccFOV, OutFOV, nOut):
    '''
    This routine sends a rectangular array of rays through a refractive atmosphere.
    The bending angle for each ray (in radians) is given by the gradient of the Alpha
    screen (aScreen, the integrated line-of-sight refractivity).  The offest for each input ray 
    is calculated and the rays are accumulated onto a virtual CCD. This virtual CCD is
    the returned variable.
    
    This routine differs from raytr() in that the corners of each pixel are mapped onto the
    output array, as opposed to raytr(), where each input ray is accumulated onto a single
    pixel in the output array. In THIS routine, the input pixel are mapped onto a RECTANGLE
    covering the output array. The fractional covereage of output array pixels is calculated
    and the correspnding contribution to each output pixel is accumulated.
    
    
    aScreen	- The alpha screen
    ap		- The opacity screen
    Z		- The distance from the alpha screen to the observer's plane (AU)
    OccFOV	- The width of the aScreen (km)
    OutFOV	- The desired width of the virtual CCD's FOV (km)
    nOut	- The number of points across the virtual CCD
    
    EFY, SwRI, 1-DEC-2013
    '''
        
    # STEP 1: Calculate the physical positions for each input ray. Same as raytr().
    
    npts = aScreen.shape[0]	# Assume a square screen.
    FOV2 = 0.5 * OccFOV
    mpts = linspace(-FOV2, FOV2, npts)
    xmesh,ymesh = meshgrid(mpts, mpts) # Note: returns (x,y), not (y,x) 
    rmesh = sqrt(xmesh**2 + ymesh**2)

    # STEP 2: Get the offsets for each ray. Same as raytr().
    DX, DY = Offsets(aScreen, Z, OccFOV)

    xObs = xmesh + DX
    yObs = ymesh + DY	# New ray locations in the observer's plane
    
    # STEP 3: Estimate width and height of each input pixel on the output array
    # from the SPACING between adjacent DX and DY offests.
    # NEED TO FIX GRADIENT W. A LENGTH SCALE (EFY, 19-JAN-2015)    
    scale = (1.0 * OccFOV)/npts # scale of DX and DY in km per division.
    outH = (gradient(DY), scale)[0]
    outW = (gradient(DX), scale)[1]
    
    
    # STEP 4: Calculate the physical positions of the output pixels.
    
    FOV2 = 0.5 * OutFOV
    mpts = linspace(-FOV2, FOV2, nOut)
    xmOut, ymOut = meshgrid(mpts, mpts) # physical positions (km) of pixel centers in the output plane.
    
    dOut = OutFOV/(nOut-1.0) # Spacing between pixels in the output plane
    
    
    # STEP 5: Loop over each input ray. Sum the footprints on the obs plane.
    
    vccd = xmOut * 0.0 # Allocate the virtual CCD
    
    for i in range(npts):
        for j in range(npts):
            ftprint = getfootprint(abs(xmOut-xObs[j,i]), abs(ymOut-yObs[j,i]), dOut, dOut, outW[j,i], outH[j,i])
            vccd += ftprint
            
    return(vccd)

#-----------------------------------------------------

def getfootprint(Sx, Sy, w1, h1, w2, h2):
    '''
    This routine calculates the fractional overlap of rectangle 2 over rectangle 1.

    Sx, Sy: The separations between the centers of R1 and R2. 2-D arrays for the pixels in the obs plane.
    w1, h1: The width and height of rectangles in the obs plane. Same for all pixels, hence scalars.
    w2, h2: The width and height of the input ray's footprint in the observer's plane
    
    EFY, SwRI, 6 DEC 2013.
    '''
    
    # STEP 1: Linear equations for overlap
    # Overlaps are linear functions of Sx and Sy for a given range of Sx and Sy values. Here are the slopes
    # and intercepts for the horizontal and vertical cases.
    
    mx = -1.0/w1
    bx = (w1 + w2)/(2.0 * w1)
    
    my = -1.0/h1
    by = (h1 + h2)/(2.0 * h1)
    
    # STEP 2: Finding pixels that are in one of three cases: no overlap, partial overlap, and maximum overlap.
    
    # Begin by assuming that all pixels are in the no overlap case.
    
    fpX = 0.0 * Sx # Sx has the same dimensions as the observer's plane grid.
    fpY = 0.0 * Sx # Sx has the same dimensions as the observer's plane grid.
    
    # Now find the pixels that are in a state of partial overlap.
    
    partial_x = Sx < (0.5 * (w1 + w2))
    partial_y = Sy < (0.5 * (h1 + h2))
    
    # Some fraction of these will be in a state of maximum overlap.
    
    maxover_x = Sx <= (0.5 * abs(w2 - w1))
    maxover_y = Sy <= (0.5 * abs(h2 - h1))
    
    
    # Now compute the fractional overlaps for the partial and the maximum overlap cases. We'll
    # calculate the x and y fractional overlaps, then return their product.
    
    fpX[partial_x] = mx * Sx[partial_x] + bx
    fpY[partial_y] = my * Sy[partial_y] + by
    
    max_xoverlap = 1.0
    max_yoverlap = 1.0
    if (w2 < w1): max_xoverlap = w2/w1
    if (h2 < h1): max_yoverlap = h2/h1
    
    fpX[maxover_x] = max_xoverlap
    fpY[maxover_y] = max_yoverlap
    
    ftprint = fpX * fpY
    
    return(ftprint)
    

#-----------------------------------------------------

def Offsets(a0, Z, FOV):
    '''Calculate offsets (DX, DY) arrays
    a0	- array of integrated refractivities
    Z	- distance (AU)
    FOV	- field of view (km)
    '''
    
    npts = a0.shape[0]
    
    Zkm = Z * 149.6e6 # distance in km
    
    pixScale = npts/FOV
    scale = (1.0 * FOV)/npts
    
    # NEED TO FIX GRADIENT W. A LENGTH SCALE (EFY, 19-JAN-2015)
    ga = gradient(a0, scale)
    
    DX = ga[1] * Zkm * pixScale
    DY = ga[0] * Zkm * pixScale  # Correct: the derivative wrt y is the [0] index.
    
    return(DX, DY)

#-----------------------------------------------------

def backtrace(px, py, hood, a0, Z, FOV):
    '''This routine traces points in the observer's plane back to their origins in
    the aperture plane. It returns a boolean mask marking aperture plane points that
    are source points mapping to px,py in the observer's plane.
    
    px,py	- location of the observer's plane point (in km)
    hood	- Neighborhood, the distance (km) around px,py where we will identify rays in the ap plane
    a0		- array of integrated refractivities
    Z		- distance (AU)
    FOV		- field of view (km)
    '''
    
    npts = a0.shape[0]
    Zkm = Z * 149.6e6 # distance in km
    pixScale = npts/FOV
    
    FOV2 = 0.5 * FOV
    mpts = linspace(-FOV2, FOV2, npts)
    xmesh,ymesh = meshgrid(mpts, mpts)

    DX,DY = Offsets(a0, Z, FOV)
    yObs = ymesh+DY
    xObs = xmesh+DX
    
    srcRays = sqrt((xObs-px)**2 + (yObs-py)**2) < hood
    
    return(srcRays)
    

#-----------------------------------------------------

def T2Pv_N2b(T):
    '''
    This routine calculates the vapor pressure of beta-phase N2 ice at a given T.
    Based on coefficients from Brown and Ziegler.
    EFY, SwRI, 2-JUN-2013
    T	- temperature (K)
    Pv_mubar	- returns vapor poressure (cgs units: microbar)
    '''
    
    MicroBarPerTorr = 1332.2
    N2b=([1.64302619*10**1, -6.50497257*10**2,  -8.52432256*10**3,   1.55914234*10**5,  -1.06368300*10**6,  0.0,  0.0])
    
    Pv_torr = exp(N2b[0]+N2b[1]/T+N2b[2]/T**2+N2b[3]/T**3+N2b[4]/T**4+N2b[5]/T**5+N2b[6]/T**6)
    Pv_mubar = Pv_torr * MicroBarPerTorr
    
    return(Pv_mubar)


#-----------------------------------------------------

def XtendP(r,P, rTop, Nr):
    '''
    This routine simply extrapolates an (r,P) curve by assuming the same scale height as the last
    few points and extending an exponential pressure curve to an altitude of rTop.
    EFY, SwRI, 2-JUN-2013
    
    r	- radius vector (km)
    P	- pressure vector (microbar)
    Nr	- number of points to place between the the max of r[] and rTop
    rTop	- Extend P to rTop (km)
    '''
    
    H = median((r[-5:-2]-r[-4:-1])/log(P[-4:-1]/P[-5:-2]))
    
    rX = (arange(Nr) + 1.0) * (rTop - r[1])/Nr+ r[-1]
    PX = P[-1] * exp((r[-1]-rX)/H)
    
    rX = concatenate([r,rX])
    PX = concatenate([P,PX])
    
    
    return(H, rX, PX)
    
    
#-----------------------------------------------------

def XtendT(r, T, rTop, Nr, b = 1000.):
    '''
    This routine extends a temperature profile by determining _b_ near the top of the 
    given T(r) profile and creating a T(r) = T_ref * (r/r_ref)^b profile for points above
    the given T(r) profile.
    EFY, SwRI, 3-JUN-13
    
    r	- radius vector for given profile (km)
    T	- temperature vector for a given profile
    rTop	- Extend T(r) up to rTop
    Nr	- number of points to place between the the max of r[] and rTop
    b	- parametric exponent in T(r) - T_ref * (r/r_ref)^b 
    '''
    
    T_ref = T[-1]
    r_ref = r[-1]	# choose the last element of the r and T vectors as the reference values
    
    if (b > 999): b = median(log(T[-4:-1]/T_ref) / log(r[-4:-1]/r_ref)) # Assume no user would pick b > 999
    
    rX = (arange(Nr) + 1.0) * (rTop - r[1])/Nr+ r[-1]
    TX = T_ref * (rX/r_ref)**b
    
    rX = concatenate([r,rX])
    TX = concatenate([T,TX])
    
    
    return(b, rX, TX)

#-----------------------------------------------------

def rebin(a, *args):
    '''rebin ndarray data into a smaller ndarray of the same rank whose dimensions
    are factors of the original dimensions. eg. An array with 6 columns and 4 rows
    can be reduced to have 6,3,2 or 1 columns and 4,2 or 1 rows.
    example usages:
    >>> a=rand(6,4); b=rebin(a,3,2)
    >>> a=rand(6); b=rebin(a,2)
    '''
    shape = a.shape
    lenShape = len(shape)
    factor = asarray(shape)/asarray(args)
    evList = ['a.reshape('] + \
             ['args[%d],factor[%d],'%(i,i) for i in range(lenShape)] + \
             [')'] + ['.sum(%d)'%(i+1) for i in range(lenShape)] + \
             ['/factor[%d]'%i for i in range(lenShape)]
    print ''.join(evList)
    return eval(''.join(evList))
    
#-----------------------------------------------------

def r_scale(theta0, rVal):
    '''
    Returns a theta-dependent scaling factor in the form of a function,
    theta0 - angles from 0 to 2pi
    rVal - radius scaling values at each angle theta
    '''
    
    tFun = interp1d(theta0, rVal, kind = 'linear')
    return(tFun)
    
    

#-----------------------------------------------------

def alt_scale(r, rSurf, rObl):
    '''
    Returns an altitude-dependent scaling factor in the form of a function,
    rSurf - surface radius (km)
    rObl - radius at which the oblateness scaling is in complete effect
    '''
    
    m = 1.0/(rObl - rSurf)
    b = -m * rSurf
    wt = m * r + b
    wt[r >= rObl] = 1.0
    wt[r <= rSurf] = 0.0
        
    return(wt)

#-----------------------------------------------------

def r_stretched(r, rSurf, rObl, tFun, theta):
    '''
    Returns an altitude-dependent scaling factor in the form of a function,
    rSurf - surface radius (km)
    rObl - radius at which the oblateness scaling is in complete effect
    '''
    
    inDisk = r < rSurf
    inObl = r > rObl
    
    
    rf = (r - rSurf)/(rObl - rSurf) # The radius-dependent term that scales the stretch
    rf[inObl] = 1.0
    rf[inDisk] = 0.0
    
    sf = 1.0 + rf * (tFun(theta) - 1.0)
    r2 = r * sf
        
    return(r2)

#-----------------------------------------------------

def gndpos(occFile, f0, g0):
    '''
    This routine reads the FITS table file for the Mt John occultation (thanks, Leslie) and
    extracts the fNOM an gNOM columns. The ground station position, referenced to the cetner of
    the shadow, are given by -(fNom - f0) and -(gNom - g0).
    
    occFile	- the path to the Mt John occultation FITS Table file
    f0, g0	- the offsets to the nominal f and g offsets (km)
    
    doc_fnam	- the ground station f position relative to the center of the shadow (km)
    doc_gnam	- the ground station g position relative to the center of the shadow (km)
    tsec		- times asscoiated with gnd station offsets doc_fnom and doc_gnom
    
    '''
    
    doc_fits = pf.open(occFile)
    doc_tbl = doc_fits[1].data
    FNOM = doc_tbl.field(3)
    GNOM = doc_tbl.field(4)
    
    doc_fnom = -(FNOM - f0) # The ground station offset from the shadow center (km)
    doc_gnom = -(GNOM - g0)
    
    tsec = doc_tbl.field(1)
    
    return(doc_fnom, doc_gnom, tsec)

#-----------------------------------------------------

def chuzrays(aScr, Z, FOV, minD, maxD, outR):
    '''
    This routine identifies rays in the input grid which have a final position in the output grid
    (the observer's plane) that lies within a certain radius (outR) of the shadow center. The 
    input and output coordinates are returned.
    
    aScr	- the alpha screen
    minD, maxD	- the radius in the aperture plane from which we select rays (km)
    
    outputs:
    iX, iY	- Ray coordinates in the input plane (km)
    oX, oY	- Ray coordinates in the output plane
    
    EFY, SwRI, 16 MAY 2014
    iX,iY,oX,oY = chuzrays(aScr, Z, FOV, minD, maxD, outR) 
    pts = chuzrays(aScr, FOV, minD, maxD, outR)
    '''
    
    npts = aScr.shape[0] # Assume that aScr is a square array
    FOV2 = 0.5*FOV
    Zkm = Z * 149.6e6 # distance in km
    pixScale = npts/FOV    
    
    mpts = linspace(-FOV2, FOV2, npts)
    xmesh,ymesh = meshgrid(mpts, mpts)
    rmesh = sqrt(xmesh**2 + ymesh**2)
    
    DX, DY = Offsets(aScr, Z, FOV)
    xObs = xmesh + DX
    yObs = ymesh + DY	# New ray locations in the observer's plane
    rObs = sqrt(xObs**2 + yObs**2)
    
    selectRays = (rmesh > minD) & (rmesh < maxD) & (rObs < outR)
    
    iX = xmesh[selectRays]
    iY = ymesh[selectRays]
    oX = xObs[selectRays]
    oY = yObs[selectRays]
    pts = array((iX, iY, oX, oY))
    #return(iX,iY,oX,oY)
    return(pts)

#-----------------------------------------------------

def DoFO(dTdr=None):
    '''
    oblateness	- the fraction by which the equatorial radius exceeds the average and the polar radius is less.
    dg0			- change in the g0 parameter (km). New g0 = g0 (from LAY's plot_w_lat.pro) + dg0
    dTdr		- An optional argument. If present dTdr follows this profile above T_max.
    
    pa, pixf0, pixg0, lc, tFOV, rd, Tp, aScr, ap = DoFO(oblateness, dg0, dTdr)
    '''

    startT = tm.time()
    
    # Some parameters describing the oblateness and geometric shadow center.
    dg0 = 0.0
    #oblateness = 0.03
    oblateness = 0.0
    transBot = 1.0 # The bottom and top of the transition region from circular to 
    transTop = 2.0 # stretched (oblate) radii.
    
    
    platform = 3 # 1 = pico, 2 = ajax, 3 = flume, 4 = lloyd
    
    # Load an occultation lightcurve (contains fnom and gnom) and 
    # load an atmospheric profile

    if (platform == 1):
        print "Finding files on Pico"
        doc_pn = '/Volumes/Manoa/Felix/Work12/Pluto_Haze/lc01_MTJ_Doc_3.fits'
        d = loadtxt('/Volumes/Manoa/Felix/Work13/PlutoHaze13/LAY_AtmProfile/Pluto/2009_oc/table_heat_7_aat.txt', skiprows = 37)
    elif (platform == 2):
        print "Finding files on ajax"
        doc_pn = '/Volumes/Firescrew/Work14/PlutoHaze14/lc01_MTJ_Doc_3.fits'
        d = loadtxt('/Volumes/Firescrew/Work14/PlutoHaze14/table_heat_7_aat.txt', skiprows = 37)
    elif (platform == 3): 
        print "Finding files on flume"
        doc_pn = '/Volumes/Felix/Work14/PlutoHaze14/lc01_MTJ_Doc_3.fits'
        d = loadtxt('/Volumes/Felix/Work14/PlutoHaze14/table_heat_7_aat.txt', skiprows = 37)
    elif (platform == 4): 
        print "Finding files on lloyd"
        doc_pn = '/d1/efy/work14/PLHaze/20090901_frOlkin/lc01_MTJ_Doc_3.fits'
        d = loadtxt('/d1/efy/work14/PLHaze/Pluto/2009_oc/table_heat_7_aat.txt', skiprows = 37)
        
    NP_Ang = 247.59
    #NP_Ang = 0.0
    
    f0 = 499.86621 # From LAY's "plot_w_lat.pro"
    g0 = -844.97886 + dg0 # Same source

    rSurf = 1178.645

    # Check to see if the user supplied a value for dTdr (defaults to None). If so,
    # call t1, r1 = newDtdr(d, dTdr)
    
    d_sm = 10.0 # the altitude over which the T profile is altered in newDtdr().
    if (dTdr != None): # the user supplied a slope for the upper atm.
        t1, r1 = newDtdr(d, dTdr, d_sm)
    else:
        t1 = d[:,1]
        r1 = d[:,0]
    
		#r1 = d[:,1]
		#t1 = d[:,0] # FOR Zalucha's T-file
    
    if r1[0] < 0.5 * rSurf: # check to see if r's are zero at the surface
        r1 += rSurf
    
    # Get rid of LAY's AAT troposphere.
    #AboveTrop = r1 > 1181
    #r1 = r1[AboveTrop]
    #t1 = t1[AboveTrop]
    
        
    # Now replace with a Zalucha profile
    #d2 = loadtxt('/Volumes/Manoa/Felix/Work13/PlutoHaze13/Zalucha_TvsR.txt', skiprows=1)
    #t1 = d2[:,0]
    #r1 = d2[:,1]
    #print "Using Zalucha.txt"
    
    rSurf = r1[0]
    bLyr = 15.0 # Height of the transition layer in km
    rObl = rSurf + bLyr
    
    rTop = 6000
    Nr = 1000
    b, rx, tx = XtendT(r1, t1, rTop, Nr)
    
    # gSurf = 65.8 * (rSurf/1186)**2
    gSurf = 61.9 * (rSurf/1186)**2
    PSurf = T2Pv_N2b(tx[0])
    lam = 0.5
    Px, nu = T2nu(rx, tx, gSurf, rSurf, PSurf, lam)
    
    rMax = 3000.
    dr = 1.0
    sMax = 3000.
    ds = 1.0
    
    r_imp, a0 = nu2alpha(rx, rSurf, nu, rMax, dr, sMax, ds)
    aFun = interp1d(r_imp, a0, kind='linear', bounds_error=False, fill_value=0.0)
    
    OccFOV = 6000. # FOV for input grid of rays
    #npts = 4096
    npts = 8192
    
    theta1 = linspace(-2*pi, 2*pi, 2000)
    rVal = 1.0 + 0.5 * oblateness * (1.0 - cos(2*theta1))
    tFun = r_scale(theta1, rVal)
        
    #aScr = AlphaScreen(OccFOV, npts, aFun, tFun, rSurf, rObl)
    aScr = AlphaScreen2(OccFOV, npts, aFun, tFun, rSurf, transBot, transTop, NP_Ang)
    
    # Now build the opacity screen
    ap = DiskScreen(npts, OccFOV, rSurf, tFun)
    
    
    # Do a virtual CCD with the screens
    
    OutFOV = 6000.
    Z = 30. # distance in AU
    nOut = 1024 # for vCCD only
    #vc = raytr(aScr, ap, Z, OccFOV, OutFOV, nOut, rSurf)
    #vc3 = raytr3(aScr, ap, Z, OccFOV, OutFOV, nOut, rSurf)
    
    
    # Now compute a Fourier optics version. We'll cheat by using a 
    # wavelength that's extra long such that OutFOV/npts (resolution = 0.73 km per grid):
    # Res = sqrt(lam*Z/npts), so lam = Res**2 * (npts/Z)
    
    res = OutFOV/npts
    Z_km = Z * 149.6e6
    lam_km = res**2 * npts / Z_km
        


    print "lam (mm) = ", lam_km * 1.0e6
    
    k = 2*pi/lam_km # wavenumber, turns alpha field into phase delays
    phases = aScr * k
    
    ap1 = cos(phases) + 1j * sin(phases)
    
    pa = occ_lc(ap1*ap)
    
    pa2 = rebin(pa, 1024, 1024)
    
    
    # now extract a lightcurve along the ground station path
    
    
    doc_f1, doc_g1, tsec1 = gndpos(doc_pn, f0, g0) # doc_f, doc_g in km; need to calc output screen in km.
    doc_f_fun = interp1d(tsec1, doc_f1, kind='linear', bounds_error=False, fill_value=0.0)
    doc_g_fun = interp1d(tsec1, doc_g1, kind='linear', bounds_error=False, fill_value=0.0)
    
    zoomFac = 10
    tsec = zoom(tsec1, zoomFac, order=1, mode='nearest')
    doc_f = doc_f_fun(tsec)
    doc_g = doc_g_fun(tsec)
        
    pixf, pixf_frac = divmod(doc_f/res + 0.5*npts, 1) # divmod(x,1) returns the integer quotient and the frac
    pixg, pixg_frac = divmod(doc_g/res + 0.5*npts, 1)
    
    inFOV = (pixf >= 0) & (pixf <= (npts-2)) & (pixg >= 0) & (pixg <= (npts-2))
    
    tFOV = tsec[inFOV] # These are the timesteps for points that fall within OutFOV.
    
    # We'll use the fractional contributions from the four closest pixels to doc_f and doc_g
    fw0 = 1.0 - pixf_frac[inFOV]
    fw1 = pixf_frac[inFOV]
    gw0 = 1.0 - pixg_frac[inFOV]
    gw1 = pixg_frac[inFOV]
    
    pixf0 = (pixf[inFOV]).astype(int)
    pixg0 = (pixg[inFOV]).astype(int)
    pixf1 = (pixf[inFOV]).astype(int) + 1
    pixg1 = (pixg[inFOV]).astype(int) + 1
    
    lc = fw0*gw0*pa[pixg0, pixf0] + fw1*gw0*pa[pixg0, pixf1] + fw0*gw1*pa[pixg1, pixf0] + fw1*gw1*pa[pixg1, pixf1]  
        
    endT = tm.time()
    elapT = endT-startT
    print "Elapsed time: ", elapT
    
    return(pa, pixf0, pixg0, lc, tFOV, r1, t1, aScr, ap)
    



#-----------------------------------------------------

def DoF1(r0, p, m, r):
    '''
    r0	- the radii at which we supply Temperatures (p's) and dT/dr (m's)
    p	- the Temperatures at the r0 points
    m	- the slopes at the r0 points
    r	- the radii at which we return the interpolated T's
    
    pa, pixf0, pixg0, lc, tFOV, rd, Tp, aScr, ap = DoF1(r0, p, m, r)
    '''

    startT = tm.time()
    
    # Some parameters describing the oblateness and geometric shadow center.
    dg0 = 0.0
    oblateness = 0.05
    transBot = 1.0 # The bottom and top of the transition region from circular to 
    transTop = 5.0 # stretched (oblate) radii.
    
    
    platform = 3 # 1 = pico, 2 = ajax, 3 = flume, 4 = lloyd
    
    # Load an occultation lightcurve (contains fnom and gnom) and 
    # load an atmospheric profile

    if (platform == 1):
        print "Finding files on Pico"
        doc_pn = '/Volumes/Manoa/Felix/Work12/Pluto_Haze/lc01_MTJ_Doc_3.fits'
        d = loadtxt('/Volumes/Manoa/Felix/Work13/PlutoHaze13/LAY_AtmProfile/Pluto/2009_oc/table_heat_7_aat.txt', skiprows = 37)
    elif (platform == 2):
        print "Finding files on ajax"
        doc_pn = '/Volumes/Firescrew/Work14/PlutoHaze14/lc01_MTJ_Doc_3.fits'
        d = loadtxt('/Volumes/Firescrew/Work14/PlutoHaze14/table_heat_7_aat.txt', skiprows = 37)
    elif (platform == 3): 
        print "Finding files on flume"
        doc_pn = '/Volumes/Felix/Work14/PlutoHaze14/lc01_MTJ_Doc_3.fits'
        d = loadtxt('/Volumes/Felix/Work14/PlutoHaze14/table_heat_7_aat.txt', skiprows = 37)
    elif (platform == 4): 
        print "Finding files on lloyd"
        doc_pn = '/d1/efy/work14/PLHaze/20090901_frOlkin/lc01_MTJ_Doc_3.fits'
        d = loadtxt('/d1/efy/work14/PLHaze/Pluto/2009_oc/table_heat_7_aat.txt', skiprows = 37)
        
    NP_Ang = 247.59
    NP_Ang = 0.0
    
    f0 = 499.86621 # From LAY's "plot_w_lat.pro"
    g0 = -844.97886 + dg0 # Same source

    rSurf = 1178.645
    
    t1,r1 = makeTr(r0, p, m, r)

    # Check to see if the user supplied a value for dTdr (defaults to None). If so,
    # call t1, r1 = newDtdr(d, dTdr)
    
    
    if r1[0] < 0.5 * rSurf: # check to see if r's are zero at the surface
        r1 += rSurf
    
    # Get rid of LAY's AAT troposphere.
    #AboveTrop = r1 > 1181
    #r1 = r1[AboveTrop]
    #t1 = t1[AboveTrop]
    
        
    
    rSurf = r1[0]
    bLyr = 15.0 # Height of the transition layer in km
    rObl = rSurf + bLyr
    
    rTop = 6000
    Nr = 1000
    b, rx, tx = XtendT(r1, t1, rTop, Nr)
    
    # gSurf = 65.8 * (rSurf/1186)**2
    gSurf = 61.9 * (rSurf/1186)**2
    PSurf = T2Pv_N2b(tx[0])
    lam = 0.5
    Px, nu = T2nu(rx, tx, gSurf, rSurf, PSurf, lam)
    
    rMax = 3000.
    dr = 1.0
    sMax = 3000.
    ds = 1.0
    
    r_imp, a0 = nu2alpha(rx, rSurf, nu, rMax, dr, sMax, ds)
    aFun = interp1d(r_imp, a0, kind='linear', bounds_error=False, fill_value=0.0)
    
    OccFOV = 6000. # FOV for input grid of rays
    npts = 4096
    #npts = 8192
    
    theta1 = linspace(-2*pi, 2*pi, 2000)
    rVal = 1.0 + 0.5 * oblateness * (1.0 - cos(2*theta1))
    tFun = r_scale(theta1, rVal)
        
    #aScr = AlphaScreen(OccFOV, npts, aFun, tFun, rSurf, rObl)
    aScr = AlphaScreen2(OccFOV, npts, aFun, tFun, rSurf, transBot, transTop, NP_Ang)
    
    # Now build the opacity screen
    ap = DiskScreen(npts, OccFOV, rSurf, tFun)
    
    
    # Do a virtual CCD with the screens
    
    OutFOV = 6000.
    Z = 30. # distance in AU
    nOut = 1024 # for vCCD only
    #vc = raytr(aScr, ap, Z, OccFOV, OutFOV, nOut, rSurf)
    #vc3 = raytr3(aScr, ap, Z, OccFOV, OutFOV, nOut, rSurf)
    
    
    # Now compute a Fourier optics version. We'll cheat by using a 
    # wavelength that's extra long such that OutFOV/npts (resolution = 0.73 km per grid):
    # Res = sqrt(lam*Z/npts), so lam = Res**2 * (npts/Z)
    
    res = OutFOV/npts
    Z_km = Z * 149.6e6
    lam_km = res**2 * npts / Z_km
        


    print "lam (mm) = ", lam_km * 1.0e6
    
    k = 2*pi/lam_km # wavenumber, turns alpha field into phase delays
    phases = aScr * k
    
    ap1 = cos(phases) + 1j * sin(phases)
    
    pa = occ_lc(ap1*ap)
    
    pa2 = rebin(pa, 1024, 1024)
    
    
    # now extract a lightcurve along the ground station path
    
    
    doc_f1, doc_g1, tsec1 = gndpos(doc_pn, f0, g0) # doc_f, doc_g in km; need to calc output screen in km.
    doc_f_fun = interp1d(tsec1, doc_f1, kind='linear', bounds_error=False, fill_value=0.0)
    doc_g_fun = interp1d(tsec1, doc_g1, kind='linear', bounds_error=False, fill_value=0.0)
    
    zoomFac = 10
    tsec = zoom(tsec1, zoomFac, order=1, mode='nearest')
    doc_f = doc_f_fun(tsec)
    doc_g = doc_g_fun(tsec)
        
    pixf, pixf_frac = divmod(doc_f/res + 0.5*npts, 1) # divmod(x,1) returns the integer quotient and the frac
    pixg, pixg_frac = divmod(doc_g/res + 0.5*npts, 1)
    
    inFOV = (pixf >= 0) & (pixf <= (npts-2)) & (pixg >= 0) & (pixg <= (npts-2))
    
    tFOV = tsec[inFOV] # These are the timesteps for points that fall within OutFOV.
    
    # We'll use the fractional contributions from the four closest pixels to doc_f and doc_g
    fw0 = 1.0 - pixf_frac[inFOV]
    fw1 = pixf_frac[inFOV]
    gw0 = 1.0 - pixg_frac[inFOV]
    gw1 = pixg_frac[inFOV]
    
    pixf0 = (pixf[inFOV]).astype(int)
    pixg0 = (pixg[inFOV]).astype(int)
    pixf1 = (pixf[inFOV]).astype(int) + 1
    pixg1 = (pixg[inFOV]).astype(int) + 1
    
    lc = fw0*gw0*pa[pixg0, pixf0] + fw1*gw0*pa[pixg0, pixf1] + fw0*gw1*pa[pixg1, pixf0] + fw1*gw1*pa[pixg1, pixf1]  
        
    endT = tm.time()
    elapT = endT-startT
    print "Elapsed time: ", elapT
    
    return(pa, pixf0, pixg0, lc, tFOV, r1, t1, aScr, ap)
    



#-----------------------------------------------------

def DoF2(r,T, oblateness, transBot, transTop):
    '''
    oblateness	- the fraction by which the equatorial radius exceeds the average and the polar radius is less.
    dg0			- change in the g0 parameter (km). New g0 = g0 (from LAY's plot_w_lat.pro) + dg0
    dTdr		- An optional argument. If present dTdr follows this profile above T_max.
    
    pa, pixf0, pixg0, lc, tFOV, rd, Tp, aScr, ap = DoF2(r, T, oblateness, transBot, transTop)
    '''

    startT = tm.time()
    
    # Some parameters describing the oblateness and geometric shadow center.
    dg0 = 0.0
    # oblateness = 0.07
    #transBot = 1.0 # The bottom and top of the transition region from circular to 
    #transTop = 2.0 # stretched (oblate) radii.
    
    
    platform = 3 # 1 = pico, 2 = ajax, 3 = flume, 4 = lloyd
    
    # Load an occultation lightcurve (contains fnom and gnom) and 
    # load an atmospheric profile

    if (platform == 1):
        print "Finding files on Pico"
        doc_pn = '/Volumes/Manoa/Felix/Work12/Pluto_Haze/lc01_MTJ_Doc_3.fits'
    elif (platform == 2):
        print "Finding files on ajax"
        doc_pn = '/Volumes/Firescrew/Work14/PlutoHaze14/lc01_MTJ_Doc_3.fits'
    elif (platform == 3): 
        print "Finding files on flume"
        doc_pn = '/Volumes/Felix/Work14/PlutoHaze14/lc01_MTJ_Doc_3.fits'
    elif (platform == 4): 
        print "Finding files on lloyd"
        doc_pn = '/d1/efy/work14/PLHaze/20090901_frOlkin/lc01_MTJ_Doc_3.fits'
        
    #NP_Ang = 247.59
    NP_Ang = 0.0
    
    f0 = 499.86621 # From LAY's "plot_w_lat.pro"
    g0 = -844.97886 + dg0 # Same source

    rSurf = 1178.645

    # Check to see if the user supplied a value for dTdr (defaults to None). If so,
    # call t1, r1 = newDtdr(d, dTdr)
    
    r1 = r
    t1 = T
    
    if r1[0] < 0.5 * rSurf: # check to see if r's are zero at the surface
        r1 += rSurf
    
    # Get rid of LAY's AAT troposphere.
    #AboveTrop = r1 > 1181
    #r1 = r1[AboveTrop]
    #t1 = t1[AboveTrop]
    
        
    # Now replace with a Zalucha profile
    #d2 = loadtxt('/Volumes/Manoa/Felix/Work13/PlutoHaze13/Zalucha_TvsR.txt', skiprows=1)
    #t1 = d2[:,0]
    #r1 = d2[:,1]
    #print "Using Zalucha.txt"
    
    rSurf = r1[0]
    bLyr = 15.0 # Height of the transition layer in km
    rObl = rSurf + bLyr
    
    rTop = 6000
    Nr = 1000
    b, rx, tx = XtendT(r1, t1, rTop, Nr)
    
    # gSurf = 65.8 * (rSurf/1186)**2
    gSurf = 61.9 * (rSurf/1186)**2
    PSurf = T2Pv_N2b(tx[0])
    lam = 0.5
    Px, nu = T2nu(rx, tx, gSurf, rSurf, PSurf, lam)
    
    rMax = 3000.
    dr = 1.0
    sMax = 3000.
    ds = 1.0
    
    r_imp, a0 = nu2alpha(rx, rSurf, nu, rMax, dr, sMax, ds)
    aFun = interp1d(r_imp, a0, kind='linear', bounds_error=False, fill_value=0.0)
    
    OccFOV = 6000. # FOV for input grid of rays
    npts = 4096
    #npts = 8192
    
    theta1 = linspace(-2*pi, 2*pi, 2000)
    rVal = 1.0 + 0.5 * oblateness * (1.0 - cos(2*theta1))
    tFun = r_scale(theta1, rVal)
        
    #aScr = AlphaScreen(OccFOV, npts, aFun, tFun, rSurf, rObl)
    aScr = AlphaScreen2(OccFOV, npts, aFun, tFun, rSurf, transBot, transTop, NP_Ang)
    
    # Now build the opacity screen
    ap = DiskScreen(npts, OccFOV, rSurf, tFun)
    
    
    # Do a virtual CCD with the screens
    
    OutFOV = 6000.
    Z = 30. # distance in AU
    nOut = 1024 # for vCCD only
    #vc = raytr(aScr, ap, Z, OccFOV, OutFOV, nOut, rSurf)
    #vc3 = raytr3(aScr, ap, Z, OccFOV, OutFOV, nOut, rSurf)
    
    
    # Now compute a Fourier optics version. We'll cheat by using a 
    # wavelength that's extra long such that OutFOV/npts (resolution = 0.73 km per grid):
    # Res = sqrt(lam*Z/npts), so lam = Res**2 * (npts/Z)
    
    res = OutFOV/npts
    Z_km = Z * 149.6e6
    lam_km = res**2 * npts / Z_km
        


    print "lam (mm) = ", lam_km * 1.0e6
    
    k = 2*pi/lam_km # wavenumber, turns alpha field into phase delays
    phases = aScr * k
    
    ap1 = cos(phases) + 1j * sin(phases)
    
    pa = occ_lc(ap1*ap)
    
    pa2 = rebin(pa, 1024, 1024)
    
    
    # now extract a lightcurve along the ground station path
    
    
    doc_f1, doc_g1, tsec1 = gndpos(doc_pn, f0, g0) # doc_f, doc_g in km; need to calc output screen in km.
    doc_f_fun = interp1d(tsec1, doc_f1, kind='linear', bounds_error=False, fill_value=0.0)
    doc_g_fun = interp1d(tsec1, doc_g1, kind='linear', bounds_error=False, fill_value=0.0)
    
    zoomFac = 10
    tsec = zoom(tsec1, zoomFac, order=1, mode='nearest')
    doc_f = doc_f_fun(tsec)
    doc_g = doc_g_fun(tsec)
        
    pixf, pixf_frac = divmod(doc_f/res + 0.5*npts, 1) # divmod(x,1) returns the integer quotient and the frac
    pixg, pixg_frac = divmod(doc_g/res + 0.5*npts, 1)
    
    inFOV = (pixf >= 0) & (pixf <= (npts-2)) & (pixg >= 0) & (pixg <= (npts-2))
    
    tFOV = tsec[inFOV] # These are the timesteps for points that fall within OutFOV.
    
    # We'll use the fractional contributions from the four closest pixels to doc_f and doc_g
    fw0 = 1.0 - pixf_frac[inFOV]
    fw1 = pixf_frac[inFOV]
    gw0 = 1.0 - pixg_frac[inFOV]
    gw1 = pixg_frac[inFOV]
    
    pixf0 = (pixf[inFOV]).astype(int)
    pixg0 = (pixg[inFOV]).astype(int)
    pixf1 = (pixf[inFOV]).astype(int) + 1
    pixg1 = (pixg[inFOV]).astype(int) + 1
    
    lc = fw0*gw0*pa[pixg0, pixf0] + fw1*gw0*pa[pixg0, pixf1] + fw0*gw1*pa[pixg1, pixf0] + fw1*gw1*pa[pixg1, pixf1]  
        
    endT = tm.time()
    elapT = endT-startT
    print "Elapsed time: ", elapT
    
    return(pa, pixf0, pixg0, lc, tFOV, r1, t1, aScr, ap)
    


#-----------------------------------------------------

def newDtdr(d, dTdr, d_sm):
    '''
    This routine lets the user change the slope of the thermal profile above the thermal inversion.
    It finds the altitude where T_max occurs, and replaces all points above T_max with 
    a profile whose slope is dTdr.
    
    NOTE: currently this routine changes the contents of d[].
    
    d		- An array of radii and temperatures.
    dTdr	- the user-requested slope above T_max
    d_sm	- altitude over which the T profile changes
    
    t1, r1 = newDtdr(d, dTdr)
    
    24-APR-2014, EFY, SWRI
    '''
        
    # STEP 1: find the maximum temperature
    t1 = d[:,1]
    r1 = d[:,0]
    
    T_max = t1.max() # value of the maximum T
    iT_max = t1.argmax() # index of the maximum T
    
    # STEP 2: get differential altitudes with respect to the altitude of T_max
    dr = r1[iT_max:] - r1[iT_max] # Only the radii above iT_max, relative to the radius of T_max
    
    # STEP 3: Calculate weights for a gradual change of the T profile
    wt = dr * 0.0
    sm_range = (dr <= d_sm)
    wt[sm_range] = (d_sm - dr[sm_range])/d_sm
    
    t1[iT_max:] = (1.0 - wt) * (dr * dTdr + T_max) + wt * t1[iT_max]
    
    return(t1, r1)
    
#-----------------------------------------------------
    
def DoP(outPath, TFile):
    '''
    This routine calls DoFO(obl, dg0) repeatedly with varying values of oblateness and offsets
    to g0. The resulting observer's plane intensity field, the oversampled f,g positions and
    timesteps and the oversampled lightcurves are stored in a subdirectory.
    18-APR-2014 EFY, SWRI
    
    inputs
    outPath	- the directory in which subdirectories will be created and results stored. outPath needs
    a trailing /.
    TFile	- path to a temp vs. r file (skip37 rows)
    '''
    
    # Loop over the follwoing oblateness values
    oblVals = arange(5) * 0.02 + 0.01
    
    # and the following offsets for g0
    dg0Vals = (arange(5) - 2.0) * 2.0
    
    for i in range(5):
        for j in range(5):
            dirName = outPath + 'obl'+str(i) + '_dg0'+str(j)
            os.mkdir(dirName)
            print "Making dir: ", dirName, " for obl, dg0 = ", oblVals[i], dg0Vals[j]
            pa, pixf0, pixg0, lc, tFOV = DoFO(TFile, oblVals[i], dg0Vals[j])
            save(dirName+'/pa', pa)
            save(dirName+'/pixf0', pixf0)
            save(dirName+'/pixg0', pixg0)
            save(dirName+'/lc', lc)
            save(dirName+'/tFOV', tFOV)
            
            
            
    

#-----------------------------------------------------
    
def DoTProf(outPath, Tpath):
    '''
    This routine looks up several T(r) profiles and calls DoFO() repeatedly
    to calculate obs-plane-int-fields.
    
    outPath	- location to write files, needs to end
    Tpath	- location to find LAY's thermal profile files - needs to end in /
    
    22-APR-2014, EFY, SwRI
    '''
    oblateness = 0.05
    dg0 = 0.0
    
    TFile = array(['003_005_10', '003_005_70', '003_050_10', '003_050_70', '300_005_10', '300_005_70', '300_050_10', '300_050_70'])
    
    os.chdir(outPath)
    for i in range(8):
        inFile = Tpath + TFile[i] + '.txt'   
        print "Using profile ", TFile[i]
        os.mkdir(TFile[i])
        pa, pixf0, pixg0, lc, tFOV = DoFO(inFile, oblateness, dg0)
        save(TFile[i]+'/pa', pa)
        save(TFile[i]+'/pixf0', pixf0)
        save(TFile[i]+'/pixg0', pixg0)
        save(TFile[i]+'/lc', lc)
        save(TFile[i]+'/tFOV', tFOV)

        
    
#-----------------------------------------------------    

def makeTr(r0, p, m, r):
    '''
    Generates a T(r) profiles from user-supplied T's and T_slopes at selected
    radii values. Uses csplines to interpolate between points and slopes.
    FIXED SLOPES SCALING to unit interval in version 30.
    
    r0	- the radii at which we supply Temperatures (p's) and dT/dr (m's)
    p	- the Temperatures at the r0 points
    m	- the slopes at the r0 points
    r	- the radii at which we return the interpolated T's
    
    T,R = makeTr(r0, p, m, r)
    '''
    
    T = zeros(0) * 1.0
    R = zeros(0) * 1.0
    
    # STEP 1: Loop over consecutive pairs or control points
    npair = r0.size - 1
    
    for ipair in range(npair):
        i0 = ipair # index of the beginning point
        i1 = i0+1  # index of the ending point
        
        # Map radii between the two points to a unit interval
        t = lambda x: (x - r0[i0])/(r0[i1] - r0[i0]) 
        
        deltaX = p[i1] - p[i0] # length of interval is used to scale the slopes
        
        # Here are the coefficients for the cspline for points in the interval
        coefs = zeros(4) * 1.0
        coefs[0] = 2*p[i0] + deltaX*m[i0] - 2*p[i1] + deltaX*m[i1]
        coefs[1] = -3*p[i0] - 2*deltaX*m[i0] + 3*p[i1] - deltaX*m[i1]
        coefs[2] = deltaX*m[i0]
        coefs[3] = p[i0]
        
        # Make a boolean list of points in the interval
        r_gd = (r >= r0[i0]) & (r <= r0[i1]) 
        r1 = t(r[r_gd]) # and transform them onto the unit interval
        
        T0 = polyval(coefs, r1)
        
        # STEP 2: Accumulate the interpolated T's into a single array.
        T = append(T, T0)
        R = append(R, r[r_gd])
        
    return(T,R)

#-----------------------------------------------------
    
