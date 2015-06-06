import numpy as np
import astropy.io.fits as pf

#-----------------------------------------------------

def load_spe(fname):
    '''Loads the data from a *.spe file (created by WinView) into a numpy array.
    For reference to the SPE header format, see convert_princeton_header.pro.
    EFY, SwRI, 3-JUN-2012'''
    
    hdsize = 4100 # size of the header in bytes. We'll usually read it as an uint16 array.
    pixTypes = {0:'float32', 1:'int32', 2:'int16', 3:'uint16'}
    
    f0 = open(fname, mode='rb') 
    hd = np.fromfile(f0, 'int16', hdsize/2) # read the header as 2-byte integers
    
    xdim = hd[42/2] * 1L # the x-dimension is at integer 21 in the header.
    ydim = hd[656/2] * 1L # the y-dimension is at integer 328 in the header.
    dataType = hd[108/2]
    nFrm = hd[1446/2] * 1L
    
    nPix = xdim*ydim*nFrm
    pixT = pixTypes[dataType]
    
    print "xs, ys, numFrames, pixType = ", xdim, ydim, nFrm, pixT
    
    d = np.fromfile(f0, pixT, nPix) # load the data following the header
    d.shape = (nFrm, ydim, xdim) # reshape the data stream as a 3-D array
    f0.close
    return(d)
    
    

#-----------------------------------------------------

def spe2fits(inFile, outFile):
    '''Saves a WinView *.spe file (called *inFile*) as a fits file (outFile).
    Calls load_spe(inFile) to load the array, thne saves a FITS file with a 
    bare-bones header.
    EFY, SwRI, 3-JUN-2012'''
    
    d = load_spe(inFile)
    hdu = pf.PrimaryHDU(d)
    hdu.writeto(outFile)
    
    return

#-----------------------------------------------------

