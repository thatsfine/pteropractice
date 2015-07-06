import numpy as np

# Degrees to decimal radian function
def degToRad(deg):
    return (deg*((np.pi)/180.))

#Radian to decimal degrees function
#pre - input is in radians
#post - degree representation
def radToDeg(rad):
    return (rad*(180./(np.pi)))

# Decimal hours to h::m::s
def decHrToHMS(hrs):
    return (hrs//1, ((hrs%1)*60)//1, ((((hrs%1)*60.)%1)*60))

# h:m:s to decimal hr
def hmsToDecHr(h,m,s):
    return (h+(m/60.)+(s/3600.))


# conversions to decimal form
def hmsToDday(d, h, m, s):
    ''' [dday] = hms2dday(d, h, m, s)
    dday is decimal day given day d, hours h, minutes m, and decimal seconds s
    KH, 6/8/15
    '''
    return d+(s/(60.*60.*24.))+(m/(60.*24.))+(h/24.)  

def raToDeg(h,m,s):
    ''' [RAdeg] = ra2deg(rAtuple)
    Returns decimal degree form of right ascension RAdeg given rAtuple=(hour, minute,
    second)
    KH, 6/8/15
    '''
    return h*(360./24) + m*(360./(24.*60.)) + s*(360./(24.*60.*60.))
    
    
def decToDeg(d,m,s):
    ''' [DECdeg] = dec2deg(dectuple)
    Returns decimal degree form of declination DECdeg given dectuple=(degrees, minutes,
    seconds).
    KH, 6/8/15
    
    Changed to use np.sign and np.abs
    EFY, 6/10/2015
    '''
    DECdeg = np.sign(d) * (np.abs(d) + m/60. + s/3600.)
    #if d < 0:
    #    DECdeg = d - (m/60.) - (s/(60.*60.))
    #else:
    #    DECdeg = d + (m/60.) + (s/(60.*60.))
    return(DECdeg)
    
# Julian days and MJD
def julDay(y, m, d, h, mn, s):
    ''' [JDay] = julday(y, m, d, h, mi, s)
    JDay is decimal Julian day number given year y, month m, day d, hour h, minutes mi,
    and decimal seconds s
    KH, 6/8/15
    '''
    dd = hmsToDday(d, h, mn, s)
    if (m == 1) | (m == 2):
        y = y - 1
        m = m + 12
    if (y > 1582) | ((y == 1582) & (m >= 10) & (dd >= 15)):
        A = int(y/100.)
        B = 2. - A + int(A/4.)
    else:
        B = 0.
    C = int(365.25*y)
    D = int(30.6001*(m + 1))
    jday = B + C + D + dd + 1720994.5
    return(jday)
    
def mJulDay(y, m, d, h, mi, s):
    ''' [mJDay] = mJulDay(y, m, d, h, mi, s)
    mJDay is modified Julian day number given year y, month m, day d, hour h, minutes mi,
    and decimal seconds s
    KH, 6/8/15
    '''
    return julDay(y, m, d, h, mi, s) - 2400000.5

def hourAng(gmt, rA):
    ''' [hrAng] = hourAng(gmt, rA)
    Takes Greenwich Mean Time as gmt and Right Ascension in (hours, minutes, seconds) tuple
    as rA & returns the hour angle.
    ACR 6/8/15
    '''
    (rAh, rAm, rAs) = rA
    decHr = hmsToDecHr(rAh, rAm, rAs)
    degRA = 360 * (decHr/24.)
    hrAng = gmtToLST(gmt) - degToRad(degRA)
    return hrAng


# gmt to gst function is below, first two numDaysFromJan and getB are
# helper functions

# finds number of days between 1/1 and today, prep for gmtToGST
def numDaysFromJan(y,m,d):
    x=0
    m-=1
    while(m>0):
        if (m==2):
            if(y%4==0):
                x+=29
            else:
                x+=28
        elif (m%2==1):
            if (m>7):
                x+=30
            if (m<=7):
                x+=31
        elif (m%2==0):
                if (m>7):
                    x+=31
                if (m<=7):
                    x+=30
        m -= 1
    return x+d

# gmt to gst functions below
# calculate constant B
def getB(y):
    jd=julDay(y, 1, 0, 0, 0, 0)
    t=(jd-2415020)/36525.
    r=6.6460656+(2400.051262*t)+(0.00002581*(t**2))
    u=r-(24.*(y-1900))
    return 24-u

# from gmt to gst
# returns answer in decimal hrs
def gmtToGST (y,m,dy,h,mn,s):
    a=0.0657098
    b=getB(y)
    c=1.002738
    d=0.997270
    fromJan=numDaysFromJan(y,m,dy)
    tZero=fromJan*a-b
    gmtDec=h+mn/60.+s/3600.
    ans=tZero+gmtDec*c
    if(ans>24):
        ans-=24
    elif(ans<0):
        ans+=24
    return ans

# gmt to lst (calls prev) Note: lst = RA + hour angle
# returns answer in decimal hrs
def gmtToLST (y,m,d,h,mn,s,dir,deg):
    decGST= gmtToGST(y,m,d,h,mn,s)
    degHrs=deg/15.
    if(dir=="W"):
        decGST=decGST-degHrs
    elif(dir=="E"):
        decGST=decGST+degHrs
    if(decGST>24):
        decGST-=24
    elif(decGST<0):
        decGST+=24
    return decGST

# eq to horiz coord conversion
def eq2hor(gmt, rA, lat, dec):
    # [alt, az] = eq2hor(gmt, rA, lat, dec)
    #Converts to alt, az coordinates given Greenwich Mean Time gmt, right ascension tuple
    #rA (hours, minutes, seconds), latitude lat, and declination in decimal degrees dec.
    #CYC 6/08/15
    
    hourAng = hourAng(gmt, rA) #calculates hour angle
    deg_hourAng=radToDeg(hourAng) #converts hour angle to degrees
    sinAlt=sin(dec)*sin(lat) + cos(dec)*cos(lat)*cos(deg_hourAng)
    alt = np.arcsin(sinAlt) #outputs altitude
    cosAz = (sin(dec)-sin(lat)*sin(alt))/(cos(lat)*cos(alt))
    az = np.arccos(cosAz) #outputs azimuth
    return (alt, az) #returns altitude, azimuth as a tuple
