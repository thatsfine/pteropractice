# conversions to decimal form
def hms2dday(d, h, m, s):
    ''' [dday] = hms2dday(d, h, m, s)
    dday is decimal day given day d, hours h, minutes m, and decimal seconds s
    KH, 6/8/15
    '''
    sec = s/(60.*60.*24.)
    min = m/(60*24)
    hour = h/(24.)
    dday = d + hour + min + sec
    return(dday)
    
def ra2deg(rAtuple)
    ''' [RAdeg] = ra2deg(rAtuple)
    Returns decimal degree form of right ascension RAdeg given rAtuple=(hour, minute,
    second)
    KH, 6/8/15
    '''
    h, m, s = rAtuple
    RAdeg = h*(360./24) + m*(360./(24.*60.)) + s*(360./(24.*60.*60.))
    return (RAdeg)    
    
def dec2deg(dectuple)
    ''' [DECdeg] = dec2deg(dectuple)
    Returns decimal degree form of declination DECdeg given dectuple=(degrees, minutes,
    seconds).
    KH, 6/8/15
    '''
    d, m, s = dectuple
    if d < 0:
        DECdeg = d - (m/60.) - (s/(60.*60.))
    else:
        DECdeg = d + (m/60.) + (s/(60.*60.))
    return(DECdeg)
    
# Julian days and MJD

def julday(y, m, d, h, mi, s):
    ''' [JDay] = julday(y, m, d, h, mi, s)
    JDay is decimal Julian day number given year y, month m, day d, hour h, minutes mi,
    and decimal seconds s
    KH, 6/8/15
    '''
    dd = hms2dday(d, h, mi, s)
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
    JDay = B + C + D + dd + 1720994.5
    return(JDay)
    
def mJulDay(y, m, d, h, mi, s):
    ''' [mJDay] = mJulDay(y, m, d, h, mi, s)
    mJDay is modified Julian day number given year y, month m, day d, hour h, minutes mi,
    and decimal seconds s
    KH, 6/8/15
    '''
    dd = hms2dday(d, h, mi, s)
    mJDay = Julday(y, m, dd) - 2400000.5
    return(mJDay)
        
# gmt to gst

# gmt to lst (calls prev)

# h:m:s to hr (dec)