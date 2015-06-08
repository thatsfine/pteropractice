# Julian days and MJD

# find number of days between 1/1 and today
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
def getB(y,m,d,h,mn,s):
	jd=julday(y, m, d, h, mn, s):
	t=(jd-2415020)/36525
	r=6.6460656+(2400.051262*t)+(0.00002581*(t**2))
	u=r-(24*(y-1900))
	return 24-u

# from gmt to gst
def toGST (y,m,d,h,mn,s):
	a=0.0657098
	b=getB(y,m,d,h,mn,s)
	c=1.002738
	d=0.997270
	fromJan=numDaysFromJan(y,m,d)
	tZero=days*a-b
	gmtDec=h+m/60+s/3600
	ans=tZero+gmtDec*c
	if(ans>24):
		ans-=24
	elif(ret<0):
		ans+=24
	return (ans//1,((ans%1)*60)//1,((ans//1)*60)%1)*60)

# gmt to lst (calls prev) RA + hour angle
def toLST (y,m,d,h,mn,s,dir,deg):
	(gsH, gsM, gsS) = toGST(y,m,d,h,mn,s)
	decGST=gsH+gsM/60+gsS/60
	degHrs=deg/15
	if(dir=="W"):
		decGST-=degHrs
	elif(dir=="E"):
		decGST-degHrs
	if(decGST>24):
		decGST-=24
	elif(decGST<0):
		decGST+=24
	return (decGST//1,((decGST%1)*60)//1,((decGST//1)*60)%1)*60)

# h:m:s to hr (dec)

# eq to horiz coord conversion
def eq2hor(gmt, rA, lat, dec):
    # [alt, az] = eq2hor(gmt, rA, lat, dec)
    #Converts to alt, az coordinates given Greenwich Mean Time gmt, right ascension tuple
    #rA (hours, minutes, seconds), latitude lat, and declination in decimal degrees dec.
    #CYC 6/08/15
    
   hourAng = hourAng(gmt, rA) #calculates hour angle
   deg_hourAng=15*hourAng #converts hour angle to degrees
   sinAlt=sin(dec)sin(lat) + cos(dec)cos(lat)cos(deg_hourAng)
   alt = np.arcsin(sinAlt) #outputs altitude
   cosAz = (sin(dec)-sin(lat)sin(alt))/(cos(lat)cos(alt))
   az = np.arccos(cosAz) #outputs azimuth
   return (alt, az) #returns altitude, azimuth as a tuple
