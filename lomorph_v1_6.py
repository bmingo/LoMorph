#!/usr/bin/python

# Code for morphological classification of LOFAR extended sources
# Based on Judith Croston's original code, modified and expanded by Beatriz Mingo.
# Please see Mingo et al. 2019 and 2022 for details and credits.

# Version 1.6 for github - includes optimisation for the LOFAR deep fields data

####################
#MAIN CHANGES ######
####################

#Masking of unrelated sources has now been moved from the main code to the flood-filling function.

#The masking order has also changed - first mask the source IN, then mask extraneous sources OUT (before flood-filling). This is more appropriate when the source density is higher (deeper data), to minimise overenthusiastic flood-filling.

#The search radius for unassociated components is now dependent on source size. This makes the code much faster, as most sources are fairly small, and also minimises issues with very large sources.

#The code now also exports the included and excluded components for each source as ds9 regions, so they can be used to check problematic sources.

####################
#TO DO / WISHLIST ##
####################

#Assess if independence from cutouts can be achieved

#Estimate perpendicular sizes

#Increase aperture angle for cases where angle of dist2 and distmax2 is very different?

#Replace astropy with pyephem to speed things up? --> not currently viable, other optimisations implemented

#Tidy up, optimise, improve...

#Translate to Python 3


####################
#LIBRARIES##########
####################


from astropy.io import fits
from astropy.coordinates import SkyCoord, FK5, Angle, Latitude, Longitude
from astropy import wcs
import astropy.units as u
import pyregion
import numpy as np
import numpy.ma as ma
import glob
import os
import sys
import math
import matplotlib.path as Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.measure import label
import time
from shutil import copyfile



####################
#FUNCTIONS##########
####################


# Flood-filling and masking function (uses scikit image label)

def flood_mask(coords,rdel,ddel,ell_ra,ell_dec,ell_maj,ell_min,ell_pa,n_comp,hdu,dataFill,excludeComp):

    #Preliminary tasks

    #Create temporary ds9 region file to use as initial mask
    regwrite=open('temp.reg','w')
    regwrite.write('# Region file format: DS9 version 4.1\n'+'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n'+'fk5\n')

    #Define auxiliary array filled with "1", which we will use to create the masks, and auxiliary data array to fill in the ellipses 
    sizey,sizex=data.shape
    one_mask=1*np.isfinite(np.zeros((sizey,sizex)))


    for n in range(0,n_comp):
        regwrite.write('ellipse('+str(ell_ra[n])+','+str(ell_dec[n])+','+str(ell_maj[n])+'",'+str(ell_min[n])+'",'+str(ell_pa[n]+90.0)+')\n')
    regwrite.close()

    
    #In the auxiliary data array, turn to an arbitrary non-zero value the pixels inside the ellipse
    region=pyregion.open('temp.reg').as_imagecoord(hdu[0].header)
    mask=region.get_mask(hdu=hdu[0])
    dataFill[mask==1]=0.02

    #Masking out nearby sources (previously in main code)
    if excludeComp>0:
        region2=pyregion.open('temp2.reg').as_imagecoord(hdu[0].header)
        maskOut=region2.get_mask(hdu=hdu[0])
        #Making sure that we only mask where the cutout is defined
        exclude_overlap=maskOut+one_mask        
        dataFill[exclude_overlap==2]=np.nan
        
    #Transform the data array into boolean, then binary values
    data_bin=1*(np.isfinite(dataFill))

    #Label the data array: this method will assign a numerical label to every island of "1" in our binary data array; we want 8-style connectivity
    data_label=label(data_bin, connectivity=2)

    #Multiply the label array by the source regions array (post-masking of extraneous sources). This allows us to identify which labels correspond to the clusters of pixels belonging to the source
    include_overlap=data_label*mask

    #Get the list of label values, excluding zero
    multi_labels=np.unique(include_overlap[np.nonzero(include_overlap)])
    #print str(multi_labels)

    #Initialise the cumulative mask
    multi_mask=np.zeros((sizey,sizex))

    #Main loop
    for i in range (0,len(multi_labels)):
        #Because of how masked arrays work, we need to explicitly set to zero the areas of the array we want masked out... we need a temporary masked array for the intermediate step
        temp_mask=(ma.masked_where(data_label!=multi_labels[i],one_mask))
        temp_mask[temp_mask!=1]=0
        #As we have used a 1/0 matrix as a basis, iteratively adding the island masks together will give us the full mask we need 
        multi_mask=(multi_mask+temp_mask)

    #The output mask will only contain "1" in the areas corresponding to the islands of interest
    flooded_mask=multi_mask

    palette=plt.cm.viridis
    palette.set_bad('k',0.0)
    plt.rcParams['figure.figsize']=(10.67,8.0)
    A=np.ma.array(data_label,mask=np.isnan(data_label))
    y,x = np.mgrid[slice((0),(sizey),1),
                   slice((0),(sizex),1)]
    plt.pcolor(x,y,A,cmap=palette,vmin=np.nanmin(A),vmax=np.nanmax(A))
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.axis([x.min(),x.max(),y.min(),y.max()]) 
    plt.colorbar()
        
    plt.savefig('Test2.pdf', dpi=300, format='pdf')
    plt.clf()
    
    return(flooded_mask)


########################

# Maxdist function

def maxdist(a,x,y):

    # Set up xv and yv, make grid, calculate dx and dy, find distances, return maximum distance

    ny,nx = a.shape
    xa = np.linspace(0, nx-1, nx)
    ya = np.linspace(0, ny-1, ny)
    xv, yv = np.meshgrid(xa, ya)
    dx = xv-x
    dy = yv-y
    d2dx = dx * dx
    d2dy = dy * dy
    d2dxdy = d2dx + d2dy
    ddxdy = np.sqrt(d2dxdy)
    dmax = np.max(ddxdy[~np.isnan(a)])
    return dmax


########################

# Length function
# Loop over x and y values if greater than 0
# Print maximum value

def length(a):

    sizey, sizex = a.shape
    ddd = np.zeros_like(a)
    for x in range(sizex):
        for y in range(sizey):
            if not np.isnan(a[y,x]):
                ddd[y,x] = maxdist(a,x,y)

    return np.max(ddd)


########################

# Wedge function
# Define a triangular region based on an angle +/- a given number of radians

def wedge(slope,plusAngle,minusAngle):
    
    plusSlope=0.0
    minusSlope=0.0
    maxWedge=0.0
    minWedge=0.0

    plusSlope=slope+plusAngle
    if plusSlope>np.pi:
        plusSlope=plusSlope-(2*np.pi)
    minusSlope=slope-minusAngle
    if minusSlope<(-np.pi):
        minusSlope=minusSlope+(2*np.pi)

    return(plusSlope,minusSlope)

    
####################
#MAIN CODE##########
####################

start_time=time.time()

#Input catalogue


infile='lockman_final_withids_res_v1.0.csv'
compfile='lockman_comp_v1.0.csv'
sources=open(infile,'r')

#Output text file
g = open('4rms_lockman_morph_out.txt','w')

g.write('#index source rclass core_bright core_dist sumflux LM_size LM_dec_size dynrange dist1 dist2 maxdist1 maxdist2 PA1 PA2 angleDiff failed_mask mask_comp\n')

#Source counter
step=1

# For each source read in fits file - need header to convert source centre to pix
# Use region file to define mask and array to look at
# Use SDSS ID coords as nucleus


#Initialising classification counting varibles that we will need later
fr1=0
fr2=0
fr12=0
fr21=0
small_fr1=0
small_fr2=0
small_hybrid=0
unresolved=0
faint=0


#Loading in the data
sources.readline()
#sources.readline()
for line in sources:
    line = line.strip()
    column = line.split(',')
    source = column[0]
    rx = column[24]
    ry = column[25]
    n_comp = column[29]
    #Getting catalogued source size. Used later to assess distance for exclusion of unassociated components.
    try:
        maj = float(column[10])
    except ValueError:
        maj = 0.0
    try:
        majLGZ = float(column[26])/3600
    except ValueError:
        majLGZ = 0.0
    minSize=0.0045
    srcSize = max(maj,majLGZ,minSize)
    filename='cutouts/'+source+'.fits'
    npyname='newthres4rms/'+source+'.npy'
    if os.path.isfile(filename) and os.path.isfile(npyname):
        hdu=fits.open(filename)
        ra=hdu[0].header['CRVAL1']
        dec=hdu[0].header['CRVAL2']
        xra=hdu[0].header['CRPIX1']
        yra=hdu[0].header['CRPIX2']
        rdel=hdu[0].header['CDELT1']
        ddel=hdu[0].header['CDELT2']
        data=hdu[0].data
        ymax,xmax=data.shape
        xr=np.array(range(xmax))
        yr=np.array(range(ymax))
        xg,yg=np.meshgrid(xr,yr)
        xmin,xmax=xg.min(),xg.max()
        ymin,ymax=yg.min(),yg.max()
        bmaj_pix=abs(6.0/(3600*ddel))
        bmin_pix=abs(6.0/(3600*ddel))
        gfactor=2.0*np.sqrt(2.0*np.log(2.0))
        beamArea=2.0*np.pi*bmaj_pix*bmin_pix/(gfactor*gfactor)
        print '     '
        print str(step)+'    '+str(source)

        #Now we need to load the components for each source

        #Create the appropriate arrays and counter to fill them in
        try: 
            n_comp=int(float(n_comp))
        except ValueError:
            n_comp=1
        #The catalogue reports zero when there is only 1 component - fixing for consistency
        if n_comp==0:
            n_comp=1
        ell_ra=np.zeros(shape=(n_comp,))
        ell_dec=np.zeros(shape=(n_comp,))
        ell_maj=np.zeros(shape=(n_comp,))
        ell_min=np.zeros(shape=(n_comp,))
        ell_pa=np.zeros(shape=(n_comp,))
        ell_flux=np.zeros(shape=(n_comp,))
        compcounter=0

        #To mask out unrelated components
        excludeComp=0
        regwrite2=open('temp2.reg','w')
        regwrite2.write('# Region file format: DS9 version 4.1\n'+'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n'+'fk5\n')


        #Load the components table and skip the first line; get the values out
        components=open(compfile,'r')
        components.readline()
#        components.readline()
        for line2 in components:
            line2=line2.strip()
            column2=line2.split(',')
            compName=column2[0]
            source2=column2[21]
            comp_ra=column2[1]
            comp_dec=column2[2]
            comp_flux=float(column2[5])
            comp_maj=float(column2[10])*3600
            comp_min=float(column2[11])*3600
            comp_pa=float(column2[12])
            if (source2==source):
                if n_comp>=1:
                    ell_ra[compcounter]=comp_ra
                    ell_dec[compcounter]=comp_dec
                    ell_flux[compcounter]=comp_flux
                    ell_maj[compcounter]=comp_maj
                    ell_min[compcounter]=comp_min
                    ell_pa[compcounter]=comp_pa
                    compcounter+=1
                else:
                    ell_ra=comp_ra
                    ell_dec=comp_dec
                    ell_flux=comp_flux
                    ell_maj=comp_maj
                    ell_min=comp_min
                    ell_pa=comp_pa
                    
            #Defining ellipse regions to mask out unrelated components. The search distance to mask out unrelated components is now based on the source size.
            else:
                try:
                    floatOK=float(comp_ra)+float(comp_dec)+float(rx)+float(ry)
                    compSep=1.0
                    if (abs(float(comp_ra)-float(rx))<=(srcSize*4) and abs(float(comp_dec)-float(ry))<=(srcSize*4)):
                        coordsOpt=SkyCoord(ra=float(rx)*u.degree, dec=float(ry)*u.degree, frame='fk5')
                        coordsComp=SkyCoord(ra=float(comp_ra)*u.degree, dec=float(comp_dec)*u.degree, frame='fk5')
                        compSep=coordsOpt.separation(coordsComp)/u.degree
                        if compSep<=(srcSize*2):
                            regwrite2.write('ellipse('+str(comp_ra)+','+str(comp_dec)+','+str(comp_maj)+'",'+str(comp_min)+'",'+str((float(comp_pa)+90.0))+')\n')
                            if excludeComp!=1:
                                excludeComp=1
                except ValueError:
                    print 'Failed to create exclusion region list!'
                    pass

                    
        regwrite2.close()
                
        #Data arrays
        dataFill=np.load(npyname)
        data=np.load(npyname)
        
        #We need to calculate the extent of the data array before we mess up with it!
        sizey,sizex=dataFill.shape

        
        #We need the SDSS coordinates converted to x,y; no small angle approximation, just in case!
    
        coords=SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='fk5')
        coords2=SkyCoord(ra=float(rx)*u.degree, dec=float(ry)*u.degree, frame='fk5')
        dist_x,dist_y=coords.spherical_offsets_to(coords2)
        dx=float((dist_x.to(u.arcsec)/u.arcsec)/(3600.0*rdel))
        dy=float((dist_y.to(u.arcsec)/u.arcsec)/(3600.0*ddel))
        rxpix=xra+int(dx)-(xmin)
        rypix=yra+int(dy)-(ymin)


        #In LoTSS DR1 we filtered out the unassociated components here. This has now been incorporated as part of the flood-filling function.
        
        #Here goes the flood filling call
        #DEBUG - Putting this in to run the code on as many sources as possible, in case some cutouts fail.
        failedMask=0
        try:
            flooded_mask=flood_mask(coords,rdel,ddel,ell_ra,ell_dec,ell_maj,ell_min,ell_pa,n_comp,hdu,dataFill,excludeComp)
            data[flooded_mask==0]=np.nan
        except:
            failedMask=1
            print 'Failed cutout!'
            pass
    
        #We need to make the data array smaller, as it has loads of empty space...
        #Creating delimiters for new array
        xindex_min=0
        xindex_max=0
        yindex_min=0
        yindex_max=0
        index_max=0
        index_min=0
        padding=5
        
        #Finding the first and last non-zero columns/rows
        xsum=np.zeros(sizex)
        ysum=np.zeros(sizey)

        xsum=np.nansum(data, axis=0)
        ysum=np.nansum(data, axis=1)
        
        nonzerox=np.flatnonzero(xsum)
        #We have issues with a few cutouts...
        if len(nonzerox)>0:
            xindex_min=nonzerox[0]
            xindex_max=nonzerox[(len(nonzerox)-1)]
        else:
            xindex_min=0
            xindex_max=sizex
        
        nonzeroy=np.flatnonzero(ysum)
        if len(nonzeroy)>0:
            yindex_min=nonzeroy[0]
            yindex_max=nonzeroy[(len(nonzeroy)-1)]
        else:
            yindex_min=0
            yindex_max=sizey

        
        #Making sure we have some padding for the plots!
        if (xindex_min-padding)<0:
            xindex_min=0
        else:
            xindex_min=xindex_min-padding
        if (xindex_max+padding)>sizex:
            xindex_max=sizex
        else:
            xindex_max=xindex_max+padding
        if (yindex_min-padding)<0:
            yindex_min=0
        else:
            yindex_min=yindex_min-padding
        if (yindex_max+padding)>sizey:
            yindex_max=sizey
        else:
            yindex_max=yindex_max+padding

        #We prefer square plots...
        index_max=max(xindex_max,yindex_max)
        index_min=min(xindex_min,yindex_min)
        
        a=np.zeros((index_max,index_max))
        a=data[index_min:index_max,index_min:index_max]
        sizey,sizex=a.shape

        b=np.zeros((index_max,index_max))
        c=np.zeros((index_max,index_max))
        a2=np.zeros((index_max,index_max))
        b=np.copy(a)
        c=np.copy(a)
        a2=np.copy(a)
        
        #For the morphology code to work we need to calculate the offsets of the optical source to our new (0,0) point!
        rxpix=rxpix-index_min
        rypix=rypix-index_min

        #For brightest cluster of pixels:
        
        fmax=0.0
        distmax=0.0
        distmax1=0.0
        ftot=0.0
        pftot=0.0
        pixtot=0.0
        xsum=0.0
        ysum=0.0
        xvals=0.0
        yvals=0.0
        npix=0.0
        distmax_x=0.0
        distmax_y=0.0
        xpeak=0.0
        ypeak=0.0
        dist1=0.0

        #To calculate second brightest cluster of pixels:

        fmax2=0.0
        distmax2=0.0
        distmax2Prelim=0.0
        distmax12=0.0
        ftot2=0.0
        pftot2=0.0
        pixtot2=0.0
        xsum2=0.0
        ysum2=0.0
        xvals2=0.0
        yvals2=0.0
        npix2=0.0
        distmax2_x=0.0
        distmax2_y=0.0
        distmax2_xPrelim=0.0
        distmax2_yPrelim=0.0
        distmax1=0.0
        distmax1_x=0.0
        distmax1_y=0.0
        xpeak2=0.0
        ypeak2=0.0
        dist2=0.0

        PA1=0.0
        PA2=0.0

        #Global
        secaxisAngle=0.0
        angleDiff=0.0
        rclass=0
        normFlux=0.0
        coreBright=0
        coreSep=0.0
        coreDist=0.0
        siz=0.0
        deconv_siz=0.0

#Finding the first peak
        
        for xval in range(sizex-2):
            for yval in range(sizey-2):
                x=xval+1
                y=yval+1
             
                areaflux=((a[y,x]+a[y,x+1])+(a[y,x]+a[y+1,x])+(a[y,x]+a[y,x-1])+(a[y,x]+a[y-1,x]))/4.0
                pixflux=a[y,x]
                if not np.isnan(areaflux):
                    pixdist=np.sqrt((x-rxpix)**2.0+(y-rypix)**2.0)
                    ftot=ftot+areaflux
                    pixtot+=1
               
                    if areaflux>fmax:
                        fmax=areaflux
                        xpeak=x
                        ypeak=y
                    if pixdist>distmax:
                        distmax=pixdist
                        distmax_x=x
                        distmax_y=y
                if not np.isnan(pixflux):
                    xvals+=x
                    yvals+=y
                    npix+=1
                    xsum+=x*pixflux
                    ysum+=y*pixflux
                    pftot=pftot+pixflux
        
#Some of the below (commented out) for median distances is no longer needed, but might be useful in the future.
        #if filter to avoid bad cutouts!
        if pftot==0:
            pftot=1
        if npix==0:
            npix=1
#        centx=xsum/pftot
#        centy=ysum/pftot
#        medx=xvals/npix
#        medy=yvals/npix
        dist1=np.sqrt((xpeak-rxpix)**2.0+(ypeak-rypix)**2.0)
#        distCen=np.sqrt((xpeak-centx)**2.0+(ypeak-centy)**2.0)
#        roffset=np.sqrt((medx-centx)**2.0+(medy-centy)**2.0) # dist centroid to median
#        roffsetarc=3600.0*ddel*roffset
#        rapeak=rdel*(xpeak+xmin-1-xra)+ra
#        racen=rdel*(centx+xmin-1-xra)+ra
#        decpeak=ddel*(ypeak+ymin-1-yra)+dec
#        deccen=ddel*(centy+ymin-1-yra)+dec
        #if filter to avoid bad cutouts!
        if pixtot>0:
            meanflux=ftot/pixtot
            dynrange=fmax/meanflux
        else:
            dynrange=0.0
        halfdist=distmax/2.0
        siz=ddel*3600.0*length(a) 
        
        #Total flux, corrected for beam size
        normFlux=pftot/beamArea

        #Deconvolved size
        if siz>0.0:
            deconv_siz=np.sqrt(siz*siz-36.0)
        else:
            deconv_siz=0.0
        

        #Filtering for minimum flux cut
        if (pixtot>=5 and normFlux>=0.001):


            #Testing orientation of the brightest pixel to opt id line, defining a triangular exclusion region for later masking.

            axisX=xpeak-rxpix
            axisY=ypeak-rypix
            axisAngle=math.atan2(axisY,axisX)
            maxAngle,minAngle=wedge(axisAngle,np.pi/3.0,np.pi/3.0)

###########################
#Option to pre-classify core-bright FRI
            if (dist1<3.0):
                #Path for core-bright FRIs
                if distmax>=8.0:
                    rclass=1
                    fr1+=1
                    coreBright=1
                    #To preserve original dist1
                    coreSep=dist1

                    #Build a circular mask to exclude the core, so we can look for the secondary peaks
                    
                    #Radius of the circular region we want to exclude; for now, 5 pixels, slightly larger than  beam size, as bright cores can be troublesome for dist2
                    radius=5
                    xcore,ycore=np.ogrid[-ypeak:sizey-ypeak,-xpeak:sizex-xpeak]
                    circMask=xcore*xcore+ycore*ycore<=radius*radius
                    a2[circMask]=np.nan
                    b[circMask]=np.nan
                    c[circMask]=np.nan

                    #Re-initialising all the cumulative variables...
                    fmax=0.0
                    distmax=0.0
                    distmax1=0.0
                    ftot=0.0
                    pftot=0.0
                    pixtot=0.0
                    xsum=0.0
                    ysum=0.0
                    xvals=0.0
                    yvals=0.0
                    npix=0.0
                    distmax_x=0.0
                    distmax_y=0.0
                    xpeak=0.0
                    ypeak=0.0
                    dist1=0.0

                    #Finding the first peak (beyond the core), orientation, etc (same as code above, we overwrite the values)
        
                    for xval in range(sizex-2):
                        for yval in range(sizey-2):
                            x=xval+1
                            y=yval+1
             
                            areaflux=((a2[y,x]+a2[y,x+1])+(a2[y,x]+a2[y+1,x])+(a2[y,x]+a2[y,x-1])+(a2[y,x]+a2[y-1,x]))/4.0
                            pixflux=a2[y,x]
                            if not np.isnan(areaflux):
                                pixdist=np.sqrt((x-rxpix)**2.0+(y-rypix)**2.0)
                                ftot=ftot+areaflux
                                pixtot=pixtot+1.0
               
                                if areaflux>fmax:
                                    fmax=areaflux
                                    xpeak=x
                                    ypeak=y
                                if pixdist>distmax:
                                    distmax=pixdist
                                    distmax_x=x
                                    distmax_y=y
                            if not np.isnan(pixflux):
                                xvals+=x
                                yvals+=y
                                npix+=1
                                xsum+=x*pixflux
                                ysum+=y*pixflux
                                pftot=pftot+pixflux


                    #if filter to avoid bad cutouts!
                        if pftot==0:
                            pftot=1
                        if npix==0:
                            npix=1
                    dist1=np.sqrt((xpeak-rxpix)**2.0+(ypeak-rypix)**2.0)
                    halfdist=distmax/2.0

                    axisX=xpeak-rxpix
                    axisY=ypeak-rypix
                    axisAngle=math.atan2(axisY,axisX)

                    #This time we want to increase the size of the exclusion wedge, as these sources are a little more problematic...
                    maxAngle,minAngle=wedge(axisAngle,(np.pi/2.0),np.pi/2.0)

###################
                #Unresolved sources    
                else:
                    rclass=5
                    unresolved+=1
                dist2=0.0
                distmax2=0.0

#Back to the general flow                

#Now the same method for the 2nd peak applies to everything, so we don't need special cases...

            #Dummy if, to avoid unindenting everything, in case we need to change things again...
            if (1>0):


#Array b: turning values to zero in the cone along the line between the optical core and the 1st peak, plus/minus given angle, for second peak search.
#Array c: zeroing everything outside the wedge, to make sure distmax 1 is in the right place.
                
                for xvalb in range(sizex-2):
                    for yvalb in range(sizey-2):
                        xb=xvalb+1
                        yb=yvalb+1
                        instX=xb-rxpix
                        instY=yb-rypix
                        instAngle=math.atan2(instY,instX)
                        if (minAngle>np.pi/3.0):
                            if (instAngle>minAngle or instAngle<maxAngle):
                                b[yb,xb]=float('nan')

                            else:
                                c[yb,xb]=float('nan')

                        else: 
                            if (instAngle<maxAngle and instAngle>minAngle):
                                b[yb,xb]=float('nan')

                            else:
                                c[yb,xb]=float('nan')


                #Looking for second peak, making sure it's not the core.

                #(changed from 5 to 4 pix for v.1.0 and to 3 pix for v.1.3)
                coreDist=3.0
                
                for xvalc in range(sizex-2):
                    for yvalc in range(sizey-2):
                        xc=xvalc+1
                        yc=yvalc+1

                        areaflux2=((b[yc,xc]+b[yc,xc+1])+(b[yc,xc]+b[yc+1,xc])+(b[yc,xc]+b[yc,xc-1])+(b[yc,xc]+b[yc-1,xc]))/4.0
                        pixflux2=b[yc,xc]
                    
                        areaflux1=((c[yc,xc]+c[yc,xc+1])+(c[yc,xc]+c[yc+1,xc])+(c[yc,xc]+c[yc,xc-1])+(c[yc,xc]+c[yc-1,xc]))/4.0

                        if not np.isnan(areaflux2):
                            pixdist2=np.sqrt((xc-rxpix)**2.0+(yc-rypix)**2.0)
                            pixdist12=np.sqrt((xc-xpeak)**2+(yc-ypeak)**2.0) #To calculate distance between both peaks

                            #Need to take into account resolution issues again, demand that both peaks be at least 2 beam sizes apart.
                            if (areaflux2>fmax2 and pixdist2>coreDist and pixdist12>2*coreDist):
                                fmax2=areaflux2
                                xpeak2=xc
                                ypeak2=yc
                            if pixdist2>distmax2Prelim:
                                distmax2Prelim=pixdist2
                                distmax2_xPrelim=xc
                                distmax2_yPrelim=yc
                        
                        if not np.isnan(areaflux1):
                            pixdist2=np.sqrt((xc-rxpix)**2.0+(yc-rypix)**2.0)

                            if pixdist2>distmax1:
                                distmax1=pixdist2
                                distmax1_x=xc
                                distmax1_y=yc
 
                #To calculate the angle of the second peak and the wedge to ensure that distmax2 is in the right place
                secaxisX=xpeak2-rxpix
                secaxisY=ypeak2-rypix
                secaxisAngle=math.atan2(secaxisY,secaxisX)
                secmaxAngle,secminAngle=wedge(secaxisAngle,np.pi/3.0,np.pi/3.0)

                dist2=np.sqrt((xpeak2-rxpix)**2.0+(ypeak2-rypix)**2.0)
                
            
#Second wedge check, to make sure that distmax2 is in the right place. We can reuse array b, which already has a large area zeroed, for increased efficiency.
                for xvald in range(sizex-2):
                    for yvald in range(sizey-2):
                        xd=xvald+1
                        yd=yvald+1
                        
                        instX2=xd-rxpix
                        instY2=yd-rypix
                        instAngle2=math.atan2(instY2,instX2)
                        
                        areaflux3=((b[yd,xd]+b[yd,xd+1])+(b[yd,xd]+b[yd+1,xd])+(b[yd,xd]+b[yd,xd-1])+(b[yd,xd]+b[yd-1,xd]))/4.0
                        if not np.isnan(areaflux3):
                            pixdist3=np.sqrt((xd-rxpix)**2.0+(yd-rypix)**2.0)
                            
                            if (pixdist3>distmax2):
                                if (secminAngle>np.pi/3.0):
                                    if not (instAngle>secminAngle or instAngle<secmaxAngle):
                                        distmax2=pixdist3
                                        distmax2_x=xd
                                        distmax2_y=yd
                                else:
                                    if not (instAngle<secmaxAngle and instAngle>secminAngle):
                                        distmax2=pixdist3
                                        distmax2_x=xd
                                        distmax2_y=yd

                #Fixing bug which causes distmax2 to be 0 in some cases where it shouldn't:
                if (distmax2==0.0 and distmax2Prelim>0.0):
                    distmax2=distmax2Prelim
                    distmax2_x=distmax2_xPrelim
                    distmax2_y=distmax2_yPrelim
                               
                
                #Making sure that we avoid problems with one-sided and unresolved sources
                if dist2==0.0:
                    distmax2=0.0
                    distmax2_x=0.0
                    distmax2_y=0.0
                if dist2>=length(a)*0.9:
                    dist2=0.0
                    distmax2=0.0
                    distmax2_x=0.0
                    distmax2_y=0.0

                halfdist1=distmax1/2.0
                halfdist2=distmax2/2.0

                #Angle difference between both sides, output position angles

                PA1=(axisAngle*180.0/np.pi)-90.0
                if PA1<-360.0:
                    PA1=PA1+360.0
                if secaxisAngle!=0.0:
                    PA2=(secaxisAngle*180.0/np.pi)-90.0
                    if PA2<-360.0:
                        PA2=PA2+360.0

                angleDiff=max(PA1, PA2) - min(PA1, PA2)
                if angleDiff>180.0:
                    angleDiff=abs(angleDiff-360.0)
                
                #Classification
                #Classes: 0=faint, 1=FRI, 2=FRII, 3=FRI-II hybrid, 4=FRII-I hybrid, 5=unresolved, 6=small FRI, 7=small FRII, 8=small hybrid
                if (rclass!=5) and ((dist1+dist2<20.0 and siz<40.0) or (siz<27.0)):
                    if rclass==1:
                        fr1-=1
                        small_fr1+=1
                        rclass=6
                    else: 
                        if distmax1>=8.0:
                            if dist1>=halfdist1:

                                if dist2>halfdist2:
                                    rclass=7
                                    small_fr2+=1
                                else:
                                    rclass=8
                                    small_hybrid+=1

                            else:

                                if dist2<=halfdist2:
                                    rclass=6
                                    small_fr1+=1
                                else:
                                    rclass=8
                                    small_hybrid+=1
                     
                        else:
                            if rclass!=5:         
                                rclass=5
                                unresolved+=1

                if rclass==0:
                    if distmax1>=8.0:

                        if dist1>=halfdist1:

                            if dist2>halfdist2:
                                rclass=2
                                fr2+=1
                            else:
                                rclass=4
                                fr21+=1

                        else:

                            if dist2<=halfdist2:
                                rclass=1
                                fr1+=1
                            else:
                                rclass=3
                                fr12+=1
                     
                    else:
                        if rclass!=5:         
                            rclass=5
                            unresolved+=1
       
                        
                    
        #Case for sources with size or dynamic rangefluxes or number of pixels < threshold    
        else:
            rclass=0
            faint+=1
            dist1=0.0
            distmax=0.0
            distmax1=0.0
            dist2=0.0
            distmax2=0.0
            angleDiff=0
            if pixtot<=1:
                normFlux=0.0
            
        #Create the plots
        palette=plt.cm.viridis
        palette.set_bad('k',0.0)
        plt.rcParams['figure.figsize']=(10.67,8.0)
        A=np.ma.array(a,mask=np.isnan(a))
        y,x = np.mgrid[slice((0),(sizey),1),
                       slice((0),(sizex),1)]
        plt.pcolor(x,y,A,cmap=palette,vmin=np.nanmin(A),vmax=np.nanmax(A))
        plt.title(str(step)+', '+source+', d1='+"{0:.1f}".format(dist1)+', d2='+"{0:.1f}".format(dist2)+', maxd1='+"{0:.1f}".format(distmax1)+', maxd2='+"{0:.1f}".format(distmax2)+', FR='+str(rclass)+', dRange='+"{0:.1f}".format(dynrange)+', Size='+"{0:.1f}".format(deconv_siz), fontsize=12)
        plt.rc('xtick', labelsize=14)
        plt.rc('ytick', labelsize=14)
        plt.axis([x.min(),x.max(),y.min(),y.max()]) 
        plt.colorbar()

        plt.plot(distmax1_x,distmax1_y,'k^',markersize=13, mew=4)
        plt.plot(distmax1_x,distmax1_y,'^',color='orange',markersize=11, mew=2)
        plt.plot(distmax2_x,distmax2_y,'kv',markersize=13, mew=4)
        plt.plot(distmax2_x,distmax2_y,'v',color='orange',markersize=11, mew=2)
        
        plt.plot(xpeak,ypeak,'k2',markersize=17, mew=5)
        plt.plot(xpeak,ypeak,'2',color='cyan',markersize=15, mew=3)
        plt.plot(xpeak2,ypeak2,'k1',markersize=17, mew=5)
        plt.plot(xpeak2,ypeak2,'1',color='cyan',markersize=15, mew=3)

        plt.plot(rxpix,rypix,'kx',markersize=17, mew=4)
        plt.plot(rxpix,rypix,'rx',markersize=15, mew=2)
        
        plt.savefig(str(step)+'_A'+str(rclass)+'_'+source+'_map.pdf', dpi=300, format='pdf')
        plt.clf()


        #Write output table
        g.write(str(step)+' '+source+' '+str(rclass)+' '+str(coreBright)+' '+str(coreSep)+' '+str(normFlux)+' '+str(siz)+' '+str(deconv_siz)+' '+str(dynrange)+' '+str(dist1)+' '+str(dist2)+' '+str(distmax1)+' '+str(distmax2)+' '+str(PA1)+' '+str(PA2)+' '+str(angleDiff)+' '+str(failedMask)+' '+str(excludeComp)+'\n')

        
    #Copy region files and save with source name and index, for later checking.
    inclname=(str(step)+'_A'+str(rclass)+'_'+source+'_incl.reg')
    exclname=(str(step)+'_A'+str(rclass)+'_'+source+'_excl.reg')

    copyfile('temp.reg', inclname)
    copyfile('temp2.reg', exclname)
    

    #Increase the counter
    step+=1

g.close()
    
print 'Total of '+str(fr1)+' FRIs, '+str(fr2)+' FRIIs, '+str(fr12)+' FR I-II hybrids, '+str(fr21)+' FR II-I hybrids, '+str(small_fr1)+' small FRI candidates, '+str(small_fr2)+' small FRII candidates, '+str(small_hybrid)+' small hybrid candidates, '+str(unresolved)+' unresolved, '+str(faint)+' faint sources.'
run_time=time.time()-start_time

print 'Time elapsed: '+str(run_time)+' seconds'








