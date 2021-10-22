import numpy as np
import sys
import os
from astropy.table import Table
from astropy.io import fits
import glob
from subim import extract_subim
from overlay import find_noise_area,find_noise
from download_image_files import LofarMaps
from astropy.coordinates import SkyCoord
import astropy.units as u

# Code to generate cutouts and numpy arrays from HETDEX galaxy zoo output catalogues.
# Reads the output of aggregate_lofgalzoo.py

def get_fits(field,fra,fdec,fsource,fsize):

    sc=SkyCoord(fra*u.deg,fdec*u.deg,frame='icrs')
    s=sc.to_string(style='hmsdms',sep='',precision=2)
    name=fsource
    newsize=2.5*fsize/3600.0

    #lm=LofarMaps()
    mosname=field
    filename='/beegfs/lofar/deepfields/data_release/'+field+'/radio_image.fits'
    hdu=extract_subim(filename,fra,fdec,newsize)
    if hdu is not None:
        hdu.writeto(field+'/cutouts/'+name+'.fits',overwrite=True)
        flag=0
    else:
        print 'Cutout failed for '+str(fsource)
        flag=1

    return flag
# Read in tables

fieldname = sys.argv[1]
sourcecat = fieldname+'/final_withids_res.fits'

summary=Table.read(sourcecat)

# Go through summary table, use size field to define cutout reg. Use table with optIDs and no probs

for row in summary:
    maj=row['Maj']
    lgzsiz=row['LGZ_Size']
    ssource=row['Source_Name']
    #ssize=row['Size']
    ssource=ssource.rstrip()
    print "source is "+ssource+" and..."
    sra=row['RA']
    sdec=row['DEC']
    optra=row['optRA']
    optdec=row['optDec']
    flux=row['Peak_flux']
    rms=row['Isl_rms']
    
    if np.isnan(maj):
        ssize=lgzsiz
    else:
        ssize=maj*3600.0
   
        
    if ssize<20.0:
        ssize=20.0
    #print ssize
    #continue
    flag=get_fits(fieldname,sra,sdec,ssource,ssize)

    cutout=fieldname+'/cutouts/'+ssource+'.fits'
    dname='/beegfs/lofar/deepfields/data_release/'+fieldname+'/radio_image.fits'
    if os.path.isfile(dname):
        lhdu=fits.open(dname)
        if flag==0:
            nlhdu=fits.open(cutout)
            #print "Size is: "+str(ssize)
            # Now just need to write out numpy arrays!
            
            d=nlhdu[0].data
            
            #rms=find_noise_area(nlhdu,sra,sdec,2.0*ssize/3600.0)[1]
            #rms=rmsval
            peak=flux
            dyncut=50.0
            '''
            ratio=peak/dyncut
            print peak,rms,ratio
            if rms<ratio:
                print "rms < ratio"
                thres=ratio
                thres2=ratio
                thres3=ratio
            else:
            '''
            print "rms > ratio"
            print rms
            thres=3.0*rms
            thres2=4.0*rms
            thres3=5.0*rms
                
            # print 'rms is',rms
            
            #        d[d<3*rms]=np.nan
            #        np.save("3rms/"+ssource+'.npy',d)
            #print thres2
            #print d.shape
            d[d<thres]=np.nan
            np.save(fieldname+"/newthres3rms/"+ssource+".npy",d)
            d[d<thres2]=np.nan
            np.save(fieldname+"/newthres4rms/"+ssource+'.npy',d)
            d[d<thres3]=np.nan
            np.save(fieldname+"/newthres5rms/"+ssource+'.npy',d)

