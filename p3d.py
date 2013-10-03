"""module to create 3d renderings from spreadsheet format"""
from __future__ import print_function, division
import os.path
from matplotlib import mlab
import convexhull
import numpy as np
from scipy.interpolate import Rbf
from scipy.interpolate import RectBivariateSpline
import texture


def readspreadsheets():
    sdict = {}
    # get stars
    keys = ['stars','filaments','gas','dust','dustgas']
    names = ['x','y','layer1','layer2','layer3','layer4','flux']
    for key in keys:
        skip = 1
        sdict[key] = mlab.csv2rec(key+'.csv',skiprows=skip)
        sdict[key].dtype.names = names
    return sdict

def complete_grid(z, debug=False):
    """what grid coordinates are missing, assuming a spacing of about
    4.5 or so, and return a set of full possible grid coordinates"""
    zc = z.copy()
    zc.sort()
    zdiffs = zc[1:] - zc[:-1]
    indchange = np.where(zdiffs > 1) # ignore small differences
    unique = list(zc[indchange]) # unique values
    aunique = np.array(unique)
    if debug: print("aunique", aunique)
    zdiffs = aunique[1:] - aunique[:-1]
    indmissing = np.where(zdiffs > 7) # indicates a missing value or more
    # add value for last diff
    if debug: print('indmissing', indmissing)
    unique = unique + [zc[indchange[0][-1]+1]]
    for im in indmissing[0][::-1]:
        diff = zdiffs[im]
        ngap = int(diff/4.8)
        # insert new values into gaps
        vals = list((np.arange(ngap)+1)*diff/(ngap+1) + unique[im])
        unique = unique + vals
    unique.sort()
    if min(unique) > 4.5:
        unique = [min(unique)-4.5] + unique
    unique = [0.] + unique
    unique = unique + [max(unique)+4.5,max(unique)+9]
    return unique
    
def fillgrid(tab):
    """figure out a near regular spacing for x and y, pad to """
    flux =tab['flux']
    xin = tab['x']
    yin = tab['y']
    xg = np.array(complete_grid(xin))
    yg = np.array(complete_grid(yin))
    # make full set of all possible coordinates of sampling grid
    npoints = len(xg)*len(yg)
    x, y = np.meshgrid(xg, yg)
    # zero all grid values
    v = 0.*x
    # for each set of x,y points in the spreadsheet, find the closest grid
    # point in that set and set the value to the spreadsheet value
    for i,point in enumerate(zip(xin,yin)):
        d2 = (point[0]-x)**2 + (point[1]-y)**2
        if d2.min() < 0.1:
            matchind = np.where(d2 == d2.min())
            v[matchind] = flux[i]
    return x, y, v
    
def interpolate_grid(x,y,v,xsize,ysize,scale=1.,epsilon = 4.5):
    """generate an array of xsize,ysize dimensions, and 
    interpolate the values of at x,y onto this array
    after scaling x and y by scale. An error is generated
    if the maximum of the scaled x or y is larger than the
    given sizes"""  
    xs = x * scale
    ys = y * scale
    epsilon = epsilon * scale
    if xs.max() >= xsize or ys.max() >= ysize:
        raise ValueError('scale, xsize, ysize incompatible with supplied x, y values')
    yo, xo = np.meshgrid(np.arange(ysize),np.arange(xsize))
    # now do interpolation
    rbf = Rbf(xs, ys, v, epsilon=epsilon)
    im = rbf(xo, yo)
    return im    

def spline_interpolate(x,y,v, xsize,ysize, scale=1.):
    """use spline interpolation given a regular set of sampled grid point values"""
    xs = x[:,0]*scale
    ys = y[0]*scale
    print (xs, ys)
    spline = RectBivariateSpline(xs,ys,v,(0,xsize,0,ysize))
    yo, xo = np.meshgrid(np.arange(ysize),np.arange(xsize))
    return spline(xo, yo)
    
def lin2interp(im):
    """simpleminded factor of 2 linear interpolation
    produces an image with twice the size in each dimension"""
    shape = im.shape
    out = np.zeros((shape[0]*2,shape[1]*2))*1.
    out[::2,::2] = im
    out[1:-1:2] = (out[:-2:2] + out[2::2])/2
    out[:,1:-1:2] = (out[:,:-2:2] + out[:,2::2])/2
    return out
    
def make_combined():
    """read in csv file and generate data to represent 
    summed elements over all 4 planes """
    sdict = readspreadsheets()
    idict = {}
    ysize = 1100
    xsize = int(1100*110./90)
    scale = 2.5
    offset = 52/4
    for key in sdict:
        print(key)
        if key == 'stars':
            idict[key] = sdict[key]
            continue
        x,y,v = fillgrid(sdict[key])
        im = interpolate_grid(y+offset,x+offset,v,int(ysize/4),int(xsize/4),scale)
        im[im<0] = 0
        #idict[key] = im
        idict[key] = lin2interp(lin2interp(im))
    return idict    

def make_masks(idict):
    """from the summed component images, generate the effective mask regions
    for each component"""
    # sum gas and filaments
    gas = (idict['gas'] + idict['filaments']) > 100
    dust = idict['dust'] > 15
    dustgas = idict['dustgas'] > 35
    dustgas[830:] = 0
    # where gas overlays dust, set dustgas to 1
    dustgas[gas & dust] = 1
    # where gas overlays dust, or dustgas, set it to 0
    gas[gas & dust] = 0
    gas[gas & dustgas] = 0
    # same for dust 
    dust[dust & dustgas] = 0
    # none of these should overlap now
    return {'gas':gas, 'dust':dust, 'dustgas':dustgas}
    
def make_3d(idt=None,intensity=False):
    offset = 52
    if intensity:
        ifactor = 1.
    else:
        ifactor = 0.
    if idt is None:
        idict = make_combined()
    else:
        idict = idt
    if intensity:
        igas = idict['gas'] + idict['filaments']
        idust = idict['dust']
        idustgas = idict['dustgas']
    masks = make_masks(idict)
    shape = masks['gas'].shape
    print(shape)
    gashexgrid = texture.hex_grid(shape,7)
    gastex = texture.dots("linear",shape,7,1,gashexgrid)
    dusttex = texture.lines("linear",shape,5*3, 5*5, 0.7, 0)
    dustgashexgrid = texture.hex_grid(shape,2*5)
    dustgastex = texture.dots('linear',shape,7,3,dustgashexgrid)
    im = masks['gas']*gastex + masks['dust']*dusttex + masks['dustgas']*dustgastex
    if intensity:
        iim = masks['gas']*igas + masks['dust']*idust + masks['dustgas']*idustgas
        im = im + iim/iim.max() * 100
    # now add stars
    stars = idict['stars']
    x = stars['x'] * 10 + offset
    y = stars['y'] * 10 + offset
    mag = np.log(stars['flux'])
    # scale mag to allowable star size ranges
    mags = ((mag-mag.min())/(mag.max()-mag.min())*15 + 10).astype(np.int32)
    nostar_im = im.copy()
    for xs, ys, mag in zip(x, y, mags):
        star = texture.make_star(mag, 10)
        size = len(star)
        subim = im[ys-mag:ys-mag+size,xs-mag:xs-mag+size]
        nostar_subim = nostar_im[ys-mag:ys-mag+size,xs-mag:xs-mag+size]
        subim[star>0] = star[star>0] + ifactor*nostar_subim[star>0].max()
    return im
    
