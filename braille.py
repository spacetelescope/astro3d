import texture
import numpy as np

top = 24
mid = 12
bot = 0

left = 0
right = 12

pattern = [(left, top), (left, mid), (left, bot), (right, top), (right, mid), (right, bot)]

def set_points(bval):
    """given my own binary encoding for braille, return the 
    coordinates corresponding to all the dots matching that encoding
    
    the scheme is 
    
    1  8
    2 16
    4 32
    
    as far as which bits of the value coorespond to which braille dot is 
    present
    """
    points = []
    for i in range (6):
        if 2**i & bval:
            points.append(pattern[i])
    return points
    
def make_char_cell(bval):
    points = np.array(set_points(bval))
    if len(points):
        xcoords = points[:,0] + 20
        ycoords = points[:,1] + 23
        im = texture.dots(profile="spherical", shape=(70,51), width=7, scale=1, 
            locations=zip(xcoords,ycoords))
        return im[10:-10,10:-10]
    else:
        return np.zeros((50,31),dtype=np.float32)   
        
def make_label(bvals):
    outim = np.zeros((50,len(bvals)*31),dtype=np.float32)
    for i,bval in enumerate(bvals):
        outim[:,i*31:31*(i+1)] = make_char_cell(bval)
    return outim
    