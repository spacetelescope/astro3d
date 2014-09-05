import numpy as np

def write_ascii(tri, norm, filename):
    """
    write the surface model to an ascii format STL file. 
    This function appends .STL to the given filename
    """
    print "writing out %d triangles" % len(norm)
    f = open(filename+".STL", 'w')
    f.write('solid bozo\n')
    for t,n in zip(tri,norm):
        f.write("facet normal %e %e %e\n" % tuple(n))
        f.write("\touter loop\n")
        for point in t:
            f.write("\t\tvertex %e %e %e\n" % tuple(point))
        f.write("\tendloop\n")
        f.write("endfacet\n")
    f.write('endsolid bozo')
    f.close()
    
def write_binary(tri, norm, filename):
    """
    Write a binary version of an STL file
    """
    strhdr = "binary STL format"
    strhdr = strhdr + (80-len(strhdr))*" "
    ntri = len(tri)
    print "writing out %d triangles" % ntri
    newtri = tri.reshape(-1,9)
    larray = np.zeros((1,),dtype="<u4")
    larray[0] = ntri
    bin = np.hstack((norm, newtri))
    cbin = bin.astype("<f4")
    ubc = np.zeros((ntri,1),dtype="<u2")
    ucbin = np.fromstring(cbin.tostring(),dtype='<u2').reshape((-1,24))
    tbin = np.hstack((ucbin,ubc))
    f = open(filename+".STL","wb")
    f.write(strhdr)
    f.write(larray.tostring())
    f.write(tbin.tostring())
    f.close()
    