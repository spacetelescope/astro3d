'''Create a unitsphere recursively by subdividing all triangles in an octahedron recursivly.

A unitsphere has a radius of 1, which also means that all points in this sphere
have an absolute value of 1. Another feature of an unitsphere is that the normals 
of this sphere are exactly the same as the vertices.

This recursive method will avoid the common problem of the polar singularity, 
produced by 2d parameterization methods.

If you wish a sphere with another radius than that of 1, simply multiply every single
value in the vertex array with this new radius 
(although this will break the "vertex array equal to normal array" property)
'''
import numpy as np


octahedron_vertices = np.array( [ 
    [ 1.0, 0.0, 0.0], # 0 
    [-1.0, 0.0, 0.0], # 1
    [ 0.0, 1.0, 0.0], # 2 
    [ 0.0,-1.0, 0.0], # 3
    [ 0.0, 0.0, 1.0], # 4 
#    [ 0.0, 0.0,-1.0]  # 5                                
] )
octahedron_triangles = np.array( [ 
    [ 0, 4, 2 ],
    [ 2, 4, 1 ],
    [ 1, 4, 3 ],
    [ 3, 4, 0 ],
#    [ 0, 2, 5 ],
#    [ 2, 1, 5 ],
#    [ 1, 3, 5 ],
#    [ 3, 0, 5 ]
] )

def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
    arr[:,0] /= lens
    arr[:,1] /= lens
    arr[:,2] /= lens                
    return arr

def divide_all( vertices, triangles, values=None ):    
    #new_triangles = []
    new_triangle_count = len( triangles ) * 4
    # Subdivide each triangle in the old approximation and normalize
    #  the new points thus generated to lie on the surface of the unit
    #  sphere.
    # Each input triangle with vertices labelled [0,1,2] as shown
    #  below will be turned into four new triangles:
    #
    #            Make new points
    #                 a = (0+2)/2
    #                 b = (0+1)/2
    #                 c = (1+2)/2
    #        1
    #            /\       Normalize a, b, c
    #          /   \
    #       b/______\ c    Construct new triangles
    #      /\      /\       t1 [0,b,a]
    #    /   \   /   \      t2 [b,1,c]
    #  /______\/______\     t3 [a,b,c]
    # 0       a        2    t4 [a,c,2]    
    v0 = vertices[ triangles[:,0] ]
    v1 = vertices[ triangles[:,1] ]
    v2 = vertices[ triangles[:,2] ]
    a = ( v0+v2 ) * 0.5
    b = ( v0+v1 ) * 0.5
    c = ( v1+v2 ) * 0.5  
    normalize_v3( a )
    normalize_v3( b )
    normalize_v3( c )
    # now average values if present
    if values is not None:
        val0 = values[triangles[:,0]]
        val1 = values[triangles[:,1]]
        val2 = values[triangles[:,2]]
        aval = (val0+val2) * 0.5
        bval = (val0+val1) * 0.5
        cval = (val1+val2) * 0.5
        newvalues = np.hstack((val0,bval,aval,bval,val1,cval,aval,bval,cval,aval,cval,val2))
    
    #Stack the triangles together.
    vertices = np.vstack(( 
             np.hstack((v0,b,a)).reshape((-1,3)),
             np.hstack((b,v1,c)).reshape((-1,3)),
             np.hstack((a,b,c)).reshape((-1,3)),
             np.hstack((a,c,v2)).reshape((-1,3)) 
             ))
    #Now our vertices are duplicated, and thus our triangle structure are unnecesarry.
    triangles = np.arange( len(vertices) ).reshape( (-1,3) )
    if values is not None:
        return vertices, triangles, newvalues     
    return vertices, triangles

def create_unit_hemisphere( recursion_level=2 ):
    vertex_array, index_array = octahedron_vertices, octahedron_triangles
    for i in range( recursion_level - 1 ):
        vertex_array, index_array  = divide_all(vertex_array, index_array)
    return remove_duplicate_vertices(vertex_array, index_array)

def create_pattern_hemisphere(recursion_level=5,dlevel=4,skip=1):
    """
    create a dot pattern on the surface of a sphere
    """
    ver, tri = create_unit_hemisphere(dlevel)
    # this set will have a value of 1 associated with all vertices
    # now compute the grid with 0 values (except at vertices that exist
    # for lower level just called)
    zver, ztri = create_unit_hemisphere(dlevel+skip)
    mind = identify_matching_vectors(ver, zver)
    print ver
    print zver
    print mind
    print "bozo"
    val = np.zeros(len(zver))*0.
    val[mind] = 1.
    ver, tri = zver, ztri
    for i in range(recursion_level-(dlevel+skip)):
        ver, tri, val = divide_all(ver, tri, val)
        print i
        #print ver, tri, val
        ver, tri, val = remove_duplicate_vertices(ver, tri, val)
    return ver, tri, val

def apply_intensity(ver, val, scale=1.):
    """
    Scale vertex vectors by desired factor. 
    Scale = 1 corresponds to a 1% scale increase for a val=1
    """
    return ver*(1+scale*val.reshape(-1,1)/100.)

def vertex_array_only_unit_sphere( recursion_level=2 ):
    vertex_array, index_array = create_unit_sphere(recursion_level)
    if recursion_level > 1:    
        return vertex_array.reshape( (-1) )
    else:
        return vertex_array[index_array].reshape( (-1) )

def vector_map(v):
    """
    map unit vectors into a complex number such that each vector is
    a unique complex number. Works so long as not relying on differences
    that require double precision.
    """
    x = (v[:,0]*10**6).astype(np.int32)
    y = (v[:,1]*10**6).astype(np.int32)
    z = v[:,2]
    return (x + y/10.**7) + z*1j 
    

def remove_duplicate_vertices(ver, tri, extra=None):
    u = vector_map(ver)
    # find duplicates
    nver, nind, ninv = np.unique(u, return_index=True, return_inverse=True)
    # update the triangle indices appropriately
    utri = ninv[tri]
    uver = ver[nind]
    if extra is not None:
        uextra = extra[nind]
        return uver, utri, uextra
    return uver, utri

def identify_matching_vectors(v1, v2):
    """assumes all of v1 is in v2"""
    u1 = vector_map(v1)
    u2 = vector_map(v2)
    # sort and keep record of orginal indices
    u1ind = np.argsort(u1)
    u2ind = np.argsort(u2) 
    su1 = u1[u1ind]
    su2 = u2[u2ind]
    inv2 = np.argsort(u2ind)
    mind = np.searchsorted(su2,su1)
    return inv2[mind]    
        
def unit_hemisphere(recursion_level=2):
        ver, tri = create_unit_hemisphere(recursion_level)
        # now clean out duplicate vertices and update triangle indices to
        # repoint to equivalent vertices
        return remove_duplicate_vertices(ver,tri)
        
def base_points(vertices):
    """
    Determine which vertices lie on the z=0 plane and sort them into
    order along the circle they lie on, return angle as second parameter
    """
    zver = vertices[vertices[:,2]==0.,:]
    angle = np.arctan2(zver[:,1],zver[:,0])
    ind = np.argsort(angle)
    return zver[ind]
    
def make_base(zver,pyramid_scale=.95):
    """
    Make a base of triangles from the sphere perimeter points and the
    interior pyramid which has a hexagonal base.
    """
    btri = []
    nper = zver.shape[0]
    breaks = [0, nper/6, nper/3, nper/2, 2*nper/3, 5*nper/6,nper]
    for i in range(6):
        prange = range(breaks[i],breaks[i+1])
        if i == 5:
            prange.append(0)
            iplus = 0
        else:
            prange.append(breaks[i+1])
            iplus = i+1
        btri = btri + zip(prange[:-1],prange[1:],[i+nper]*(len(prange)-1))
        btri.append((prange[-1],i+nper,iplus+nper))
    # now add interior surface
    ai = nper+6
    itri = [(nper  ,nper+1,ai),(nper+1,nper+2,ai),(nper+2,nper+3,ai),
            (nper+3,nper+4,ai),(nper+4,nper+5,ai),(nper+5,nper,  ai)]
    btri = btri + itri
    iangs = ((np.arange(6.)+.5)/6)*np.pi*2
    ixpoints = pyramid_scale*np.cos(iangs)
    iypoints = pyramid_scale*np.sin(iangs)
    bvertices = list(zver) + zip(ixpoints, iypoints, [0.]*6) + [(0.,0.,pyramid_scale)]
    return np.array(btri), np.array(bvertices)

def fixordering(tri, up=True):
    """
    Compute the normal vector, and if it is in the wrong direction, reorder
    the triangle vertices so that it is in the right direction. up=True means z
    component is supposed to be positive, False, negative.
    """
    # compute normals by standard cross product
    if up:
        sign = 1
    else:
        sign = -1
    cross = np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    mask = sign*cross[:,2] <= 0.
    # fix where mask is true
    ttri = tri.copy()
    tri[mask,0,:] = ttri[mask,2,:]
    tri[mask,2,:] = ttri[mask,0,:]
    cross = np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    return tri, normalize_v3(cross)

def make_model(nlevels=2, pyramid_scale=0.95):
    """
    Construct a list of triangles that define the surface of a spherical model
    """
    ver, tri = unit_hemisphere(nlevels)
    zver = base_points(ver)
    btri, bver = make_base(zver, pyramid_scale)
    # for the two components, make the substitution of actual vertex coordinates
    # into the triangle definitions
    exttri = ver[tri]
    inttri = bver[btri]
    exttri, extnorm = fixordering(exttri)
    inttri, intnorm = fixordering(inttri,up=False)
    tri = np.vstack((exttri,inttri))
    norm = np.vstack((extnorm, intnorm))
    return tri, norm
    
def make_imodel(nlevels, pyramid_scale=0.95):
    """
    Construct a list of triangles that define the surface of a spherical model
    """
    ver, tri, val = create_pattern_hemisphere(nlevels, 2, 1)
    ver = apply_intensity(ver, val, 10)
    zver = base_points(ver)
    btri, bver = make_base(zver, pyramid_scale)
    # for the two components, make the substitution of actual vertex coordinates
    # into the triangle definitions
    exttri = ver[tri]
    inttri = bver[btri]
    exttri, extnorm = fixordering(exttri)
    inttri, intnorm = fixordering(inttri,up=False)
    tri = np.vstack((exttri,inttri))
    norm = np.vstack((extnorm, intnorm))
    return tri, norm


        
        
        
    