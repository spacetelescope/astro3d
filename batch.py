from __future__ import print_function, division
import texture


shape = (200,200)
sep = [25,15,5]
wid = [15,10,3]

def dobatch():
    for s,w in zip(sep,wid):
        for prof in ["spherical","linear"]:
            print(s,w,prof)
            lim = texture.lines(prof,shape,w,s,1.,0.)
            texture.save(lim, "lines_%s_spacing%d_width_%d" % (prof,s,w))
            dim = texture.dots(prof,shape,w,s,1., texture.point_grid(shape,s))
            texture.save(lim, "dots_%s_spacing%d_width_%d" % (prof,s,w))
            rlim = lim.transpose()
            lim[rlim > lim] = rlim[rlim > lim]
            texture.save(lim, "hatch_%s_spacing%d_width_%d" % (prof,s,w))
        