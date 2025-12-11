
# from ngsolve import *
from netgen.occ import *
from netgen.meshing import IdentificationType
from math import *
import numpy as np
import sys
#from mpi4py import MPI

def naca4(number, n):

    m = float(number[0])/100.0
    p = float(number[1])/10.0
    t = float(number[2:])/100.0

    a0 = +0.2969
    a1 = -0.1260
    a2 = -0.3516
    a3 = +0.2843
    a4 = -0.1036

    x = np.linspace(0.0, 1.0, n+1)

    yt = [5*t*(a0*sqrt(xx)+a1*xx+a2*pow(xx, 2)+a3 *
               pow(xx, 3)+a4*pow(xx, 4)) for xx in x]

    xc1 = [xx for xx in x if xx <= p]
    xc2 = [xx for xx in x if xx > p]

    if p == 0:
        xu = x
        yu = yt

        xl = x
        yl = [-xx for xx in yt]

        xc = xc1 + xc2
        zc = [0]*len(xc)
    else:
        yc1 = [m/pow(p, 2)*xx*(2*p-xx) for xx in xc1]
        yc2 = [m/pow(1-p, 2)*(1-2*p+xx)*(1-xx) for xx in xc2]
        zc = yc1 + yc2

        dyc1_dx = [m/pow(p, 2)*(2*p-2*xx) for xx in xc1]
        dyc2_dx = [m/pow(1-p, 2)*(2*p-2*xx) for xx in xc2]
        dyc_dx = dyc1_dx + dyc2_dx

        theta = [atan(xx) for xx in dyc_dx]

        xu = [xx - yy * sin(zz) for xx, yy, zz in zip(x, yt, theta)]
        yu = [xx + yy * cos(zz) for xx, yy, zz in zip(zc, yt, theta)]

        xl = [xx + yy * sin(zz) for xx, yy, zz in zip(x, yt, theta)]
        yl = [xx - yy * cos(zz) for xx, yy, zz in zip(zc, yt, theta)]

    X = xu[::-1] + xl[1:]
    Z = yu[::-1] + yl[1:]

    return X, Z

def occ_naca_profile(type = "2412", width=4, height=4, depth=0, angle=0, h=0.01):
    thanks = "The occ_naca_profile function was provided by Xaver Mooslechner. Thanks!"
    if "mpi4py" in sys.modules:
        if (MPI.COMM_WORLD.rank == 0):
                print(thanks)
    else:
        print(thanks)

    xs, ys = naca4("2412", 600)
    pnts = []
    for i in range(len(xs)):
        pnts.append( (xs[i], ys[i],0) )
    rect = Rectangle(width,height).Face().Move((-width/2+1,-height/2,0))
    rect.edges.name = "outlet"
    rect.edges.Min(X).name = "inlet"
    curve = Wire(SplineApproximation(pnts))
    wing = Face(curve)
    wing.edges.name = "wall"
    wing.edges.maxh = h
    wing = wing.Rotate(Axis((0,0,0), Z),-angle)
    air = rect - wing
    if (depth > 0):
        domain = air.Extrude(depth)
        domain.faces.Min(Z).name = "periodic"
        domain.faces.Max(Z).name = "periodic"
        domain.faces.Max(Z).Identify(domain.faces.Min(Z), "periodic", IdentificationType.PERIODIC)
        return domain
    else:
        return air
