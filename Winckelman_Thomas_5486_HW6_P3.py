
#
##
###
####
#####
####
###
##
#

import matplotlib.pyplot as plt
import numpy as np
from numba import njit



@njit
def MySimpson(x, y):                # assumes that len(x) is odd
    n = int(0.5*len(x))
    a = 0
    for j in range(n):
        i = 2*j
        a += (x[i+2]-x[i])*(y[i] + 4*y[i+1] + y[i+2])
    return a/6


@njit(fastmath=True)
def FEmaker(V, GridforFEM, domain, n):            # assumes a uniform stepsize
    x = GridforFEM
#   a = x[1:] - x[:len(x)-1]
    for i in range(1,1+len(V)):
        for j in range(len(domain)):
            t = domain[j]
            if x[i-1]<t and t<=x[i]:
                V[i-1][j] = t - x[i-1] # / a1 (x[i]-x[i-1])
            elif x[i]<t and t<x[i+1]:
                V[i-1][j] = x[i+1] - t # /a2 (x[i+1]-x[i])


@njit(fastmath=True)
def CompleteTheCoefficientMatrix(A, n, h, g1, g4):
    h = np.power(h,-2)                  # h[i] corresponds to h_{i+1}^2 in lecture notes 24
    x = np.multiply(h,g1)               # g1[i] corresponds to int_{t[i+1]}^{t[i+2]} p(s) ds
    A[n-1][n-2] = -x[n-2] + g4[n-2]
    A[n-1][n-1] += x[n-1] + x[n]
    A[0][0] += x[0] + x[1]
    A[0][1] = -x[1] + g4[0]
    for i in range(1,n-1):
        A[i][i+1] = -x[i+1] + g4[i]
        A[i][i] += x[i] + x[i+1]
        A[i][i-1] = -x[i] + g4[i-1]


@njit(fastmath=True)
def FunctionMultiplier1(VF,V,f):
    for i in range(len(VF)):
        for j in range(len(VF[0])):
            VF[i][j] = V[i][j]*f[j]


@njit(fastmath=True)
def FunctionMultiplier2(VF,v,F1,F2):
    for i in range(len(VF)):
        for j in range(len(VF[0])):
            VF[i][j] = v[j]*F1[i][j]*F2[i][j]


@njit
def MakeB(B, a, b, quad_res, n):
    GridforFEM = np.linspace(a, b, n+2)
    for j in range(n+1):
        B[j] = np.linspace(GridforFEM[j], GridforFEM[j+1], quad_res)
    return GridforFEM


@njit
def finalize(soln, V, final):
    for i in range(len(final)):
        final[i] = np.dot(soln,V[:,i])

#
#   initialize (this helps Numba, somehow)
#
x = np.linspace(0,1,11)
y = x**2
t = MySimpson(x,y)
V = np.zeros((4,2))
FEmaker(V, x, x, 2)
CompleteTheCoefficientMatrix(V, 2, 2, x, x)
FunctionMultiplier1(V,V,x)
FunctionMultiplier2(V,x,V,V)
MakeB(V, 0, 1, len(V[0]), 3)


def FEM4SL(p, r, f, n, a=0, b=np.pi, alpha=0, beta=0, quad_res=100, ReportProgress=True):
        #
        #
        #   pre-process
        #
        #
    quad_res = int(quad_res)
    if not (alpha==0 and beta==0):
        raise ValueError('Currently, only alpha=0 and beta=0 is supported.')
    if quad_res<3:
        print("Warning: " + str(quad_res) + " is less than the minimal accepted " +
              "value of 'quad_res' (3) to which the setting was therefore switched.")
        quad_res = 3
    if not a<b:
        raise ValueError('Please make a<b.')
    if ReportProgress:
        print('')
        print('    Now pre-Pprocessing', end=' ')
    if quad_res%2 == 0:
        quad_res += 1
        #
        #
        #   make lots of different grids
        #
        #
    B = np.zeros(( 2, quad_res ))
    GridforFEM = MakeB( B, a, b, quad_res, 2 )
    B = np.zeros(( n+1, quad_res ))
    GridforFEM = MakeB( B, a, b, quad_res, n )
    if ReportProgress:
        print(' ... ', end=' ')
    if isinstance(p, int):
        p = p*np.ones(np.shape(B))
    else:
        temp = p(np.ones((2,1)))
        p = p(B)                                        # assumes p, r, and f can be passed a numpy.array
    if isinstance(f, int):
        f = f*np.ones(np.shape(B))
    else:
        temp = f(np.ones((2,1)))
        f = f(B)                                        # dimensions are (n+1)-by-quad_res
    if isinstance(r, int):
        r = r*np.ones(np.shape(B))
    else:
        temp = r(np.ones((2,1)))
        r = r(B)
    domain = np.reshape(B[:,:quad_res-1], (1,-1))[0]    # a discretization of the interval [a,b]
    f = np.reshape(f[:,:quad_res-1], (1,-1))[0]         # the values of f on said discretization
    r = np.reshape(r[:,:quad_res-1], (1,-1))[0]         # the values of r on said discretization
    if ReportProgress:
        print(' ... ', end=' ')
        #
        #
        #   make the eponymous finite elements
        #
        #
    m = len(domain)
    V = np.zeros((n,m))
    FEmaker( V, GridforFEM, domain, n )                 # each row of V is the discretized range of one of the FE's
    V *= (n+1)/(b-a)  # V /= (b-a)/(n+1)
    b = np.copy(V)
    FunctionMultiplier1( b, V, f )                      # if the discretized range of v_i-times-f where v_i is one of the FE's
    MainDiagonal = np.zeros((n,m))
    OffDiagonal = np.zeros((n-1,m))
    FunctionMultiplier2( MainDiagonal, r, V, V )
    FunctionMultiplier2( OffDiagonal, r, V[1:], V[:len(V)-1] )
    if ReportProgress:
        print(' ... ')
        #
        #   compute all integrals
        #   for i=0,...,n compute the integrals int_{t[i]}^{t[i+1]}p(s)ds
        #   and for i=1,...,n compute the integrals int_{t[i]}^{t[i+1]}v_i(s)f(s)ds
        #
        print('    Now Computing Integrals.')
    g1 = np.zeros(n+1)
    g2 = np.zeros(n)
    g3 = np.zeros(n)
    g4 = np.zeros(n-1)
    g1[0] = MySimpson( B[0], p[0] )
    if ReportProgress:
        for j in range(n):
            g1[j+1] = MySimpson( B[j+1], p[j+1] )
            g2[j] = MySimpson( domain, b[j] )
            g3[j] = MySimpson( domain, MainDiagonal[j] )
            if j<n-1:
                g4[j] = MySimpson( domain, OffDiagonal[j] )
    else:
        for j in range(n):
            g1[j+1] = MySimpson( B[j+1], p[j+1] )
            g2[j] = MySimpson( domain, b[j] )
            g3[j] = MySimpson( domain, MainDiagonal[j] )
            if j<n-1:
                g4[j] = MySimpson( domain, OffDiagonal[j] )
    if ReportProgress:
        print('    Now post-processing.')
        print('')
        #
        #
        #   setup the linear SOE
        #
        #
    h = GridforFEM[1:] - GridforFEM[:len(GridforFEM)-1]
    # return g1, g2, g3, g4, h
    A = np.diag(g3)                     # hg1[i] corresponds to (h_{i+1}^(-2))int_{t[i+1]}^{t[i+2]} p(s) ds
    CompleteTheCoefficientMatrix(A, n, h, g1, g4)
    b = g2
    soln = np.linalg.solve(A,b)
    return soln, V, domain




@njit
def p(x):
    return 1+np.sin(x)


@njit
def f(x):
    return np.sin(x)


temp = p(np.ones((2,2)))
temp = f(np.ones((2,2)))


#
##
###
##
#



def temp(p, r, f, n=30, a=0, b=np.pi, quad_res=10, Plot=True, PlotBases=True):
    soln, V, domain = FEM4SL(p, r, f, a=a, b=b, n=30, quad_res=10, ReportProgress=False)
    soln, V, domain = FEM4SL(p, r, f, a=a, b=b, n=30, quad_res=10, ReportProgress=False)
    soln, V, domain = FEM4SL(p, r, f,  a=a, b=b, n=n, quad_res=quad_res)
    final = np.zeros(len(V[0]))
    finalize(soln,V,final)
    if Plot:
        fig, ax = plt.subplots()
        if PlotBases:
            for j in range(len(V)):
                plt.plot(domain, soln[j]*V[j])
            plt.plot(domain,final, '--', linewidth=1, color='black')
        else:
            plt.plot(domain,final)
        ax.grid()
        fig.tight_layout()
        plt.show()
    return soln, V, domain, final



t = temp(p, 1, f, n=30)
t = temp(p, 1, f, n=500)
t = temp(p, 1, f, n=2000, PlotBases=False)




    
