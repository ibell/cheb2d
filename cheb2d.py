import timeit
import ChebTools
import numpy as np

def Clenshawbycol(yscaled, m_c):
    # See https://en.wikipedia.org/wiki/Clenshaw_algorithm#Special_case_for_Chebyshev_series
    Norder = m_c.shape[1] - 1;
    u_k = 0; u_kp1 = m_c[:,Norder]; u_kp2 = 0;
    for k in range(Norder-1, 0, -1):
        # Do the recurrent calculation
        u_k = 2.0 * yscaled * u_kp1 - u_kp2 + m_c[:,k];
        # Update the values
        u_kp2 = u_kp1; u_kp1 = u_k;
    return m_c[:,0] + yscaled * u_kp1 - u_kp2;

def Clenshaw(xscaled, m_c):
    # See https://en.wikipedia.org/wiki/Clenshaw_algorithm#Special_case_for_Chebyshev_series
    Norder = len(m_c) - 1;
    u_k = 0; u_kp1 = m_c[Norder]; u_kp2 = 0;
    k = 0;
    for k in range(Norder-1, 0, -1):
        # Do the recurrent calculation
        u_k = 2.0 * xscaled * u_kp1 - u_kp2 + m_c[k];
        # Update the values
        u_kp2 = u_kp1; u_kp1 = u_k;
    return m_c[0] + xscaled * u_kp1 - u_kp2;

class Chebyshev2D:
    def __init__(self, xrange, yrange, degrees, f):
        degreex, degreey = degrees
        self.cex = ChebTools.generate_Chebyshev_expansion(degreex, lambda x: 1, *xrange)
        self.cey = ChebTools.generate_Chebyshev_expansion(degreey, lambda x: 1, *yrange)
        self.C = np.zeros([d+1 for d in degrees]).T
        for ir, ynode in enumerate(self.cey.get_nodes_realworld()):

            # For each nodal value of y, generate the expansion in x
            ce = ChebTools.generate_Chebyshev_expansion(degreex, lambda x: f(x, ynode), *xrange)
            self.C[ir, :] = ce.coef()

            # Check expansion
            # chk = ce.get_node_function_values() - f(ce.get_nodes_realworld(), ynode)

        self.m_xmin, self.m_xmax = xrange
        self.m_ymin, self.m_ymax = yrange
        self.xnodes = self.cex.get_nodes_realworld()

        self.L = self.buildL(len(self.cey.coef())-1)

    def scalex(self, x):
        """ Scale input x value into [-1,1] """
        return (2*x - (self.m_xmax + self.m_xmin)) / (self.m_xmax - self.m_xmin)

    def scaley(self, y):
        """ Scale input y value into [-1,1] """
        return (2*y - (self.m_ymax + self.m_ymin)) / (self.m_ymax - self.m_ymin)

    def eval(self, x, y):
        # Flatten in x direction with Clenshaw for each row in C to get the functional
        # values of the expansion at the ynodes
        xscaled = self.scalex(x)
        fatynodes = Clenshawbycol(xscaled, self.C) # f at Chebshev-Lobatto ynodes for specified value of xscaled

        # Build expansion from functional values at y nodes
        c = np.dot(self.L, fatynodes)
        # return Clenshaw(self.scaley(y), c)
        cey = ChebTools.ChebyshevExpansion(c, self.m_ymin, self.m_ymax)
        return cey.y(y)

    def buildL(self, N):
        L = np.zeros((N + 1, N + 1)) # Matrix of coefficients
        for j in range(N+1):
            for k in range(j, N+1):
                p_j = 2 if (j == 0 or j == N) else 1
                p_k = 2 if (k == 0 or k == N) else 1
                L[j, k] = 2.0 / (p_j*p_k*N)*np.cos((j*np.pi*k) / N);
                # Exploit symmetry to fill in the symmetric elements in the matrix
                L[k, j] = L[j, k]
        return L

def test_2DGaussian():
    f = lambda x,y: np.sin(x)*np.exp(-x**2-y**2)
    ce2D = Chebyshev2D((-1,1),(-1,1), degrees=(5,5), f=f)
    x = ce2D.xnodes[1]
    print(ce2D.C)
    print('values', ce2D.eval(x, 0.7), f(x, 0.7), ce2D.eval(x, 0.7)/f(x, 0.7)-1)

    cex = ChebTools.generate_Chebyshev_expansion(20, lambda x: np.sin(x), -1, 1)
    xx = np.linspace(-1, 1, 100)
    M = 1000
    tic = timeit.default_timer()
    for i in range(M):
        cex.y(xx)
        #Clenshaw(xx, cex.coef())
    toc = timeit.default_timer()
    print((toc-tic)/M/len(xx)*1e6,'us/call')

    M = 1000
    tic = timeit.default_timer()
    for i in range(M):
        ce2D.eval(x, 0.7)
    toc = timeit.default_timer()
    print((toc-tic)/M*1e6, 'us/call')

    tic = timeit.default_timer()
    for i in range(M):
        f(x, 0.7)        
    toc = timeit.default_timer()
    print((toc-tic)/M*1e6, 'us/call')

test_2DGaussian()