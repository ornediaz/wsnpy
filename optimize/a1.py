# This script is all wrong, because my functions are not convex. 

from cvxopt import solvers, matrix, spmatrix, mul, div
''' We want to solve:
min x/y
s.t x >= 1
    y <= 2

we take:
        p = e^x
        q = e^y 

we convert it to:
min ln(x) / ln y
s.t.    -x <= -1
         y <=  2
'''

G = matrix([   
                [-1.0, 0.0],
                [ 0.0, 1.0],
            ])
h = matrix([-1., 2., 0.], (3,1))
def F(x=None, z=None):
    if x is None:
        return 1, matrix((2.,1.),(2,1))
    if x[1] == 0: return None
    f = matrix([[1/x[0]+ x[1])
    Df = matrix([x[2], 0.0, 0.0, -x[2], x[0], -x[1]], (2,3))
    if z is None:
        return f, Df
    H = matrix(0.0, (3,3))
    return f, Df, H
sol = solvers.cp(F, G, h)
print sol['x']