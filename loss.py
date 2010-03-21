#! /usr/bin/env python
from __future__ import print_function
import numpy as np
import os
import pdb
import subprocess
import sys
np.random.seed(0)
class Packet1(object):
    def __init__(z, ninter, ncont):
        """
        
        ninter -- is the number of the interval for which the packet was
        generated.
        
        ncont  -- number of sources that contributed to the packet
        
        """
        z.ninter = ninter
        z.ncont = ncont
def plot_logical(f):
    def helper(f, i):
        ch = [helper(f, j) for j, h in enumerate(f) if h == i] 
        return "child {node {%d} %s}" % (i, " ".join(ch))
    filename = 'ztree'
    os.system('rm %s.*' %filename)
    s = '''\documentclass[landscape,a4paper]{article}
\usepackage{tikz}
\\begin{document}
\\thispagestyle{empty}
\\begin{tikzpicture}[level distance=10mm,
level/.style={sibling distance=40mm/#1}]
\\node {0} [grow'=down] %s;
\end{tikzpicture}
\end{document}''' % " ".join([helper(f, j) for j, h in enumerate(f) if h==0])
    with open('%s.tex' %filename,'w') as f: f.write(s)
    subprocess.call(['pdflatex', '%s.tex' %filename])
    subprocess.Popen(['xpdf', '%s.pdf' %filename])
def plot_logical_test():
    """Test the plot_logical function
    
    [[shell:python net.py plot_logical_test]]

    """
    print("Hi")
    f = [-1, 0, 0, 1, 1, 1, 4]
    plot_logical(f)
    def postorder(f,i):
        """
        
        f -- parent vector
        i -- node index

        """
        for j,k in enumerate(f):
            if k == i:
                for h in postorder(f,j):
                    yield h
        yield i
    for q in postorder(f,0):
        print(q)

def loss_fun(f,ps):
    """
    f - parent vector

    ps - success probability of each link

    """
    pass
class LossNode(dict):
    def __init__(z, id, size):
        z.id = id
        z.size = size
    def __str__(z):
        return "Node %d: {%s}" % (z.id, ', '.join(
            ['%d:%d' % (key, z[key]) for key in sorted(z.keys())]))
    def extract(z):
        """Select the next packet that will be extracted."""
        return 
    def postorder(z, foo, *args, **kwargs):
        """Execute function with name foo in this subtree in postorder."""
        for n in z.ch:
            n.postorder(foo, *args, **kwargs)
        getattr(z, foo)(*args, **kwargs)
    def print(z):
        print(z)
    def discard(z):
        if z.id:
            for key in sorted(z.keys())[0:-z.size]:
                del z[key]
    def push(z, ninter, scount):
        """Receive packet from t=ninter and scount sources."""
        if z.has_key(ninter):
            z[ninter] += scount
        else:
            z[ninter] = scount
        z.discard()
    def round(z, t):
        """Execute the transmission of each node in each TDMA frame."""
        z[t] = 1
        if np.random.rand() < z.ps: # Success
            key  = min(sorted(z.keys()))
            z.f.push(key, z.pop(key))
        else:
            z.discard()
class LossTree(list):
    def __init__(z, fv, ps, size):
        z[:] = [LossNode(i, size=size) for i in xrange(len(fv))]
        for i, node in enumerate(z):
            node.f = z[fv[i]]
            node.ch = [z[j] for j, k in enumerate(fv) if k==node.id]
            node.ps = ps[i]
        z.postorder = 
    def postorder(z, foo, *args,**kwargs):
        z[0].postorder(foo, *args, **kwargs)
    def rounds(z, iterations):
        z.iterations = iterations
        for i in xrange(iterations):
            for node in z:
                node[t] = 1
            for node in z[0].ch:
                node.postorder('round', i)
        print("End of computation")
def test_postorder():
    fv = [-1, 0, 0, 1, 1, 1, 4]
    ps = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
    t = LossTree(fv, ps, 3)
    plot_logical(fv)
    t.postorder('print')
def test_simple_loss():
    fv = [-1, 0, 1]
    ps = [1, 1, 1]
    size = 4
    t = LossTree(fv, ps, size)
    t.rounds(3)
    print(t)
if __name__ == '__main__':
    if len(sys.argv) > 1:
        exec '{0}({1})'.format(sys.argv[1], ', '.join(sys.argv[2:]))
