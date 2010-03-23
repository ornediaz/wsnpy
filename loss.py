#! /usr/bin/env python
from __future__ import print_function
import numpy as np
import os
import pdb
import subprocess
import sys
np.random.seed(0)
VB = False
def vprint(*args):
    if VB:
        print(*args)
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
    def __repr__(z):
        return "Node %d: {%s}" % (z.id, ', '.join(
            ['%d:%d' % (key, z[key]) for key in sorted(z.keys())]))
    def ret_post_order(z, node_list):
        for n in z.ch:
            n.ret_post_order(node_list)
        node_list.append(z)
    def print(z):
        print(z)
    def discard(z):
        if z.id:
            for key in sorted(z.keys())[0:-z.size]:
                del z[key]
class LossTree(list):
    def __init__(z, fv, ps, size):
        z[:] = [LossNode(i, size=size) for i in xrange(len(fv))]
        for i, node in enumerate(z):
            node.f = z[fv[i]]
            node.ch = [z[j] for j, k in enumerate(fv) if k==node.id]
            node.ps = ps[i]
        node_list = []
        for n in z[0].ch:
            n.ret_post_order(node_list)
        z.postorder_list = node_list
    def round3(z, iterations, rate):
        """Incorporating rate reduction."""
        z.iterations = iterations
        old = -1
        sensing_t = -1
        for frame in xrange(iterations):
            new = int(frame * rate)
            if new > old:
                sensing_t += 1
                old = new
                for node in z[1:]:
                    node[sensing_t] = 1
                    node.discard()
            vprint("Starting frame %d" %frame)
            for node in z.postorder_list:
                if np.random.rand() < node.ps and len(node): # Success
                    key = min(sorted(node.keys()))
                    value = node.pop(key)
                    tmp = node.f.get(key, 0) + value
                    node.f[key] = tmp
                    vprint("Node %d suceeded.  Value: %d" %(node.id, tmp))
                    node.f.discard()
        z.results = np.array([z[0].get(i, 0) for i in xrange(sensing_t + 1)])
    def round4(z, iterations, rate):
        """Incorporating rate reduction."""
        z.iterations = iterations
        old = -1
        sensing_t = -1
        for frame in xrange(iterations):
            new = int(frame * rate)
            if new > old:
                sensing_t += 1
                old = new
                for node in z[1:]:
                    node[sensing_t] = 1
                    node.discard()
            vprint("Starting frame %d" %frame)
            for node in z.postorder_list:
                if np.random.rand() < node.ps and len(node): # Success
                    key = max(node, key=node.get)
                    value = node.pop(key)
                    tmp = node.f.get(key, 0) + value
                    node.f[key] = tmp
                    vprint("Node %d suceeded.  Value: %d" %(node.id, tmp))
                    node.f.discard()
        z.results = np.array([z[0].get(i, 0) for i in xrange(sensing_t + 1)])
def test_postorder():
    fv = [-1, 0, 0, 1, 1, 1, 4]
    ps = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
    t = LossTree(fv, ps, 3)
    plot_logical(fv)
    t.postorder('print')
def test_rate2():
    fv = [-1, 0, 1, 1, 1 , 1]
    ps = np.ones(6) * 0.3
    size = 50
    rate_v = np.arange(0.05, 1.0, 0.1)
    test_rate(rate_v, fv, ps, size, 'round3')
    test_rate(rate_v, fv, ps, size, 'round4')
def test_rate(rate_v, fv, ps, size, alg):
    rate_v = np.arange(0.05, 1.0, 0.1)
    mean = var = tot = np.zeros(len(rate_v))
    print("****** Executing %s" % alg)
    print("%8s %8s %8s %8s" % ("rate", "mean", "std", "sum"))
    for i, rate in enumerate(rate_v):
        np.random.seed(0)
        t = LossTree(fv, ps, size)
        getattr(t, alg)(2000, rate)
        print("%8s %8.2f %8.2f %8.2f" % (rate, t.results.mean(),
            t.results.std(), t.results.sum()))
def test_simple_loss():
    fv = [-1, 0, 1]
    ps = [1, 1, 1]
    size = 4
    t = LossTree(fv, ps, size)
    t.round2(3,1.0)
    print(t.results.mean())
def test_rate_variation():
    old = -1
    for i in xrange(20):
        new = int(i * 3.0 / 5.0)
        if new > old:
            print("%3d: 1" %i)
            old = new
        else:
            print("%3d: 0" %i)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        exec '{0}({1})'.format(sys.argv[1], ', '.join(sys.argv[2:]))
