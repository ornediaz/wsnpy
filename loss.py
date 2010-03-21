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
    def round1(z, iterations):
        z.iterations = iterations
        for i in xrange(iterations):
            for node in z[1:]:
                node[i] = 1
                node.discard()
            for node in z.postorder_list:
                if np.random.rand() < node.ps: # Success
                    key  = min(sorted(node.keys()))
                    value = node.pop(key)
                    if node.f.has_key(key):
                        node.f[key] += value
                    else:
                        node.f[key] = value
                    node.f.discard()
        average = 0.0
        for count in z[0].itervalues():
            average += count / float(iterations)
        print(average)
    def round2(z, iterations):
        z.iterations = iterations
        for i in xrange(iterations):
            for node in z[1:]:
                node[i] = 1
                node.discard()
            for node in z.postorder_list:
                if np.random.rand() < node.ps: # Success
                    key  = min(sorted(node.keys()))
                    value = node.pop(key)
                    if node.f.has_key(key):
                        node.f[key] += value
                    else:
                        node.f[key] = value
                    node.f.discard()
        results = np.zeros(iterations)
        for i in xrange(iterations):
            if z[0].has_key(i):
                results[i] = z[0][i]
        print(results.mean())
        pdb.set_trace()
def test_postorder():
    fv = [-1, 0, 0, 1, 1, 1, 4]
    ps = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
    t = LossTree(fv, ps, 3)
    plot_logical(fv)
    t.postorder('print')
def test_postorder2(repetitions):
    fv = [-1, 0, 1]
    ps = [1, 1, 0.5]
    t = LossTree(fv, ps, 3)
    # plot_logical(fv)
    # t.postorder('print')
    # print(t.postorder_list)
    t.round2(repetitions)
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
