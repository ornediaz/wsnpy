#! /usr/bin/env python

'''Provides:
* Network creation routines.
* Network plotting routines
* Tree construction routines
* Scheduling routines

Criteria for correct reception of a packet:
    DiskModelNetwork graph model:
    SINR model:
        The sinr threshold includes the noise introduced by the amplifier in
        the receptor.  
'''
from __future__ import print_function
import collections
import copy
import doctest
import glob
import inspect
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from operator import itemgetter
import os
import pdb
import platform
import SimPy.Simulation as simpy
import subprocess
import sys
sys.setrecursionlimit(2000) # otherwise copy.deepcopy raises a RuntimeError
import time as time_module
KBOL = 1.38e-23 # Boltzman constant
T0 = 290 # Room temperature in Kelvin (=17C)
INF = 9999 # Used in Dijkstra's algorithm
TIER_MAX = 9999 # This tier indicates that the node is disconnected
STATES = dict(tx=0.063, rx=0.030, id=0.030, sl=3-6) #Energy consumption
def display(filename):
    if filename.endswith('pdf'):
        if platform.system() == 'Windows':
            subprocess.Popen(['acrord32', filename])
        else:
            subprocess.Popen(['xpdf', filename])
    elif platform.system() == 'Windows':
        subprocess.Popen(['rundll32.exe', 'C:\WINDOWS\system32\shimgvw.dll,ImageView_Fullscreen', os.path.join(os.getcwd(),filename)])
    else:
        subprocess.Popen(['display', filename])
class Error(Exception): pass
class NoProgressError(Error):
    __str__ = lambda z:"No color assigned in several frames."
class UnsufficientSlots(Error):
    __str__ = lambda z:"No enough slots available."
class AdvertTError(Error):
    __str__ = lambda z:"Advert time is too short."
class IncompleteAggregationError(Error):
    __str__ = lambda z:"The aggregation schedule is incomplete."
class IncompleteConvergecastError(Error):
    __str__ = lambda z:"The convergecast schedule is incomplete."
class UnsufficientDensity(Error): 
    __str__ = lambda z: 'Insufficient node density for connectivity'
class OutOfSyncError(Error):
    __str__ = lambda z: "Out of sync error"
def vprint(*args):
    if VB:
        print(*args)
def stamp(z, d):
    d = copy.copy(d)
    for i, v in d.iteritems():
        if i == 'kwargs':
            for j, k in d['kwargs'].iteritems():
                setattr(z, j, k)
        if i != 'z':
            setattr(z, i,v)
def dijkstra(cost):
    '''Return every node's next hop to the sink using Dijkstra's algorithm.
    Parameter:
    cost -- NxN ndarray indicating cost of N nodes '''
    N = len(cost)
    dst = np.ones(N) * INF # Each node's smallest distance to the sink
    dst[0] = 0 # The source is at distance 0 from itz
    previous = - np.ones(N, dtype='int32') # parent of each node
    processed = np.zeros(N, dtype=bool)
    # If any node does not have neighbors, it will never be processed.
    while True:
        x = (dst + processed * INF * 2).argmin()
        if processed[x]: break
        processed[x] = True
        for y in np.r_[0:x, x + 1:N]:
            alt = dst[x] + cost[x, y]
            if alt < INF and alt < dst[y]:
                dst[y] = alt
                previous[y] = x
    return previous
def k_neigh(node_set, hops, tx_l):
    ''' Return set of nodes within a certain number of hops.
    Set of nodes a number of nodes within a number of hops of set of
    nodes. '''
    try:# Convert the list of nodes into a set
        node_set = set(node_set)
    except TypeError: # node_set is an integer (not iterable)
        node_set = set([node_set])
    if hops == 0:
        return set()
    for h in node_set:
        node_set = node_set.union(tx_l[h])
    return node_set.union(k_neigh(node_set, hops - 1, tx_l))
def plot_logical(f, lab=None):
    def helper(f, i):
        ch = [helper(f, j) for j, h in enumerate(f) if h == i] 
        lb = '' if lab is None else 'edge from parent node [fill=white,font=\scriptsize] {%0.2f}' % lab[i]
        return ("child {node {%d} %s %s}" % (i, " ".join(ch), lb))
    filename = 'ztree'
    os.system('rm %s.*' %filename)
    s = '''\documentclass[landscape,a4paper]{article}
\usepackage{tikz}
\\begin{document}
\\thispagestyle{empty}
\\begin{tikzpicture}[level distance=10mm,
level/.style={sibling distance=80mm /#1}]
\\node {0} [grow'=down] %s;
\end{tikzpicture}
\end{document}''' % " ".join([helper(f, j) for j, h in enumerate(f) if h==0])
    with open('%s.tex' %filename,'w') as f: f.write(s)
    subprocess.check_call(['pdflatex', '%s.tex' %filename])
    display(filename + '.pdf')
def plot_logical2(fv, ps=None):
    import yapgvb
    g = yapgvb.Digraph("tree")
    n = [g.add_node(str(i)) for i in xrange(len(fv))]
    for i in xrange(1, len(fv)):
        edge = n[i] >> n[fv[i]]
        if ps is not None:
            edge.label = str(ps[i])
    g.layout(yapgvb.engines.dot)
    fname = 'ztree.png'
    g.render(fname)
    g.write('ztree.dot')
    display(fname)
def plot_logical3(fv, ps=None, format='png', plot=2):
    #format = 'png'
    if plot == 0:
        return
    fname = 'ztree'
    with open(fname + '.dot', 'w') as f:
        f.write("digraph tree {\n")
        for i in xrange(1, len(fv)):
            lb = "" if ps is None else "[label = %s]" % ps[i] 
            f.write("%d -> %d %s;\n" % (i, fv[i], lb ))
        f.write("}\n")
    subprocess.check_call(['dot', '-T' + format, fname + '.dot', '-o',
        fname + '.' + format])
    if plot == 2:
        subprocess.check_call(['dot', '-Tpng', fname + '.dot', '-o', fname
            + '.png'])
        display(fname + '.png')
def plot_logical4(fv, ps=None):
    import dot2tex
    with open('ztree.tex', 'w') as f:
        f.write(dot2tex.dot2tex("digraph tree{\n%s}" % 
          ";\n".join(["%d -> %d %s" % (i, fv[i], 
          "" if ps is None else "[label = %s]" % ps[i]) 
          for i in xrange(1, len(fv))]), format='tikz',crop='True'))
    subprocess.call(['pdflatex', 'ztree.tex'])
    display('ztree.pdf')
class Pgf(list):
    def __init__(z, extra_preamble='\\usepackage{plotjour1}\n'):
        z.extra_preamble = extra_preamble
        z.extra_body = []
    def add(z, *args, **kwargs):
        '''Add new x-y graph.'''
        z.append(PgfAxis(*args, **kwargs))
    def __getattribute__(z, name):
        try:
            return object.__getattribute__(z, name)
        except AttributeError:
            return getattr(z[-1], name)
    def save(z, f_name=None, plot=2):
        b = []
        b.append('''\documentclass{article}
\usepackage[margin=0in]{geometry}
\usepackage{orne1}
''')
        if len(z.extra_preamble) == 1:
            b.append(z.extra_preamble)
        else:
            b.extend(z.extra_preamble)
        b.append('\\begin{document}\n')
        for i, axis in enumerate(z):
            b.append(2 * ' ' + '\\begin{tikzpicture}\n')
            b.append(4 * ' ' + '\\begin{axis}[%\n' + 6 * ' ')
            b.append((',%\n' + 6 * ' ').join(axis.options))
            b.append('\n' + 6 * ' ' +  ']%\n')
            b.extend(axis.buf)
            if axis.legend:
                b.append(6 * ' ' + '\\legend{{' + '}, {'
                        .join(axis.legend) + '}}%\n')
            b.append(4 * ' ' + '\\end{axis}\n')
            b.append(2 * ' ' + '\\end{tikzpicture}\n' + (i % 2) * '\n')
        b.extend(z.extra_body)
        b.append('\\end{document}')
        print(''.join(b))
        if f_name is None:
            f_name = name_npz()
        with open(f_name + '.tex', 'w') as f:
            f.writelines(b)
        if plot > 0:
            subprocess.call(['pdflatex', '{0}.tex'.format(f_name)])  
            os.remove(f_name + '.aux')
            os.remove(f_name + '.log')
        if plot == 2:
            display(f_name + '.pdf')
class PgfAxis():
    def __init__(z, xlabel='', ylabel=''):
        z.buf = []
        z.options = []
        z.opt('xlabel={{{0}}}'.format(xlabel))
        z.opt('ylabel={{{0}}}'.format(ylabel))
        z.legend = []
    def opt(z, *args):
        for arg in args:
            z.options.append(arg)
    def plot(z, x, y, leg=None, *args):
        """Plot vector y vs vector x.  leg is the legend and args are
        parameters to add to the addplot operation."""
        assert len(x) == len(y), "Vectors have different lengths"
        if len(args):
            z.buf.append(6 * ' ' + '\\addplot [' 
                    + (',\n' + 8 * ' ').join(args) + '] coordinates{%\n') 
        else:
            z.buf.append(6 * ' ' + '\\addplot coordinates {%\n')
        for i, (u, v) in enumerate(zip(x, y)):
            if i < len(x) - 1:
                z.buf.append(8 * ' ' + "({0}, {1})%\n".format(u, v))
            else:
                z.buf.append(8 * ' ' + "({0}, {1})}};%\n".format(u, v))
        if leg is not None:
            z.legend.append(leg)
    def mplot(z, x, ym, leg=None):
        # Plot the columns of matrix ym
        for i in xrange(ym.shape[1]):
            y = ym[:, i]
            if leg is None:
                z.plot(x, y)
            else:
                z.plot(x, y, leg[i])
class Tree(object): 
    '''Container of routines to be inherited by tree-like structures.'''
    def __init__(z, f=None, txD=None, ixD=None):
        '''Manual creation of a tree.

        Each node does not have a location assigned. 
        f     -- parent vector
        tx    -- pairs of nodes within TX range and
        ix    -- pairs of nodes within IX range. ''' 
        if f is None:
            f = [ - 1, 0, 0, 1, 2, 2]
        if txD is None:
            txD = {}
        if ixD is None:
            ixD = {0:(3, 4, 5), 1:(2, 4), 4:(5,)}
        z.c = len(f)
        v = range(z.c)
        z.f = f
        z.txL = [set() for i in v]
        z.ixL = [set() for i in v]
        for i, j in enumerate(f):
            if j >= 0:
                # Add parent and children to tx list
                z.txL[i].add(j)
                z.txL[j].add(i)
        # Add pairs given to txL and ixL
        for xL, xDict in zip((z.txL, z.ixL), (txD, ixD)):
            for i, neighborList in xDict.iteritems():
                for j in neighborList:
                    xL[i].add(j)
                    xL[j].add(i)
        z.xL = [a.union(b) for a, b in zip(z.txL, z.ixL)]
    def children(z, x):
        return [i for i, j in enumerate(z.f) if j == x]
    def plot_tree(z):        
        '''Plot a tree.
        Parameters:
        z.p       -- matrix indicating the coordinates of the nodes
        z.f       -- vector indicating the parent of each node '''
        assert len(z.p) == len(z.f), 'Vector length should equal WSN size'
        plt.ioff()
        plt.figure()
        plt.plot(z.p[:, 0], z.p[:, 1], 'o')
        plt.hold(True)
        x = max(z.p[:, 0])
        y = max(z.p[:, 1])
        for i, c in enumerate(z.p):
            plt.text(c[0] + x * 0.02, c[1], i)
        # Plot the line between each node and its parent. 
        for u, v in enumerate(z.f):
            if v >= 0: # Plot only nodes with a parent
                plt.plot(z.p[[u, v], 0], z.p[[u, v], 1])
        plt.axis([0, max(z.p[:, 0]) * 1.05, 0, max(z.p[:, 1] * 1.05)])
        plt.show()
    @property
    def tier_v(z):
        ''' Return vector indicating the parent of every node.
        z. contains f, the list of parents.''' 
        return np.array([z.tier(i) for i in xrange(len(z.f))])
    def tier(z, i):
        """Return tier of node i."""
        for tier in xrange(TIER_MAX):
            if i == 0: return tier
            elif i < 0: return TIER_MAX
            i = z.f[i]
        return tier
    def event_sources(z, loc, n_src):
        ''' Select the n_src sources closest to event and set them as
        working.

        Parameters:
        z.p   -- position vector
        z.f   -- parent vector 
        n_src -- number of sources. If 0, all nodes except the root
                 are sources. 
        '''
        assert sum(z.f >= 0) >= n_src, 'Sources exceed connected nodes'
        # Compute the distance from each node to the event
        event_dist = np.sqrt(np.sum(abs(z.p - loc) ** 2, axis=1))
        # To ensure that we do not select orphans or the sink a sources, set
        # their distance bigger than possible. 
        event_dist = np.where(z.f < 0, z.x + z.y, event_dist)
        return np.argsort(event_dist)[0:n_src]  
        # 'sort' arranges smallest first.
    def bf_schedule(z, hops=2, sort=False):
        '''Return tree schedule computed in breadth first order.

        Parameters:
        z.p       -- vector of coordinates; necessary if sort==True 
        z.f       -- vector of parents
        z.tx_l   -- tx_l[i] contains the list of neighbors of node i
        hops    -- number of hops within two nodes interfere with each other
        sort    -- sort the nodes according to their position? 
                   This sorting seems not to improve the performance,
                   so I recommend passing p=None.

        In the coloring process, no color within 'hops' hops away in the
        graph defined by 'tx_l' is used.
        
        The output is slot_v which is a vector that in position $i$
        indicates the slot assigned to node $i$.  In other words, every node
        is only assigned one slot, which means that the schedule is for data
        aggregation.
        
        '''
        c = len(z.f)
        # Initialize the color of all the nodes to -1
        slot_v = np.zeros(c, dtype=int)
        # Color the nodes in Breadth First order
        queue = copy.copy(z.children(0)) # Nodes to color (not the root)
        while queue:
            current_node = queue.pop(0) # node to color in current iteration
            next_to_add = np.array(z.children(current_node))
            if sort and len(next_to_add):
                p2 = z.p[current_node]
                vec = [p1[0] - p2[0] + 1j * (p1[1] - p2[1]) 
                       for p1 in z.p[next_to_add]]
                sortedIndex = np.argsort(np.angle(vec))
                next_to_add = next_to_add[sortedIndex]
            queue.extend(next_to_add)
            # Compute the set of nodes s whose color I cannot use. Two parts:
            # 1) Nodes that interfere with my parent's reception
            s = k_neigh(node_set=z.f[current_node], hops=hops, tx_l=z.tx_l)
            # 2) Nodes whose parents can interfere with me
            s1 = k_neigh(node_set=current_node, hops=hops, tx_l=z.tx_l) 
            for q in s1:
                s = s.union(set(z.children(q))) # Their parents
            # Start testing the next color to the one of the parent. 
            usedColors = set(slot_v[h] for h in s)
            clr = slot_v[z.f[current_node]] + 1
            while clr in usedColors:
                clr += 1
            slot_v[current_node] = clr
        # Invert the schedule
        m = max(slot_v)
        i = slot_v.nonzero()[0]
        slot_v[i] = m + 1 - slot_v[i]
        return slot_v
    def fat(z, src=None, working=None):
        '''Return parent list and connected node with the FAT method.

        Only the nodes indicated in the boolean vector 'working'
        work. Compare three techniques: using the default path, the FAT
        method, and reconstructing the tree. The method does not simulate
        packet exchange.  It is an iterative implementation. As such, it
        does not estimate T_tier.  However, the tree constructed should be
        similar to the one with FAT.

        The monitored area is a rectangle of length x and height y.  The
        origin of coordinates is in the southwestern corner and the
        coordinates of the diagonally opposed corner are (x,y). The sink
        is in (0,y/2) and the event occurs in (event_dist,y/2), where
        event_dist is the distance from the sink.

        ''' 
        if src is None:
            src = np.array([i for i, j in enumerate(f) if j >= 0])
        if working is None:
            working = np.ones(z.c, dtype='bool')
        working[0] = True # Ensure the sink is working
        working[src] = True # Ensure sources are working
        success = np.zeros(3, dtype='float32') # Results vector
        ''' Compute the success ratio using the default parent '''
        for u in src:
            while u != 0: # 0 is the sink
                u = f[u]
                if not working[u] or u < 0:
                    break
            else:
                success[0] += 1 # Success with fixed SPT
        ''' Compute the success using the FAT method '''
        fE = - np.ones(z.c, dtype=int) # The FAT
        # Initialize each node's number of children, which will be used in
        # choosing a parent.
        chN = np.zeros(z.c, dtype='int32') 
        # Initialize number of sources in complete subtree
        subTreeSize = np.zeros(z.c, dtype='int32') 
        # Initialize vector 'actvtd', that will list all the nodes that have
        # looked for children, which are those that received activation
        # tones. I use this list to compute energy consumption at the end
        # of the algorithm.
        actvtd = np.array([]) 
        # Execute FAT method starting from the tier of the sources further
        # away from the sink and ending in Tier 1.
        for tier in xrange(max(z.tr[src]), 0, - 1):
            # Determine nodes looking for parent in current tier
            aux = np.union1d(chN.nonzero()[0], src)
            active = aux[z.tr[aux] == tier]
            np.random.shuffle(active) 
            for i in active:
                # Simulate the effect of transmitting an activation tone by
                # adding neighbors within TX or IX range in next tier to the
                # 'actvtd' list.
                aux = np.array(z.txL[i] + z.ixL[i])
                if aux.size == 0: continue
                aux = aux[z.tr[aux] == tier - 1]
                actvtd = np.union1d(actvtd, aux)
                # Choose as parent the working node in the next tier with
                # the greatest number of children.
                aux = np.array(z.up[i])
                if aux.size == 0: continue
                aux = aux[working[aux]]
                if aux.size == 0: continue
                f = aux[chN[aux].argmax()]    # chosen parent
                chN[f] += 1
                subTreeSize[f] += subTreeSize[i] + (i in src)
                z.fE[i] = f
        success[1] = subTreeSize[0] # Success with FAT  
        ''' Compute the success reconstructing the tree '''
        # index of the nodes that do not work (Not Work)
        NW = np.array([i for i, j in enumerate(working) if not j])
        r = z.reachable
        r[NW, :] = INF
        r[:, Nw] = INF
        f = dijkstra(r)
        success[2] = sum(f[src] >= 0)
        reliability = success / src.size
        return fE, reliability
class CircularNet(object):    
    '''Network with nodes in a ring, and that can be segmented in
    clusters.'''
    def __init__(z):
        '''Compute the global tree (parent f_g and tier t_g).'''
        N = len(z.p)
        z.t_g = np.ones(N) * INF # Shortest distance to the sink
        z.t_g[np.sqrt((z.p**2).sum(1)) < z.r_a + z.tx_rg] = 0.0 
        z.f_g = - np.ones(N, dtype='int32') # parent of each node
        processed = np.zeros(N, dtype=bool)
        while True:
            x = (z.t_g + processed * INF * 2).argmin()
            if processed[x]: break
            processed[x] = True
            for y in np.r_[0:x, x + 1:N]:
                alt = z.t_g[x] + (INF, 1)[z.tx_rg ** 2 >
                        sum((z.p[x] - z.p[y])**2)]
                if alt < INF and alt < z.t_g[y]:
                    z.t_g[y] = alt
                    z.f_g[y] = x
        if (z.t_g >= INF).any():
            raise UnsufficientDensity
    def plot(z, f):
        assert len(z.p) == len(f), 'Not enough parents.'
        plt.ioff()
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(z.p[:, 0], z.p[:, 1], 'o')
        plt.hold(True)
        x = max(z.p[:, 0])
        y = max(z.p[:, 1])
        for i, c in enumerate(z.p):
            ax.text(c[0] + x * 0.02, c[1], i)
        # Plot the line between each node and its parent. 
        for u, v in enumerate(f):
            if v >= 0: # Plot only nodes with a parent
                ax.plot(z.p[[u, v], 0], z.p[[u, v], 1])
        #ax.axis([0, max(z.p[:, 0]) * 1.05, 0, max(z.p[:, 1] * 1.05)])
    def plot_glob(z):        
        '''Plot global shortest path to the sink.  '''
        z.plot(z.f_g)
        plt.title('global tree')
        plt.show()
    def plot_loc(z):        
        '''Plot global shortest path to the sink.  '''
        z.plot(z.f_l)
        plt.title('local trees')
        plt.show()
    def clusterize(z, h):
        ''' Divide the network in clusters.  
        
        Input: 
        h -- nl array such that (sum(h) = r_b-r_a), where *nl* is the number
        of layers
        
        Stamp the object with:
        lm    -- (layer mapping) nl array containing the layer of each node 
        n_int -- nl array containing the average number of hops towards the
                 gateway for nodes in each layer.
        n_ext -- nl array containing the average hop distance from the
                 cluster heads in each layer across the shortest path.
        f_l   -- n_ext hop towards the clusterhead
        ch_l -- list of nodes serving as cluster heads '''
        msg = "sum(h_i) = r_b - r_a does not hold"
        assert abs(sum(h) - z.r_b + z.r_a) / z.r_b < 1e-2, msg
        N = len(z.p)
        l = len(h) # Number of layers
        z.lm = np.zeros(N)
        z.n_int = np.zeros(l)
        z.n_ext = np.zeros(l)
        r = np.sqrt((z.p**2).sum(1))
        phi = np.arctan2(z.p[:,1],z.p[:,0])
        phi[phi<0] += 2 * np.pi 
        z.f_l = - np.ones(N, dtype='int32') # parent towards cluster head
        z.ch_l = [[] for i in xrange(l)] # cluster head list
        for i in xrange(l):
            r_int = z.r_a + sum(h[:i])
            r_ext = r_int + h[i]
            aux = (r >= r_int) * (r < r_ext)
            z.lm[aux] = i
            inc_a = float(h[i]) / r_ext
            angles = np.arange(0.0, z.phi_max, inc_a)
            for a in angles:
                # Compute the indexes of the cluster members
                ind = (aux * (phi >= a) * (phi < a + inc_a)).nonzero()[0]
                # Select the cluster member that is closest to the middle of
                # the internal boundary as the cluster head.
                ch = (z.p[ind] - r_int * np.array([np.cos(a + inc_a / 2),
                    np.sin(a + inc_a / 2)])**2).sum(1).argmin()
                z.ch_l[i].append(ind[ch])
                z.n_ext[i] += z.t_g[ch] / len(angles)
                # Compute Dijkstra's algorithm within the cluster
                dst = np.ones(len(ind)) * INF # Smallest distance to ch
                dst[ch] = 0.0 
                processed = np.zeros(len(ind), dtype=bool)
                while True:
                    x = (dst + processed * INF * 2).argmin()
                    if processed[x]: break
                    processed[x] = True
                    for y in np.r_[0:x, x + 1:len(ind)]:
                        alt = dst[x] + (INF, 1) [z.tx_rg ** 2 >
                                sum((z.p[ind[x]] - z.p[ind[y]])**2)]
                        if alt < INF and alt < dst[y]:
                            dst[y] = alt
                            z.f_l[ind[y]] = ind[x]
                z.n_int[i] += sum(dst) / sum(aux)
    def average_load(z, h, s):
        '''Return the average power consumption in each tier.

        The network should have the following attributes:

        '''
        # TODO This incomplete
        r =  s / np.sqrt(s)
        n_int, n_ext = z.clusterize(h)
        n_events = 10
        ev_loc = dist(n_events, z.r_a, wsn.r_b)
        for e in ev_loc:
            d = wsn.p - e
        return int_vol, ext_vol
    def consumption(z, h, s, w1, w2):
        # TODO The idea is to 
        int_vol, ext_vol = load(wsn, h, s)
        return w1 * int_vol, 
    def dist_m(z):
        '''Return the distance matrix for a coordinate vector.'''
        c = len(z.p)
        z.sig = np.zeros(c)
        distance_matrix = np.zeros((c, c))
        for i in xrange(c):
            for j in xrange(i + 1, c):
                p1 = z.p[i, :]
                p2 = z.p[i, :]
                distance_matrix = sum((p1 - p2) ** 2) ** 2
        return distance_matrix
    @staticmethod
    def circular_distribution(n, r_a, r_b, phi=2*np.pi):
        '''Return matrix of points uniformly distributed in ring.'''
        if r_a >= r_b:
            raise IncorrectError, "r_a must be bigger than r_b"
        r = np.sqrt((r_b**2 - r_a**2) * np.random.rand(n,1) + r_a**2)
        phi = np.random.rand(n,1) * np.pi / 2
        return np.hstack((r * np.cos(phi), r * np.sin(phi)))
class PhyNet(Tree):
    '''WSN using physical propagation model.'''
    def __init__(z, 
                 c=100, # number of nodes
                 x=200, y=200, # size of the monitored area
                 n_tries=50, # attempts to create a 90% connected network
                 d0=100, # reference distance in meters
                 PL0=1e8, # path loss at d0 in natural units
                 p_exp=3.5, # attenuation exponent
                 shadow=8, # standard deviation of the shadow fading
                 tx_p=1e-6, # transmit power
                 BW=256e3, # noise bandwidth
                 sinr=10):
        '''Generate random topology and construct routing tree.
        >>> print("{0:.3f}".format(PhyNet().tx_range()))
        99.310

        '''
        stamp(z, locals())
        z.noise = KBOL * T0 * BW
        z.att_m = np.ones((1, 1))
        z.p = np.array([[0, z.y / 2.0]])
        z.f = np.array([-1])
        z.mutate_network(1, c-1)
    def mutate_network(z, old, new):
        """Keep the first 'old' nodes and add 'new' new nodes.

        The routine keeps track of the previous state:
        - z.old is the number of nodes that where left intact
        - z.p_old is the previous vector of parents 

        """
        if old > min(len(z.p), len(z.att_m)):
            raise Error('Unsufficient past size to expand network')
            raise
        tot = old + new
        z.old = old
        z.f_old = z.f
        z.p = np.vstack((z.p[:old], np.zeros((new, 2))))
        att_m = z.att_m
        z.att_m = np.ones((tot, tot))
        z.att_m[:old, :old]  = att_m[:old, :old]
        for attempt in xrange(z.n_tries):
            z.p[old:] = np.random.rand(new, 2) * [z.x, z.y] 
            for i in xrange(tot):
                for j in xrange(max(i, old), tot):
                    d = sum((z.p[i] - z.p[j]) ** 2) ** .5
                    z.att_m[[i, j], [j, i]] = z.atten(d)
            # Cost function used in Dijkstra
            cost_m = np.where(z.tx_p / z.att_m / z.noise > z.sinr, 1, INF)
            z.f = np.array(dijkstra(cost_m))
            if sum(z.f >= 0) > 0.9 * (tot - 1):
                # The created network is sufficiently connected and we stop.
                break
        else:
            raise UnsufficientDensity
        # Neighbors within transmission range
        z.tx_l = [[j for j in xrange(tot) if j != i and z.tx_p / 
                    z.att_m[i, j] / z.noise > z.sinr] for i in xrange(tot)]
    def atten(z, d):
        '''Return attenuation in natural units.'''
        distance_atten = z.PL0 * (d / z.d0) ** z.p_exp
        shadow_fading = 10 ** (np.random.randn() * z.shadow / 10)
        return distance_atten * shadow_fading
    @property
    def tx_range(z):
        '''Return transmission range for certain transceiver parameters.
        tx_p / (d/d0) ** p_exp / noise > sinr '''
        margin = z.tx_p / z.noise / z.sinr / 10 ** (z.shadow / 10)
        return z.d0 * (margin / z.PL0) ** (1 / z.p_exp)
    def correct(z, tx_v, # indices of transmitters
                rx_v, #indices of receivers
                debug=False):
        '''Return indices of successful transmitters (tx_suc) and
        receivers (rx_suc) using the SINR model.'''
        signal = np.zeros(len(z.f))
        interference = np.zeros(len(z.f))
        for s, d in zip(tx_v, rx_v):
            for j in rx_v: # Transmit signal to all receivers
                # If the receiver is the destination, count the received
                # power as signal, otherwise count it as interference
                rec_pow = z.tx_p / z.att_m[s, j]
                if j == d:
                    signal[j] += rec_pow
                else:
                    interference[j] += rec_pow
        sinr = signal / (interference + z.noise)
        if debug:
            print(sinr)
        rx_suc = [i for i in rx_v if sinr[i] > z.sinr]
        tx_suc = [i for i, j in zip(tx_v, rx_v) if j in rx_suc]
        return tx_suc, rx_suc
    def correct_bi(z, src):
        """Return the subset of src that receive replies from their
        parents."""
        t1, r1 = z.correct(tx_v=src, rx_v=z.f[src])
        t2, r2 = z.correct(tx_v=r1, rx_v=t1)
        return r2
    def duly_scheduled_sinr(z, #
                            slot_v, # slot assignment vector
                            debug=False):
        '''Return ratio of unconnected nodes with the given transmission
        schedule using SINR model.

        The returned value is the fraction of unconnected and scheduled
        nodes.  Scheduled nodes are the nodes whose slot number is not
        zero. A node is unconnected if itself or any of its ancestors in
        the tree described by wsn.f are unduly scheduled. A node is unduly
        schedule if its transmission to its parent or the ACK in response
        fail due to insufficient SINR.
        '''
        scheduled = [i for i, j in enumerate(slot_v) if j]
        # Determine whether each node is duly scheduled by testing
        # bidirectional communication to its parent.
        unduly_scheduled = np.ones(len(z.f), bool)
        for color in xrange(1, max(slot_v) + 1):
            if debug:
                print("Color = {0}".format(color))
            src = np.array([i for i, s in enumerate(slot_v) if s == color])
            r2 = z.correct_bi(src)
            unduly_scheduled[r2] = False
        # Determine whether each node is connected, i.e., itself and all
        # its ancestors are duly scheduled.
        connected = np.zeros(len(z.f), bool)
        for i in scheduled:
            j = i
            while j:
                if unduly_scheduled[j]:
                    break
                j = z.f[j]
            else:
                connected[i] = True
        return 1 - float(sum(connected)) / len(scheduled)
class DiskModelNetwork(Tree):
    ''' WSN using transmission and interference range model.'''
    def __init__(z, c=100, x=200, y=200, tx_rg=50, ix_rg=100, n_tries=50):
        ''' Generate a topology and compute tier and TX and IX neighbors.'''
        z.c = c # cardinality = number of nodes
        z.x = x # Width of monitored area
        z.y = y # Height of monitored area
        z.tx_rg = tx_rg
        z.ix_rg = ix_rg
        for j in xrange(n_tries):
            # Location of the nodes
            z.p = np.random.rand(z.c, 2) * [z.x, z.y] 
            # The sink is located in the southwest corner
            z.p[0] = [0, y / 2] 
            # Distance matrix 
            distance_matrix = np.array([[[np.sqrt(sum((a - b)**2))] for a in
                z.p] for b in z.p])
            z.reachable = np.where(distance_matrix > tx_rg, 1, INF)
            z.f = dijkstra(z.reachable)
            if sum(z.f >= 0) > 0.9 * (z.c - 1): break
        else:
            raise UnsufficientDensity 
        # Compute neighbors within tx & ix range & nodes in the upper
        # tier.
        # Initialize list of lists of neighbors within TX range.
        z.txL = [[] for i in xrange(z.c)] 
        # Initialize list of lists neighbors within IX but TX range.
        z.ixL = [[] for i in xrange(z.c)] 
        # Initialize list of list of neighbors within TX or IX range.
        z.xL = [[] for i in xrange(z.c)] 
        # Initialize list of lists of neighbors in the next tier.
        z.up = [[] for i in xrange(z.c)] 
        for i, p1 in enumerate(z.p):
            for j, p2 in enumerate(z.p):
                if j == i: 
                    continue
                distance_matrix = sum((p1 - p2) ** 2) ** .5
                if distance_matrix > ix_rg: 
                    continue
                z.xL[i].append(j)
                if distance_matrix > tx_rg:
                    z.ixL[i].append(j)
                    continue
                z.txL[i].append(j)
                if z.tier(j) == z.tier(i) - 1:
                    z.up[i].append(j)
    def correct_disk(z, slot_v):
        '''Return unconnectivity ratio of schedule according to disk graph.

        fraction of unthe nodes that fail to reach the gateway in disk
        model

        DEPRECATED
        Network parameters
            f     --- parents vector
            xL --- dictionary of lists. List xL[i] contains the list of
            nodes that communicate or interfere with node i.  

        slot_v --- a slot assignment

        Determine which nodes have been assigned an infeasible slot (i.e.,
        they have been assigned the same slot than other nodes that
        interfere with them). This determination is carried out using
        interference range model.

        A node is considered to be disconnected if it or any of its
        ancestors has been assigned an unfeasible slot.'''
        # Using f and slot_v, compute the receiving slots of every node. I
        # use a dictionary of dictionaries to store the results.
        rx_slot = collections.defaultdict(dict)
        colored = [i for i, j in enumerate(slot_v) if j]
        for i in colored:
            rx_slot[z.f[i]][slot_v[i]] = i
        # Check whether each node is in a collision
        collided = np.zeros(len(z.f), bool)
        # Interfere all the recipients in the TX slot
        for i in colored:
            c = slot_v[i]
            for j in z.xL[i]:
                if j == z.f[i]: continue
                if c in rx_slot[j]:
                    collided[rx_slot[j][c]] = True
        # Interfere all the sources in the TX slot
        for i in xrange(len(z.f)):
            for k, h in rx_slot[i].iteritems():
                if collided[h]:
                    continue # I failed to receive packet
                for n in z.xL[i]:
                    if z.f[n] == i: # Do not ruin my child's transmission
                        continue
                    if slot_v[n] == k:
                        collided[n] = True
        connected = np.zeros(len(z.f), bool)
        for i in colored:
            j = i
            while j:
                if collided[j]:
                    break
                j = z.f[j]
            else:
                connected[i] = True
        return 1 - float(sum(connected)) / len(colored)
class Packet(object):
    def __init__(z, r, d=None, p=None, s=-1, f=0, n=0):
        # Relay of the packet.  In the case of FlexiTP, is the node for
        # whom the slot is claimed (the node that actually requests the
        # slot is its parent).
        z.r = r 
        z.d = d # recipient of the packet
        z.p = p # packet type
        z.s = s # source of the information
        z.f = f # number of times to be forwarded
        z.n = n # number of the claimed slot
class Packet1(object):
    def __init__(z, ninter, ncont):
        """
        
        ninter -- is the number of the interval for which the packet was
        generated.
        
        ncont  -- number of sources that contributed to the packet
        
        """
        z.ninter = ninter
        z.ncont = ncont
class Node(simpy.Process):
    '''Base class for a node in a slotted system.'''
    def __init__(z, id, sim):
        simpy.Process.__init__(z, name='Node '+ str(id), sim=sim)
        z.id = id
        z.tx_p = z.wsn.tx_p # Transmit power in dBm
        z.noise = z.wsn.noise # Noise level in dBm
        z.sinr = z.wsn.sinr # in dB
        z.en = 0 # consumed energy in J
        z.tx_d = {} # {slot: source}
        z.rx_d = {} # {slot: (relay, source)}
        z.ct_d = {} # {slot: (relay, source)} slots used by other nodes
        z.sc_d = {} # {fn: node}: when to schedule child node
        z.packet_l = [] # (relay, source) to be scheduled ASAP in any frame
    def __getattribute__(z, name):
        if name == 'f':
            return z.wsn.f[z.id]
        if name == 'tier':
            return z.wsn.tier(z.id)
        if name == 'children':
            return [n.id for n in z.sim if n.f == z.id]
        try:
            return object.__getattribute__(z, name)
        except AttributeError:
            return object.__getattribute__(z.sim, name)
    def set_radio(z, new_state): 
        '''Set new radio state keeping track of the consumed energy.'''
        assert new_state in STATES
        z.en += ((z.now() - z._last_radio_change) * STATES[z.radio])
        z.radio= new_state
        z._last_radio_change = z.now()
    def n_descendants(z):
        x = len(z.children)
        for y in z.children:
            x += z.sim[y].n_descendants()
        return x
    def __repr__(z):
        return  'SimPy node with ID {0}'.format(z.id)
    def broadcast(z, pkt, t):
        '''Place packet in the input buffer of all other nodes.  
        Before adding the packet to a node, stamp the packet with a field
        'pow' that indicates the power with which that node receives it.'''
        z.set_radio('sl')
        yield simpy.hold, z, z.Q
        for i, n in enumerate(z.sim):
            if (n is not z and hasattr(n, 'radio') 
                    and n.radio not in ['sl', 'tx']): 
                if n.radio == 'id':
                    n.set_radio('rx')
                pkt_i = copy.deepcopy(pkt)
                pkt_i.pow = z.tx_p / z.wsn.att_m[z.id, i]
                n.i.append(pkt_i)   
        # Setting transmit state ensures it will not be interrupted
        z.set_radio('tx')
        z.print("transmits {0}".format(pkt.p))
        actual_tx_time = t - z.Q
        assert actual_tx_time > 0      
        yield simpy.hold, z, actual_tx_time
    def bo(z):
        # if z.id == 8 and z.now() > 185.5:
        #     pdb.set_trace()
        bt = z.bt()
        z.print("will back off until {0:12.7f}".format(z.now() + bt))
        yield z.listen(bt)
        z.process_reception_buffer()
    def correct_rec(z, p):
        '''Return whether I received a packet of type p_type.'''
        z.process_reception_buffer()
        if z.i.p == p and z.i.d == z.id:
            z.print("received a {0} packet from {1}. :)".format(p, z.i.r))
            return True
        else:
            z.print("did not receive a {0} packet. :(".format(p))
            return False 
    def listen(z, time):
        z.i = []
        z.set_radio('id')
        return simpy.hold, z, time
    def print(z, str):
        if z.VB: print("{0:12.7f} Node {1} {2}".format(z.now(), z.id, str))
    def process_reception_buffer(z):
        '''Place in z.i the received packet if SINR exceeds threshold. The
        output is the received packet or None depending on the success o the
        operation. The output is both returned and stored in z.i.'''
        if z.i:
            pow = [p.pow for p in z.i]
            total_power = sum(pow)
            i = np.argmax(pow)
            sig = pow.pop(i)
            sinr = sig / (sum(pow) + z.noise)
            z.print("SINR= {0}".format(sinr))
            if sinr > z.sinr:
                z.i = z.i[i]
            elif total_power / z.noise > z.sinr:
                z.i = Packet(r=None, d=None, p='ruined_reception')
            else:
                z.i = Packet(r=None, d=None, p='nothing_received')
        else:
            z.i = Packet(r=None, d=None, p='nothing_received')
    def sleep(z, time):
        z.set_radio('sl')
        return simpy.hold, z, time
    def tx_pkt(z, t, **keywords):
        '''Send packet to all active nodes.'''
        pkt = Packet(r=z.id, **keywords)
        for i in z.broadcast(pkt, t): yield i
class FlexiTPNode(Node):
    bt = lambda z: np.random.rand() * (z.t_w - z.Q)
    def run_su(z): 
        """Simplified algorithm for both the scheduling and transmission
        phase. 

        The difference between both phases is the number of nodes seeking a
        slot at the beginning of each frame. This number is one in the setup
        phase, but it may be larger in the transmission phase.  This is
        controlled with the variable z.sc_d, which is stamped in the
        initialization of FlexiTPNet.

        A node X claims a slot for its child, and I enforce the reception by
        X's child and X's parent by making X make the processing for those
        two nodes: X adds the claimed slot in its child's tx_d and in its
        parent sc_d.

        In my implementation, the length of the FTS (z.nexch) should be set
        to a sufficiently large value.  The variable z.sim.nadv1 keeps track
        of the minimum number that we need in order to make the protocol
        work as if z.nexch were infinite. 

        z.packet_l: list (rel, src), where "rel" is the node that contends
        to obtain a packet and "src" is the source of the packet for whom
        the slot is claimed.  For simplicity, instead of "rel", its parent
        contends and identifies itself as "rel".

        -------------------------
        Output of the routine (it is not returned, it is stamped)

        + z.sim.adve is a measure of the latency, but a bad one because
        since each node only claims a slot at the beginning.

        + z.sim.record['laten']: I increment each time that I try to claim a
        slot in the current frame but I fail.

        """
        z.print('starts being active')
        for z.frame_n in xrange(1, z.nm + 1):
            to_advertise_l = [] # packets to be relayed in current frame
            advertised_l = [] # relays whose victory was advertised.  
            #
            # In the case that z is in the initialization phase, check
            # whether z should contend in the current frame.
            if z.frame_n in z.sc_d:
                node_id = z.sc_d.pop(z.frame_n)
                z.print('will contend for its child {0} in frame {1}===='
                        .format(node_id, z.frame_n))
                z.packet_l.append((node_id, node_id))
            z.sim.record['laten'] += len(z.packet_l)
            for rel, src in z.packet_l:
                z.sim.record['lates'] += z.sim[rel].tier
            for n_exch in xrange(z.n_exch):
                ce = z.now() + z.t_w  # end of contention time in this slot
                if n_exch == 0 and z.packet_l:
                    # Contend to claim a slot
                    for i in z.bo(): yield i
                    if z.i.p == 'nothing_received':
                        # z won the contention to claim a slot
                        for i in z.tx_pkt(p='pre', t=ce-z.now()): yield i
                        # Find the next available slot
                        for slot in xrange(1, z.nm + 1):
                            if (slot not in z.tx_d and slot not in
                                z.rx_d and slot not in z.ct_d):
                               break
                        else:
                            raise Error('No available slots remain')
                        rel, src = z.packet_l.pop(0)
                        pkt = Packet(p='claim',r=rel,s=src,n=slot,f=z.fw-1)
                        z.print('Transmits advertisement')
                        for i in z.broadcast(pkt=pkt, t=z.slot_t): yield i
                        z.rx_d[slot] = rel, src
                        z.sim[rel].tx_d[slot] = src
                        advertised_l.append(rel)
                        if z.f >= 0:
                            z.sim[z.f].packet_l.append((z.id, src))
                        z.print('====assigned slot {0} to node {1}'.format(
                                slot, src))
                        continue
                elif to_advertise_l: 
                    """I have to relay schedule updtaes from other nodes."""
                    # I update nadve (the maximum number of slots used per
                    # frame) if the current value of slot (n_exch) is bigger
                    # than the current maximum.
                    z.sim.nadve = max(n_exch, z.sim.nadve)
                    # Try to advertise the schedule
                    for i in z.bo(): yield i
                    if z.i.p == 'nothing_received':
                        for i in z.tx_pkt(p='pre', t=ce-z.now()): yield i
                        z.print('relays schedule information')
                        pkt = to_advertise_l.pop()
                        advertised_l.append(pkt.r)
                        for i in z.broadcast(pkt=pkt, t=z.slot_t): 
                            yield i
                        z.print('ends relay')
                        continue
                yield z.sleep(ce-z.now())
                # Listen 
                yield z.listen(z.slot_t)
                z.process_reception_buffer()
                if z.i.p == 'claim':
                    z.print("added item to to_advertise_l")
                    z.ct_d[z.i.n] = (z.i.r, z.i.s) 
                    if z.i.f >= 1 and z.i.r not in advertised_l:
                        p = Packet(p='claim', r=z.i.r, s=z.i.s, f=z.i.f-1, 
                                   n=z.i.n)
                        to_advertise_l.append(p)
            # Remove unsuitably scheduled nodes.
            if to_advertise_l:
                raise AdvertTError
            for node in z.sim:
                if node.packet_l or hasattr(node, 'sc_d') and node.sc_d:
                    break
            else:
                z.print("===END OF SIMULATION===")
                z.sim.stopSimulation()
            yield z.sleep(z.pause)
        raise Error('Should not reach this point')
class RandSchedNode(Node):
    '''Node executing RandSched.'''
    bt = lambda z: np.random.rand() * (z.t_w - z.Q)
    def run_su(z):
        '''Run RandSched's distributed scheduling routine.'''
        # z.b contains z.id if the node should find a slot for itself.
        z.frame_n = 0
        z.b = [z.id] if z.id else [] # sources to be scheduled
        while len(z.rx_d) < len(z.children): 
            z.print("starts serving")
            z.update()
            z.print("starts checking the channel") 
            yield z.listen(z.t_w) 
            z.process_reception_buffer() 
            if z.i.p == 'nothing_received': 
                yield z.sleep(z.fe - z.now()) 
            else:
                z.print("commits to serve")
                for i in z.serve(): yield i
            yield z.sleep(z.pause)
        if z.f >= 0:
            while not z.tx_d:
                z.update()
                for i in z.bo(): yield i
                if  z.i.p == 'nothing_received': 
                    for i in z.seek(): yield i
                else:
                    z.print("lost the contention")
                    yield z.sleep(z.fe - z.now())
                yield z.sleep(z.pause)
    def update(z):
        z.frame_n += 1 # frame number
        z.fs = z.now()  # frame start time
        z.fe = z.fs + z.ft # Frame end time.
        if z.frame_n > z.sim.last_successful_contention_frame + 10:
            z.sim.stopSimulation() 
            raise NoProgressError
    def seek(z):
        for i in z.tx_pkt(d=z.f, p='pre', t=z.fs+z.t_w-z.now()): yield i
        # sleep for an even random number of slots
        sn = np.random.randint(z.pairs) 
        yield z.sleep(2 * sn * z.slot_t)
        for i in z.tx_pkt(d=z.f, p='cont', t=z.slot_t): yield i
        yield z.listen(z.slot_t)
        if z.correct_rec(p='ack'):
            # Protect my frame by transmitting in remaining slots.
            for i in xrange(sn + 1, z.pairs):
                for i in z.tx_pkt(t=z.slot_t, d=z.f, p='cont'): yield i
                yield z.sleep(z.slot_t)
            # Final contention against all the winners.
            for i in z.tx_pkt(t=z.slot_t, d=z.f, p='cont'): yield i
            yield z.listen(z.slot_t)
            if z.correct_rec(p='ack'):
                z.tx_d[z.frame_n] = z.b[0]
                for i in z.tx_pkt(d=z.f,p='cont',t=z.slot_t,s=z.b.pop(0)):
                    yield i
        yield z.sleep(z.fe - z.now())
    def serve(z):
        s = 0 
        for i in xrange(z.pairs):
            if s:
                yield z.sleep(z.slot_t)
                for j in z.tx_pkt(d=s, p='ack', t=z.slot_t): yield j
            else: 
                yield z.listen(z.slot_t)
                if z.correct_rec(p='cont'):
                    s = z.i.r
                    for j in z.tx_pkt(d=s, p='ack', t=z.slot_t): yield j
                else:
                    yield z.sleep(z.slot_t)
        # Listen again for packets. Now only winners will transmit.
        yield z.listen(z.slot_t)
        if z.correct_rec(p='cont'):
            for i in z.tx_pkt(d=z.i.r, p='ack', t=z.slot_t): yield i
            yield z.listen(z.slot_t)
            if z.correct_rec(p='cont'):
                # Assign slot to child
                z.rx_d[z.frame_n] = (z.i.r, z.i.s)
                if z.id != 0:
                    z.b.append(z.i.s)
                z.sim.last_successful_contention_frame = z.frame_n
                z.print('assigned slot {0} to child {1}'.format(
                        z.frame_n, z.i.r))
                if z.VB: 
                    print("=" * 30)
            else:
                z.print("failed to assign slot in the last test")
        yield z.sleep(z.fe - z.now())
class ACSPNode(RandSchedNode):
    def help_children(z):
        return len(z.rx_d) < z.n_descendants() and (z.id==0 or len(z.b)<z.B)
        # 1 <= len(z.b) <= z.B in my simulations
    def bt2(z):
        """Previous back off algorithm.  Deprecated."""
        return (z.B - len(z.b) + np.random.rand()) * (z.t_w-z.Q) / z.B
    def bt(z):
        """Back off time prioritizing hop distance and then buffer
        capacity.  
        
        z.b > 0 (otherwise it would not contend)
        z.tier > 0 (otherwise it would not contend)
        
        """
        block = min(z.tier - 1, z.tiermax)
        assert block >= 0
        slot = block * z.B + z.B - min(len(z.b), z.B) + np.random.rand()
        t = slot / z.B / z.tiermax * (z.t_w - z.Q)
        assert t < z.t_w - z.Q
        return t
    def run_su(z):
        """Run the initial setup algorithm."""
        z.frame_n = 0
        z.b = [z.id] if z.id else [] # packets to be scheduled
        while z.b or z.help_children():
            if z.id == 0: 
                z.print('begins frame {0}{1}'.format(z.n_frames(),15 * '-'))
            z.update()
            z.assert_sync()
            if z.b:
                for i in z.bo(): yield i
                if z.i.p == 'nothing_received':
                    for i in z.seek(): yield i
                elif z.help_children():
                    yield z.sleep(z.fs + z.t_w - z.now())
                    for i in z.serve(): yield i
                else: 
                    z.print("lost will not help")
                    yield z.sleep(z.fe - z.now())
            else:
                z.print("starts checking the channel") 
                yield z.listen(z.t_w) 
                z.process_reception_buffer() 
                if z.i.p == 'nothing_received': 
                    yield z.sleep(z.fe - z.now()) 
                else:
                    for i in z.serve(): yield i
    def assert_sync(z, offset=0.0, tolerance=0.01):
        t = float(z.now() - offset)
        mult = np.round(t / z.ft) 
        error = abs(t - z.ft * mult)
        if error > z.ft * tolerance:
            raise OutOfSyncError
    def serve2(z, active_in_all, test_slot=-1):
        """Transmit, receive, and attempt to gain slots.

        This is a helper routine of "z.run_tx()"
        
        active_in_all -- remain active in all slots? 
        test_slot     -- slot in which to attempt to transmit
        """
        z.assert_sync(offset=z.t_w)
        z.SDS = False
        for current_slot in xrange(z.max_slots):
            if z.id == 0: 
                z.print('Frame %d, DS %d' %(z.n_frames(), current_slot))
            if current_slot == test_slot:
                for i in z.tx_pkt(d=z.f, p='incorp', t=z.slot_t): yield i
                yield z.listen(z.slot_t)
                z.succeeded = z.correct_rec(p='ack')
            elif current_slot in z.tx_d: 
                for i in z.tx_pkt(d=z.f, p='data', t=z.slot_t): yield i
                yield z.listen(z.slot_t)
                if not z.correct_rec(p='ack'):
                    z.sim.record['losse'] += 1
                    z.SDS = True
                    z.print("will raise SDS^^^")
            elif current_slot in z.rx_d:
                yield z.listen(z.slot_t)
                if z.correct_rec(p='data'):
                    for i in z.tx_pkt(d=z.i.r, p='ack', t=z.slot_t): yield i
                else:
                    for i in z.tx_pkt(d=-1, p='ack', t=z.slot_t): yield i
                    z.SDS = True
                    z.print("failed to receive a packet from {0} in slot {1}"
                            .format(z.rx_d[current_slot], current_slot))
                    z.print("will raise SDS^^^")
            elif active_in_all:
                yield z.listen(z.slot_t)
                # if z.now() > 246.986041 and z.id == 19:
                #     pdb.set_trace()
                if z.correct_rec(p='incorp'):
                    for i in z.tx_pkt(d=z.i.r, p='ack', t=z.slot_t): yield i
                else:
                    yield z.sleep(z.slot_t)
            else:
                yield z.sleep(2 * z.slot_t)
        for tone in xrange(z.nsds):
            if z.SDS:
                z.print('Broadcasts SDS signal in slot {0}'.format(tone))
                for i in z.tx_pkt(p='SDS', t=z.sglt): yield i
            else:
                yield z.listen(z.sglt)
                z.process_reception_buffer()
                if z.i.p != 'nothing_received':
                    z.print('Received SDS signal in slot {0}'.format(tone))
                    z.SDS = True
        if test_slot >= 0:
            if z.succeeded and not z.SDS:
                src = z.b.pop(0)
                z.print('won slot {0} for node {1}'.format(test_slot, src))
                if z.f > 0:
                    z.sim[z.f].b.append(src)
                    z.sim[z.f].target_slot = test_slot + 1
                assert test_slot not in z.sim[z.f].rx_d
                z.sim[z.f].rx_d[test_slot] = (z.id, src)
                z.tx_d[test_slot] = src
                z.status = 0
            else:
                z.stamp_alert_free_counter(0)
        yield z.sleep(z.pause / 2.0)
        # Remove unsuitably scheduled nodes.
        for slot, source in copy.copy(z.tx_d).iteritems():
            concurrent = [i for i, j in enumerate(z.sim) if slot in j.tx_d]
            if z.id not in z.wsn.correct_bi(concurrent):
                z.print('was expelled in slot %d.  The pkt source is %d'
                        %(slot, source))
                z.sim.record['expe1'] += 1
                z.sim.record['expe2'] -= 1 # avoid double counting
                node = z
                while node.id > 0:
                    for slt2, src2 in copy.copy(node.tx_d).iteritems():
                        if src2 == source:
                            z.sim.record['expe2'] += 1
                            del node.tx_d[slt2]
                            del z.sim[node.f].rx_d[slt2]
                    if source in node.b:
                        node.b.remove(source)
                    node = z.sim[node.f]
                z.b.append(source)
                z.target_slot += np.random.randint(6)
        yield z.sleep(z.pause / 2.0)
    def stamp_alert_free_counter(z, longbo): 
        """Stamp myself with a counter with the number of TFs without
        collision alerts that I have to wait before contending for a packet.
        
        Parameters:
        longbo --  0 or 1 
        """
        z.alert_free_counter = 0 # count of frames without SDS
        z.alert_free_backoff =  (np.random.randint(z.mult) 
                     + max(0, z.cap - len(z.b)) * z.mult
                     + longbo * z.cap * z.mult)
        assert z.alert_free_backoff >= 0, 'Incorrect backoff'
        z.print('to wait {0} consecutive frames without SDS'.
                format(z.alert_free_backoff))
    def run_tx(z):
        """Run the data transmission phase of an ACSPNode.

        z.b contains the list of nodes for which I have to claim a
        slot.  Initially z.b contains only my identity because the
        execution of RandSchedNet.init_adaptation_statistics() removes
        the slots of the full path from the data sources to the data
        sink if any of the nodes of that path belongs to the nodes
        that were just added.

        """
        z.b = [z.id] if z.f >=0 and z.id not in z.tx_d.values() else []
        z.target_slot = 0 # Next slot in which to attempt incorporation
        while True:
            while not z.b: # I don't have any slot to claim
                if z.id == 0: 
                    z.print('begins frame {0}{1}'.format(
                            z.n_frames(),15 * '-'))
                # Serve the neighbors if necessary.
                z.print('listens in case it is needed')
                yield z.listen(z.t_w)
                z.process_reception_buffer()
                if z.i.p != 'nothing_received':
                    z.sim.record['activ'] += 1
                    for i in z.serve2(1): yield i
                else:
                    for i in z.serve2(0): yield i
            z.status = 0  
            # z.status is a variable that controls the status in the
            # contention process.  z.status=0 means that the node has not
            # gained the right to use the preamble yet, and z.status=1 means
            # that it has already done it.
            while z.b: # I want to claim some slots
                z.sim.record['laten'] += len(z.b)
                for sourcu in z.b:
                    z.sim.record['lates'] += z.sim[sourcu].tier
                if z.status == 0:
                    z.print('starts contending in the window')
                    ce = z.now() + z.t_w
                    for i in z.bo(): yield i
                    if z.i.p == 'nothing_received':
                        for i in z.tx_pkt(p='pre', t=ce-z.now()): yield i
                        z.stamp_alert_free_counter(0)
                        z.status = 1
                    else:
                        yield z.sleep(ce-z.now())
                        for i in z.serve2(True): yield i
                        continue
                else:
                    for i in z.tx_pkt(p='pre', t=z.t_w): yield i
                if z.alert_free_counter < z.alert_free_backoff:
                    for i in z.serve2(True): yield i
                    if z.SDS:
                        z.stamp_alert_free_counter(1)
                    else:
                        z.alert_free_counter += 1
                    continue
                # Find the target DS
                for z.target_slot in xrange(z.target_slot, z.max_slots):
                    # Increase the variable sensea that keeps track of the
                    # number of channel check operations.
                    z.sim.record['sensea'] += 1
                    conc1 = [node for node in z.sim 
                             if z.target_slot in node.tx_d]
                    conc2 = [z.sim[node.f] for node in conc1]
                    max_sig  = 0.0
                    for sources in conc1, conc2:
                        sig = sum(node.tx_p / z.wsn.att_m[node.id, z.id] 
                                  for node in sources)
                        max_sig = max(max_sig, sig)
                    if max_sig / z.wsn.noise < z.wsn.sinr:
                        break
                else:
                    raise Error('No more slots available')
                z.print('will contend in slot {0}'.format(z.target_slot))
                # Attempt in the selected slot
                for i in z.serve2(True, z.target_slot): yield i
                z.sim.record['attem'] += 1
                if not sum(len(n.b) for n in z.sim if hasattr(n, 'b')):
                    z.print("===END OF SIMIULATION===")
                    z.sim.stopSimulation()
                    break
                z.target_slot += 1
class SimNet(list, simpy.Simulation):
    def __nonzero__(z): return True
    def simulate_net(z, routine='run_su', *args, **kwargs):
        z.print("Network tree = {0}".format(z.wsn.f))
        simpy.Simulation.initialize(z)
        for node in z:
            if node.id == 0 or node.f >= 0:
                node._last_radio_change = 0 
                node.radio = 'sl' #radio state
                simpy.Process.__init__(node, name='Node ' + str(id), sim=z)
                z.activate(node, getattr(node, routine) (*args, **kwargs),
                           prior=not node.id)
        simpy.Simulation.simulate(z, until=z.until)
    def print(z, *args, **kwargs):
        if z.VB:
            print(*args, **kwargs)
class RandSchedNet(SimNet):
    AlNode = RandSchedNode
    def __init__(z, wsn, cont_f=10, pairs=2, Q=0.1, slot_t=2, pause=10.0, 
                 VB=False, until=1e8, **kwargs):
        """
        >>> RandSchedNet(test_net1(), VB=False, until=15).schedule()
        [-1, 1]
        >>> RandSchedNet(test_net2(), VB=False, until=100).schedule()
        [-1, 2, 1]
        """
        t_w = cont_f * Q # Contention window duration
        ft = t_w + slot_t * (pairs * 2 + 3) + pause # frame duration
        stamp(z, locals())
        z.print("t_w = {0}; ft = {1}".format(t_w, ft))
        z[:] = [z.AlNode(id=i, sim=z) for i in xrange(len(wsn.f))]
        z.last_successful_contention_frame = -1
        z.simulate_net()
    def schedule(z):
        """Return schedule in a vector."""
        slots = []
        for n in z:
            if n.id == 0 or n.f == -1:
                slots.append(-1)
            elif len(n.tx_d) == 0:
                raise IncompleteAggregationError
            elif len(n.tx_d) == 1:
                slots.append(n.tx_d.keys()[0])
            else:
                raise Error, 'Incorrect schedule'
        return slots
class ACSPNet(RandSchedNet):
    """Simulate the ACSP algorithm.

    >>> ac = ACSPNet(test_net1(), VB=False, until=50)
    >>> ACSPNet(test_net3(), VB=False, until=180).print_dicts()
    ========================================
    node 0
    tx_d = {}
    rx_d = {1: (2, 2), 4: (2, 1), 6: (2, 3)}
    ========================================
    node 1
    tx_d = {2: 1, 5: 3}
    rx_d = {3: (3, 3)}
    ========================================
    node 2
    tx_d = {1: 2, 4: 1, 6: 3}
    rx_d = {2: (1, 1), 5: (1, 3)}
    ========================================
    node 3
    tx_d = {3: 3}
    rx_d = {}
    """
    AlNode = ACSPNode
    def __init__(z, wsn, B=5, tiermax=20, **kwargs):
        """B is the maximum size of each node's z.b list containing the
        nodes pending to be scheduled.
        
        tiermax   -- maximum number of tiers excluding the sink
        
        """
        RandSchedNet.__init__(z, wsn, B=B, tiermax=tiermax, **kwargs) 
        z.complete_convergecast()
    def remove_info(z, i, j):
        """Remove all information about node j from node i (ids)."""
        for slot, src in copy.copy(z[i].tx_d).iteritems():
            if src == j: 
                del z[i].tx_d[slot]
        for slot, (r, src) in copy.copy(z[i].rx_d).iteritems():
            if src == j: 
                del z[i].rx_d[slot]
        # I decided not to remove the contention information.
        #if hasattr(z[i], 'ct_d'):
        #    for slot, (r, src) in copy.copy(z[i].ct_d).iteritems():
        #        if src == j: 
        #            del z[i].ct_d[slot]
        if hasattr(z[i], 'b') and j in z[i].b:
            z[i].b.remove(j)
    def n_slots(z):
        return max(z[0].rx_d.keys())
    def purge_old_slots(z):
        """Remove outdated slots from the schedule of all the nodes in the
        network.  This routine is used after making topology changes.

        If the ancestor Y of a node X was replaced, remove all the slots
        allocated to relay X's data over the tree.
        """
        z[z.wsn.old:] = [z.AlNode(id=i, sim=z)
                         for i in xrange(z.wsn.old, len(z.wsn.f))]
        # Remove information about new nodes from the old nodes
        for i in xrange(z.wsn.old):
            for j in xrange(z.wsn.old, len(z.wsn.f)):
                z.remove_info(i, j)
        for node_id in xrange(len(z.wsn.f_old)):
            # Skip to the next iteration if the ancestor have not changed.
            i = node_id
            while i >= 0:
                if z.wsn.f[i] != z.wsn.f_old[i] or z.wsn.f[i] >= z.wsn.old:
                    # Remove all the information from node_id from its
                    # ancestors, but keep the contention information.
                    id = node_id
                    while id >= 0:
                        # Without the following the condition, my code would
                        # raise an exception if the number of nodes in the
                        # network changed.
                        if id < z.wsn.old: 
                            z.remove_info(id, node_id)
                        id = z.wsn.f_old[id]
                    break
                i = z.wsn.f[i]
    def n_frames(z):
        return int(np.ceil((z.now() - z.ft / 2) / z.ft))
    def adjust_schedule_in_tx_phase(z, nsds=2, sglt=1, cap=1, mult=2,
                                    max_slots=None, pause=3, until=None):
        """
        Execute the data transmission phase of ACSP.

        This routine executes sufficient TFs until every node has obtained
        all the slots that it needs in order to adapt to the topology
        changes.
        
        Keyword arguments:

        nsds: number of stop disturbance tones 
        sglt: signal time 
        cap: number of priorities in the backoff period 
        mult:
        max_slots: number of slots per frame
        pause: idle period at the end of each frame 
        until

        Criticism
        ----------------------

        This routine considers that a node decides whether a child gained
        a slot in just one frame, whereas in practice it takes two frames.
        
        Possible enhancements to be included later. Introduce the variables:
        nswi=1,        # number of slots with interference
        nswo=1,        # number of slots without interference
        
        The resulting frame period is:

        ft = z.t_w + (nswi + nswo) * z.slot_t * z.max_slots + nsds * sglt
        
        """
        if until is None:
            until = z.until
        if max_slots is None:
            max_slots = 10 * len(z.wsn.f)
        ft = z.t_w + 2 * z.slot_t * max_slots + nsds * sglt + pause
        print("Frame time = {0}s".format(ft))
        stamp(z, locals())
        z.purge_old_slots()
        if z.VB:
            print("======Schedule before the update is executed======")
            z.print_dicts()
        z.init_adaptation_statistics()
        z.simulate_net(routine='run_tx')
        z.print("{0} frames ellapsed".format(z.n_frames()))
        z.tot_repair = (z.natre + z.record['expe1'] +
                z.record['expe2'])
        z.mult_fact = z.tot_repair / z.natre 
        # Normalize parameters in z.record
        for k, v in z.record.iteritems():
            z.record[k] = v / z.natre
            if z.VB:
                print("Record:{0} = {1}".format(k, z.record[k]))
        if z.VB: 
            print("======Schedule after the execution of the update=====")
            z.print_dicts()
    def init_adaptation_statistics(z):
        """Compute some statistics about the adaptation process. """
        # Number of naturally required slots (slots that have to be
        # obtained because of the network mutation, excluding those
        # that have to be obtained because of expulsions).
        z.natre = float(sum(n.tier for n in z if n.id not in 
                            n.tx_d.values()))
        z.record = dict(
            laten  = 0., # Acquisition latency: time to obtain a slot
            lates  = 0., # global latency, or postponement variable.  If a
                         # node is 2 hops away from the data sink, each
                         # acquisition takes 2 slots, and they occur
                         # sequentially, lates= 2*2+2 = 6
            activ  = 0., # Time that a node listens immediately after the
                         # contention window for a packet indicating which
                         # node should be active and in which slot.
            sensea = 0., # number of channel check measurements by an
                         # aspirant trying to decide the slot that it will
                         # seek to gain until it finds a slot in which there
                         # seem to be no transmissions
            attem = 0.,  # attempts in which the nodes actually tested
            losse = 0., # Number of packets lost by the nodes that already
                        # possessed a slot because of the interference of
                        # the nodes seeking to obtain a slot.
            expe1 = 0., # Number of nodes that possessed a slot but were
                        # forced to find a new one because other nodes
                        # generating intolerable interference decided to use
                        # the slot.
            expe2 = 0.) # Number of times a node had to find a new
                        # transmission slot because some of its descendants
                        # were expelled
        z.print('*****Start computing the schedule*******')
    def incorp_ener(z):
        """Return energy consumed per incorporation in ACSP.

        This only includes the variable energy.  That is, it does not
        include the amount of energy.
        
        """
        tx = STATES['tx']
        rx = STATES['rx']
        ener = (
          + z.t_w / 2 * z.record['laten'] * tx # contention window  
          + z.slot_t  * z.record['sensea'] * rx # find available slot
          + z.slot_t  * z.record['attem'] * (tx+rx) 
          #
          # recever side
          + z.slot_t  * z.record['activ'] * rx # Listen to the slot
                                               # indicating whether it
                                               # should listen.
          + z.slot_t  * z.record['attem'] * tx # Reply that is ready to
                                               # receive
          + z.slot_t  * z.record['attem'] * (rx + tx) # Listen and send ACK
          # Nodes unduly dismissed
          + z.slot_t * z.record['losse'] * (rx + tx) * 2
            ) * z.mult_fact
        return ener
    def run_tx(z, pe):
        """ Execute transmit cycles.

        Arguments:
        - pe: Packet error probability
        
        Return:
        - M_v: length of the schedule for each frame number.
          The length of this parameters is also the convergence time.
        - G_v: number of unduly given up nodes
        - l_v: average node incorporation time """
        pass
    def print_dicts(z):
        for i, j in enumerate(z):
            print(40 * '=')
            print("node {0}".format(i))
            for s in ('tx_d', 'rx_d', 'ct_d'):
                if hasattr(j, s):
                    print("{0} = {1}".format(s, getattr(j, s)))
    def fix(z):
        """Artificially fix the unconnected nodes.

        This routine is used to:
        - Fix unduly scheduled nodes with FlexiTP.
        - Fix nodes accidentally expelled with ACSP.

        """
        for node, slot in z.unconnected():
            src = node.tx_d.pop(slot)
            while True:
                old = [i for i, j in enumerate(z) if slot in j.tx_d]
                old_n = len(z.correct_bi(old))
                new = old.append(node.id)
                new_n = len(z.correct_bi(new))
                if new_n > old_n:
                    node[slot] = src
    def ratio_unconn(z):
        """Return the fraction of nodes that cannot reach the sink."""
        nodes = [node.id for node in z if node.f >=0]
        tot = len(nodes)
        for node, slot in z.unconnected():
            src = node.tx_d[slot]
            if src in nodes:
                nodes.remove(src)
        return tot / len(nodes)
    def dismissed(z):
        """Return fraction of unduly dismissed parents."""
        return float(len(z.unconnected())) / sum(len(n.tx_d) for n in z)
    def unconnected(z):
        """Return [(node, slot_nr)] of invalid schedules."""
        unconnected_l = []
        last_used = -1
        for slot in xrange(9999):
            src = [i for i, j in enumerate(z) if slot in j.tx_d]
            if src:
                last_used = slot
                suc = z.wsn.correct_bi(src)
                unsuc = set(src).difference(suc)
                unconnected_l.extend((z[i], slot) for i in unsuc)
            elif slot - last_used > 15:
                break
        else:
            raise Error('We should not reach this point')
        return unconnected_l
    def expe12(z):
        """Return number of expelled nodes type 1 and type 2.
        
        Type one are those nodes that have an infeasible slot.

        Type two are the those nodes that lie before the tree.
        
        """
        expe1 = 0
        expe2 = 0
        last_used = -1
        for slot in xrange(9999):
            src = [i for i, j in enumerate(z) if slot in j.tx_d]
            if src:
                last_used = slot
                suc = z.wsn.correct_bi(src)
                unsuc = set(src).difference(suc)
                for node in unsuc:
                    expe1 += 1
                    expe2 += z[node].tier - 1
            elif slot - last_used > 15:
                break
        else:
            raise Error('We should not reach this point')
        return expe1, expe2
    def complete_convergecast(z):
        """Raise an error if schedule is incomplete.  

        The schedule is for uncompressed data gathering, which means that
        the number of transmission slots of every node should be its number
        of descendants plus one.
        
        If the caller intentionally did not want to give enough
        time (z.until is small), then nothing else is executed.
        
        """
        if z.until < 1e4:
            return
        to_schedule = [n.id for n in z if n.f >= 0]
        rx_d = copy.copy(z[0].rx_d)
        if len(rx_d) != len(to_schedule):
            raise IncompleteConvergecastError
        for slot, (relay, source) in rx_d.iteritems():
            if source not in to_schedule:
                raise IncompleteConvergecastError
class FlexiTPNet(ACSPNet):
    """Simulator of FlexiTP network.
    
    >>> FlexiTPNet(test_net3(), cont_f=10, Q=0.1, slot_t=2, n_exch=4, VB=False, until=100).print_dicts()
    ========================================
    node 0
    tx_d = {}
    rx_d = {0: (2, 2), 2: (2, 1), 5: (2, 3)}
    ct_d = {0: (2, 2), 1: (1, 1), 2: (2, 1), 3: (3, 3), 4: (1, 3), 5: (2, 3)}
    ========================================
    node 1
    tx_d = {1: 1, 4: 3}
    rx_d = {3: (3, 3)}
    ct_d = {0: (2, 2), 1: (1, 1), 2: (2, 1), 4: (1, 3), 5: (2, 3)}
    ========================================
    node 2
    tx_d = {0: 2, 2: 1, 5: 3}
    rx_d = {1: (1, 1), 4: (1, 3)}
    ct_d = {0: (2, 2), 2: (2, 1), 3: (3, 3), 5: (2, 3)}
    ========================================
    node 3
    tx_d = {3: 3}
    rx_d = {}
    ct_d = {1: (1, 1), 3: (3, 3), 4: (1, 3)}
    """
    AlNode = FlexiTPNode
    def __init__(z, wsn, cont_f=10, Q=0.1, slot_t=2, n_exch=70, nm=1000, 
                 fw=2, pause=3, VB=False, until=1e8):
        """Run the initial scheduling phase of FlexiTP.

        Construct a network and simulate it.


        Keyword parameters:
        wsn 
        cont_f    -- number of Q's in contention
        Q=0.1,    -- time before transmitting
        slot_t    -- duration of a transmission slot
        n_exch    -- number of slots for schedule exchange phase
        nm        -- maximum number of slots to be allocated
        fw        -- how many hops information to use
        pause     -- 3,
        VB        --
        until
        """
        t_w = cont_f * Q # contention window duration
        ft = n_exch * (t_w + slot_t) + pause # frame period
        stamp(z, locals())
        z.nadve = 0 # Number of slots used to broadcast one node's
                    # new schedule
        z[:] = [FlexiTPNode(id=i, sim=z) for i in xrange(len(z.wsn.f))]
        # Stamp each node with an indication on when it should be seeking to
        # gain a slot for itself. First, create a list of nodes in
        # breadth-first order.
        def bfs(node1_id=0, bfs_l=[]):
            bfs_l.append(node1_id)
            for node2_id in z.wsn.children(node1_id):
                bfs(node2_id, bfs_l)
            return bfs_l
        bfsl= bfs()[1:]
        z.print('The frame period is {0}.'.format(ft))
        z.print('The list of of nodes is ' + str(bfsl))
        frame_number = 1
        for node_id in bfsl:
            f_id = z.wsn.f[node_id]
            z.print('Node {0} to contend for {1} in frame {2}'
                    .format(f_id, node_id, frame_number))
            z[f_id].sc_d[frame_number] = node_id
            frame_number += z.wsn.tier(node_id)
        z.init_adaptation_statistics()
        z.simulate_net()
        z.complete_convergecast()
    def adjust_schedule_in_tx_phase(z):
        z.purge_old_slots()
        for node in z:
            if hasattr(node, 'sc_d') and node.sc_d:
                raise Error("This should not happen!")
            if node.f >= 0 and node.id not in node.tx_d.values():
                z[node.f].packet_l.append((node.id, node.id))
                z.print('Node {0} to contend for {1} in frame 0.'
                        .format(node.f, node.id))
        z.init_adaptation_statistics()
        z.simulate_net()
        for k, v in z.record.iteritems():
            z.record[k] = v / z.natre
            if z.VB:
                print("Record:{0} = {1}".format(k, z.record[k]))
        if z.VB: 
            print("======Schedule after the execution of the update=====")
            z.print_dicts()
class FlexiTPNet2(FlexiTPNet):
    """Simulator of FlexiTP without packet losses.

    This is useful in order to determine how many unfeasible allocations
    there are if no packet losses are lost propagating.

    """
    def __init__(z, wsn, fw=2, VB=False):
        """Run the initial scheduling phase of FlexiTP.

        fw : how many hops information to use

        the output are the lists: z[i].tx_d{slot: source}, where i is the
        index of each node.

        """
        z.wsn = wsn
        z.until = 1e5
        z[:] = [Node(id=i, sim=z) for i in xrange(len(wsn.f))]
        for node in z:
            node.children = [x for x in z if x.f == node.id]
        # queue = [(source, relay, slot)] used to process all the source
        # nodes in
        queue = [(i, i, 0) for i in (z[0].children)] 
        while queue:
            source, relay, slot = queue.pop(0) 
            if VB is True:
                print("Processing source={0}, relay={1}, slot={2}".
                        format(source, relay, slot))
            # Compute the set of nodes whose color I cannot use. Two parts:
            # 1) Nodes that interfere with my parent's reception
            interferers = k_neigh(node_set=z[relay].f, hops=fw, tx_l=wsn.tx_l)
            # 2) Nodes whose parents can interfere with me
            for q in k_neigh(node_set=relay, hops=fw, tx_l=wsn.tx_l):
                interferers.update(z[q].children) # Their parents
            # Start testing the next color to the one of the parent. 
            usedColors = set(z[relay].tx_d.keys())
            for interferer in interferers:
                usedColors.update(z[interferer].tx_d.keys())
            while slot in usedColors:
                slot += 1
            z[relay].tx_d[slot] = source 
            z[z[relay].f].rx_d[slot] = (relay, source)
            if source == relay:
                for j, ch in enumerate(z[relay].children):
                    queue.insert(j, (ch, ch, 0))
            if z[relay].f > 0:
                queue.insert(0, (source, z[relay].f, slot+1)) 
def test_manually_defined_network():
    '''Create simple network calling ManuallyDefinedNetwork.'''
    np.random.seed(3)
    return ManuallyDefinedNetwork(f=[ - 1, 0, 0, 1, 2, 2], txD={},
            ixD={0:(3, 4, 5), 1:(2, 4), 4:(5,)}) 
def test_pow():
    """Compute transmission range with typical WiFi parameters."""
    BW = 10 ** 7.3 # (10 * log10(bandwidth in Hz)
    noise_figure = 10 ** .5
    transceiver = dict()
    t = PhyNet(d0=100, PL0=10 ** (7.3), p_exp=3.5, shadow=8.0, tx_p=1e-5, 
                 BW=BW, sinr=1).tx_range
    print("Transmission range in meters: {0}".format(t))
def test_pow2():
    np.random.seed(2)
    wsn = PhyNet(c=30, x=200, y=200, n_tries=50, d0=100, PL0=1e8, p_exp=3.5,
                 shadow=8.0, tx_p=1e-6, noise=KBOL * T0 * 256e3, sinr=10)
    wsn.plot_tree(wsn.p, wsn.f)
    print("Tx range = {0} m".format(wsn.tx_range))
def test_pow_budget():
    '''Return the transmission range in a WIFI propagation environment.
    The parameters are taken from the paper Clark, 2002.  '''
    # Compute the tran
    tx_pow = - 20.0 # (dBm)
    BW = 73.0 # (10 * log10(bandwidth in Hz)
    F = 5.0 # Noise figure in dB
    noise = - 174.0 + BW # noise in dBm
    PE = 3.5 # Propagation exponent
    SSD = 8.0 # (dB) Standard deviation of shadowing
    PL100 = 73.0 # Path gain @ 100m (dB)
    # (dB) Standard deviation of the shadow fading or large scal fading
    shadowSTD = 8.0 
    # Thermal noise is KBOL*T*W, where KBOL is the Boltzman constant, T is the
    # room temperature (290 K = 17 C)
    sinr_threshold = 0 # dB
    # P_rx = tx_pow - PE * 10 * np.log10(node_l/100) -
    # np.random.randn() *     
    tx_range_aver = 100 * 10 ** (
        (tx_pow - PL100 - noise - F - shadowSTD - sinr_threshold) / PE / 10)
    # Above, 100 is the reference attenuation
    print("Transmission range in meters: {0}".format(tx_range_aver))
    # Create a sufficiently connected topology
def test_unconnected():
    '''Test unconnected_nodes_due_to_scheduling() for simple topologies.'''
    np.random.seed(3)
    w = DiskModelNetwork(f=[ - 1, 0, 0, 1, 2, 2], txD={},
            ixD={0:(3, 4, 5), 1:(2, 4), 4:(5,)}) 
    print(unconnected_nodes_due_to_scheduling(w.f, w.xL, [0,4,5,3,2,3]))
    w = Manual(f=[ - 1, 0, 0, 1, 2, 2], txD={},
            ixD={0:(3, 4, 5), 1:(2, 5), 4:(5,)}) 
    print(unconnected_nodes_due_to_scheduling(w.f, w.xL, [0,4,5,3,2,3]))
    w = Manual(f=[ - 1, 0, 0, 1, 2, 2], txD={},
            ixD={0:(3, 4, 5), 1:(2, 5), 3:(2,), 4:(5,)}) 
    print(unconnected_nodes_due_to_scheduling(w.f, w.xL, [0,4,5,3,2,3]))
def test_net1():
    """
    Create trivial network with two nodes.
    >>> print(test_net1().f)
    [-1  0]
    """
    np.random.seed(0)
    n = PhyNet(c=2, x=20, y=20, n_tries=50)
    # print(n)
    return n
def test_net2():
    """
    Create single hop network with three hops.
    >>> print(test_net2().f)
    [-1  0  0]
    """
    np.random.seed(0)
    return PhyNet(c=3, x=20, y=20)
def test_net3():
    """
    Chain of four nodes.
    >>> print(test_net3().f)
    [-1  2  0  1]
    """
    np.random.seed(0)
    return PhyNet(c=4, x=20, y=20)
def test_net4():
    """
    Create multi-hop network with three nodes.
    >>> print(test_net4().f)
    [-1  4  7  5  7  7  4  0]

    """
    np.random.seed(6)
    n = PhyNet(c=8, x=150, y=150, n_tries=50, tx_p=1e-13, BW=256e3, sinr=10)
    return n
def testVariance():
    """Compare the setups of FlexiTp and RandSched."""
    #TODO
    n = test_net1()
    a = ACSPNet(n)
    f = FlexiTPNet(n)
    n_rep = 2
    for i in xrange(n_rep):
        modify_network(n)
def testFlexiTP():
    n = test_net4()
    print(n.f)
    f = FlexiTPNet(n, slot_t=2, VB=True, until=400)
def graph_fatrob(tst_nr=6, action='plot'):
    assert action in ('compute', 'plot')
    file_name = os.path.join("graph_fatrob", "{0:02d}".format(tst_nr))
    outputdir = ('./', )

    def export_fig(figure_number):
        f = plt.figure(figure_number)
        for o in outputdir:
            for format in ('eps', 'pdf'):
                f.savefig('%s%s_%d.%s' %(o, file_name, figure_number, format))

    # tst_nr chooses simulation parameters. The execution times are 1:7',
       # 2:28', 3:7h, 4:16h, 5:17h, 6:35.6h
    # Select simulation parameters
    tx_rg = 60
    # Number of Monte Carlo simulations
    REPEAT = [4, 20, 100, 500, 400, 1600, 3200][tst_nr]
    # Distance to the sink normalized by transmission range
    vDist = np.array([[2,6], [2,6,12], [4,6,10], [4,7,10], [4,7,10],
           [4,7,10], [4,7,10]][tst_nr])
    # node density=c*pi*tx_rg^2/x/y. Greatly increases the execution time.
    rhoV = np.array([(8,14), (7,13,15), (7,13,15), (7,11,15),
       (7,14,21,28), (7,14,28), (7,14,28)][tst_nr])
    failP = np.array([(.05,.2,.5), (.2,), (.1,), (.01,.05,.09,.13,.17), (.09,), (.05), (.05,)][tst_nr])


    if action == 'compute':
        # Compute and store the probability of timely delivery with FAT for
        # different node densities and event locations.
        x = max(vDist) * tx_rg # Width of the sensing area square
        y = 4 * tx_rg # Height of the sensing area
        # Compute number of nodes based on the area and the node density.
        nV = np.array((rhoV * x * y / np.pi / tx_rg **2).round(), int)
        # Initialize the vector that will contain the results:
        r = np.zeros((3, rhoV.size, vDist.size, failP.size, REPEAT)) 
        np.random.seed(4)
        for k in xrange(REPEAT):
            print("Iteration {0}".format(k))
            for i,c in enumerate(nV):
                # Try a certain number of times to obtain a reasonably
                # connected network
                w = net.DiskModelNetwork(c=c, x=x, y=y, tx_rg=tx_rg,
                        ix_rg=2.1*tx_rg)
                for h, fp in enumerate(failP):
                    working = np.random.rand(c) > fp
                    for j, d in enumerate(vDist * tx_rg):
                        print("EXECUTION ({0},{1})".format(i,j))
                        src = event_sources(loc=[x,y/2], n_src=n_src)
                        r[:,i,j,h,k] = w.fat(src=src,
                                working=working.copy())[1]
            # Save the simulation results into a file
                        np.save(file_name, r=r)
    elif action == 'plot':
        '''Load and plot simulation results.'''
        r = np.load(file_name + '.npy')
        plt.close('all')
        plt.ioff()
        f1 = plt.figure(1, figsize=(6, 3 * len(failP)))
        frmt = ('rp--','<b-.','>g:','^k-')
        for k, f in enumerate(failP):
            for g, x in enumerate(r):
                s = plt.subplot(len(failP), 3, 1 + g + (3 * k))
                for i, rho in enumerate(rhoV):
                    plt.plot(vDist, x.mean(3)[i,:,k], frmt[i], 
                           label=r"$\bar{\rho}$ = %d" %rho)
                plt.legend(loc='lower left')
                plt.axis([min(vDist),max(vDist),-.1,1.1])
                plt.xlabel('Normalized event distance')
                plt.title('failP = %f' %f)
                if g == 0:
                    plt.ylabel('E[timely delivery]')
                elif g == 1:
                    s.set_yticklabels([])
        export_fig(1)
        if tst_nr == 6 and True:
            fpIndex = 0 # index of failP to choose
            x = 1 - r.mean(4)[:,:,:,fpIndex] # Type, density, distance
            P, Q = x.shape[1:3] # P number of densities, Q of distances
            W = .8 # Bar width in bar plot
            D = (1 - W) / 2 # Space between bars in bar plot
            font = { 'fontname':'Times New Roman','fontsize':9}
            font2 = FontProperties(size=10,family='serif')
            f = plt.figure(2, figsize=(2.9,2))
            ax = plt.axes([.15,.20,.8,.7]) # Place axes in figure
            for i, g in enumerate(x): # different techniques
                for j, h in enumerate(g): # different densities
                    for k, l in enumerate(h): # different distances
                        b, = plt.bar(k * (Q + 1) + j + D, l, width=W, 
                               color=str(float(i)/2))
                        if i == 0: 
                            plt.text(k * (Q + 1) + j + .5, l + .02,
                                   r'$\bar{\rho}_%d$' %j, ha='center',**font)
                        if j == 0 and k == 0: 
                            labels = (r'$f_s$', r'$f_b$', r'$f_e$')
                            b.set_label(labels[i])
            l = plt.legend(loc=2, prop=font2)
            # Type the node density used
            t = '$' + ', '.join([r'\bar{\rho}_%d = %d' %(i,r) for i, r in
               enumerate(rhoV)]) + '$'
            ax.text(3., .7, t, **font)
            # Plot the hop number
            t = np.arange(P) * (Q + 1) + float(P) / 2
            ax.set_xticks(t)
            ax.set_xticklabels(["%d" %i for i in vDist], **font)
            labels = plt.getp(plt.gca(), 'yticklabels')
            plt.setp(labels, **font)
            plt.setp(labels, fontsize=8)
            plt.xlabel(r'normalized event-sink distance, $d/r_t$',
             **font)
            plt.ylabel(r'source isolation prob. $f$', **font) 
            plt.xlim(-1, P * (Q + 1))
            plt.ylim((0, x.max() * 1.8))
            export_fig(2)
            plt.show()
def graphACSPNet0(tst_nr, action):
    """Compare the setup of ACSP and FlexiTP.
    
    Compute:
    - setup time
    - size of the schedule
    as a function of:
    - the number of layers in the network
    - the node density?

    """
    file_name = "graphRandSched2_{0:02d}".format(tst_nr)
    file_name = "{0}_{1:02d}".format(sys._getframe().f_code.co_name, tst_nr)
    tx_rg = PhyNet(d0=100, PL0=1e8, p_exp=3.5, shadow=8.0).tx_rg()
    x, y = np.array([[1,8],[3,8]][tst_nr]) * tx_rg
    rho_v = np.array([[7,14], [7,14,28]] [tst_nr])
    repetitions = [2, 1107][tst_nr] # Number of repetitions
def graphACSPNet1(tst_nr, action):
    """Compare ACSP and FlexiTP.

    Run different cycles in which I add and remove nodes to the network.

    I execute ACSP and FlexiTP until all the nodes have found a slot.  I
    have to compute the probability that. """
    pass
def test_ACSPNet_regen():
    np.random.seed(0)
    w = test_net3()
    print(w.f)
    nw = ACSPNet(w, VB=False, until=1000)
    w.mutate_network(2, 2)
    nw.VB = True
    nw.adjust_schedule_in_tx_phase(until=200)
    print(nw.n_frames())
    # nw.print_dicts()
    # w.mutate_network(6, 2)
    # w.plot_tree()
def name_npz():
    """Return string with function and parameters of grandparent funtion.

    Each frame_record is a tuple consisting of the frame object, the
    filename, the line number of the current line, the function name, a
    list of lines of context from the source code, and the index of the
    current line within that list.

    """
    frame_record = inspect.stack(0)[2]
    variables = frame_record[0].f_locals
    return "{0}_{1:02d}_{2:06d}".format(frame_record[3], 
                                            variables['tst_nr'],
                                            variables['repetitions'])
def save_npz(*args):
    frame_record = inspect.stack(0)[1]
    variables = frame_record[0].f_locals
    to_save = {}
    for variable in args:
        to_save[variable] = variables[variable].mean(axis=0)
    np.savez(name_npz() + '.npz', **to_save)
def savedict(**out):
    to_save = dict((k, v.mean(axis=0)) for k,v in out.iteritems())
    np.savez(name_npz() + '.npz', **to_save)
def load_npz():
    npz = np.load(name_npz() + '.npz')
    for i in npz.files:
        print(i)
        print(npz[i])
    return npz
def ellapsed_time():
    totseconds = int(time_module.time() - start_time)
    seconds = totseconds
    days = seconds / 86400
    seconds -= days * 86400
    hours = seconds / 3600
    seconds -= hours * 3600
    minutes = seconds / 60
    seconds -= minutes * 60
    return "{0}. Total time {1}\" ({2}d {3}h {4}' {5}).\"".format(
           time_module.asctime(), totseconds, days, hours, minutes, seconds)
def print_iter(iteration_n, total):
    np.random.seed(iteration_n)
    print("%s Iteration %d/%d" %(ellapsed_time(), iteration_n, total))
def print_nodes(c, seed):
    np.random.seed(seed)
    print("{0}.  Simulating for {1:3d} nodes"
            .format(ellapsed_time(), c))
def graphAcspPairs(tst_nr=0, repetitions=1, action=0, plot=0):
    """ACSP setup as a function of the number of pairs."""
    x = [3, 5][tst_nr] * tx_rg1
    y = x
    rho_v = np.array([[7, 14], [7, 14, 21, 29]][tst_nr])
    n_nodes = np.array((rho_v * x * y / np.pi / tx_rg1**2).round(), int)
    print('Number of nodes to be tested: {0}'.format(n_nodes))
    pairs = np.arange(1, 25, 2)
    n_slots = np.zeros((repetitions, rho_v.size, pairs.size))
    if action == 1:
        for k in xrange(repetitions):
            print_iter(k, repetitions)
            for i , c in enumerate(rho_v):
                print("Simulating for {0:3d} nodes".format(c))
                wsn = PhyNet(c=c, x=x, y=y, **net_par1)
                for j, p in enumerate(pairs):
                    try:
                        s = ACSPNet(wsn, pairs=p).n_slots()
                    except NoProgressError:
                        s = - 10000 * repetitions
                    n_slots[k, i, j] = s
        save_npz('n_slots')
    r = load_npz()
    #last_column = np.array(r['n_slots'][:,-1], dtype=float).reshape((-1, 1))
    #increase = (r['n_slots'] - last_column) / last_column
    #print("increase = {0}".format(increase)) 
    g = Pgf()
    g.add("contention pairs in ACSP", "relative schedule increase in \%")
    for rho, n_slots in zip(rho_v, r['n_slots']):
        idx = np.nonzero(n_slots > 0)
        x = pairs[idx]
        y = n_slots[idx]
        y = 100. *(y - y[-1]) / float(y[-1])
        g.plot(pairs[idx], y, r"$\bar{{\rho}}$ = {0}".format(rho))
    g.save(plot=plot)
def graphFatMatlab():
    """Plot the results obtained with matlab."""
    from scipy.io import loadmat
    d = loadmat('graph_fat_matlab.mat')
    lv = ('-+', '-.^', '--v', ':o', ':d', ':<', ':')
    lg = ('DMAC','SPT','FAT','Centralized1')
    q = d['s'][[1],1,-1,:]
    yv = (d['s'][[3,5],1,-1,:] - q) / q * 100
    alphav = d['acV'].flatten()
    #%title('nPkts = 10,x = y = 200, txRg = 60,ixRg = 150, N = 50')
    g = Pgf()
    g.add(r'Aggregation coefficient $\alpha$', 
            r'traversal time $D_t$ compared with SPT (\%)')
    for y, l in zip(yv, ('FAT', 'Centralized1')):
        p.plot(alphav, y, l)
    fv = plt.figure()
    sources = d['vS'].flatten()
    g.add(r'Number of sources $s$', 'traversal time $D_t$ (seconds)')
    for y, l in zip(d['s'][[0,1,3,5],1,:,0], lg):
        p.plot(sources, y, l)
    p.save()
def graphFlexiCycles(tst_nr=0, repetitions=1, action=0, plot=False):
    """Dependence on the number of channel change cycles. """
    x, y = np.array([[2, 2],[3,3], [4, 4]][tst_nr]) * tx_rg1
    rho = 8
    repetitions = [2, 400][tst_nr] # Number of repetitions
    cycles = [3, 3][tst_nr]
    c = int(np.round(rho * x * y / np.pi / tx_rg1**2))
    nnew = 5 # Number of new nodes to add 
    print(("Simulated network contains {0} nodes. " + 
            "In each cycle, the last {1} nodes are replaced.")
            .format(c, nnew))
    n_slots_tx = np.zeros((repetitions, cycles + 1, 3))
    n_dismissed = np.zeros((repetitions, cycles + 1, 2))
    if action == 1:
        for k in xrange(repetitions):
            print_iter(k,repetitions)
            wsn = PhyNet(c=c, x=x, y=y, **net_par1)
            nets = [FlexiTPNet(wsn, fw=2, n_exch=80), 
                    FlexiTPNet(wsn, fw=3, n_exch=80),
                    ACSPNet(wsn, cont_f=100, pairs=40)]
            n_slots_tx[k, 0, :] = [n.n_slots() for n in nets]
            n_dismissed[k, 0, :] = [n.dismissed() for n in nets[:2]]
            for q in xrange(1, cycles + 1):
                np.random.seed(q)
                wsn.mutate_network(c - nnew, nnew)
                print("Cycle number {0}/{1}".format(q, cycles))
                for j in xrange(3):
                    np.random.seed(k)
                    print("Updating net {0}".format(j))
                    if j < 2:
                        nets[j].adjust_schedule_in_tx_phase()
                    else:
                        nets[j].adjust_schedule_in_tx_phase(cap=2, mult=16)
                n_slots_tx[k, q, :] = [n.n_slots() for n in nets]
                n_dismissed[k, q, :] = [n.dismissed() for n in nets[:2]]
         # wsn.plot_tree()
        save_npz('n_slots_tx', 'n_dismissed')
    r = load_npz()
    # Plot schedule length. Takeaway: FlexiTP requires more slots when the
    # network changes.
    x_t = r'number of network cycles'
    leg = ['FlexiTP2', 'FlexiTP3', 'ACSPNet']
    g = Pgf()
    g.add(x_t, r'number of slots $M$')
    g.mplot(range(cycles + 1), r['n_slots_tx'], leg)
    g.opt(r'legend style={at={(1.02,0.5)}, anchor=west }')
    # Plot dismissal probability. Takeaway: FlexiTP incorrectly dismisses
    # parents as the number of cycles grows.
    g.add(x_t, r'dismissal probability $p_d$')
    g.mplot(range(cycles + 1), r['n_dismissed'], leg[:2])
    g.save(plot=plot)
# Slots of centralized algorithms
#     w = test_net3()
#     print(w.f)
#     nw = ACSPNet(w, VB=False, until=1000)
#     w.mutate_network(2, 2)
#     nw.VB = True
def graphFlexiLength(tst_nr=0, repetitions=1, action=0, plot=0):
    """Dependence on the normalized network length in a square network.

    tst_nr:seconds per iteration, 1:5687@eemoda
    """
    rho = 8
    y_v = np.arange(*[[1, 4, 2], [1, 10, 2]][tst_nr]) * tx_rg1
    n_nodes = np.array((rho * y_v * y_v / np.pi / tx_rg1**2).round(), int)
    print('Number of nodes to be tested: {0}'.format(n_nodes))
    n_slots = np.zeros((repetitions, n_nodes.size, 3))
    n_frames = np.zeros((repetitions, n_nodes.size, 3))
    if action == 1:
        for k in xrange(repetitions):
            print_iter(k, repetitions)
            for i, (c, y) in enumerate(zip(n_nodes, y_v)):
                print_nodes(c, k)
                wsn = PhyNet(c=c, x=y, y=y, **net_par1)
                nets = []
                for j in xrange(3):
                    np.random.seed(k)
                    print("Constructing net {0}".format(j))
                    if j < 2:
                        nets.append(FlexiTPNet(wsn, fw=j+2))
                    else:
                        nets.append(ACSPNet(wsn, cont_f=100, pairs=40))
                n_slots[k, i, :] = [n.n_slots() for n in nets]
                n_frames[k, i, :] = [n.n_frames() for n in nets]
        save_npz('n_slots', 'n_frames')
    r = load_npz()
    # Plot setup frames
    x_t = r'normalized network length $\bar{y}$'
    yn = y_v / tx_rg1
    leg = ['FlexiTP2', 'FlexiTP3', 'ACSPNet']
    g = Pgf()
    g.add(x_t, 'schedule length')
    g.mplot(yn, r['n_slots'], leg)
    g.opt(r'legend style={at={(1.02,0.5)}, anchor=west }')
    g.add(x_t, 'number of setup frames')
    g.mplot(yn, r['n_frames'], leg)
    g.save(plot=plot)
def graphRandSched1(tst_nr=1, action=0):
    """ Generate two graphs using SINR model:
    
    1) number of slots for RandSched and the First Breadth First
       centralized algorithms.
       
    2) reliability of the First Breadth First algorithms, for the unit
       disk graphs.
       
    Parameters:
    tst_nr -- 0 or 1
    
    """
    def bar(ax, x, y, labels, color):
        '''Make bar plot.

        Each row in y is grouped in one block of bars in the x-axis.
        color: if True, use colors, otherwise use grayscale

        Example:
        x = np.array([0,1])
        y = np.arange(6).reshape(3,2)
        bar(x, y, ['one', 'two', 'three'])
        l = plt.legend(prop=font2) '''
        P, Q = y.shape # P number of densities, Q of distances
        W = .8 # Bar width in bar plot
        D = (1 - W) / 2 # Space between bars in bar plot    plt.ioff()
        for i, g in enumerate(y): # different x-axis locations
            # different magnitudes within an x location
            for j, h in enumerate(g): 
                b, = ax.bar(i * (Q + 1) + j + D, h, width=W, color= ('red',
                    'blue', 'green')[j] if color else str(float(j)/Q))
                if i == 0: 
                    b.set_label(labels[j])
        t = np.arange(P) * (Q + 1) + float(Q) / 2
        plt.xlim(-1, P * (Q + 1))
        ax.set_xticks(np.arange(P) * (Q + 1) + float(Q) / 2)
        ax.set_xticklabels(["%d" %i for i in x])

    file_name = "{0}_{1:02d}".format(sys._getframe().f_code.co_name, tst_nr)
    tx_rg = PhyNet(tx_p=1e-6, BW=256e3, sinr=20, d0=100, PL0=1e8, 
                   p_exp=3.5, shadow=8.0).tx_range
    x, y = np.array([[1,8],[3,8]][tst_nr]) * tx_rg
    rho_v = np.array([[7,14], [7,14,28]] [tst_nr])
    repetitions = [10, 2400][tst_nr] # Number of repetitions
    # Number of nodes in the network (cardinalityVector)
    n_nodes = np.array((rho_v * x * y / np.pi / tx_rg**2).round(), int)
    # Slots of centralized algorithms
    slots = np.zeros((repetitions, n_nodes.size, 3)) 
    # Unconnection ratio of of centralized algorithms
    uncon = np.zeros((repetitions, n_nodes.size, 2))
    if action == 0:
        for k in xrange(repetitions):
            print("Iteration =  {0}".format(k))
            np.random.seed(k)
            for i, c in enumerate(n_nodes):
                wsn = PhyNet(c=c, x=x, y=y, n_tries=50)
                hops = [2, 3]
                for j, h in enumerate(hops):
                    slot_v = wsn.bf_schedule(hops=h)
                    slots[k, i, j] = max(slot_v)
                    uncon[k, i, j] = wsn.duly_scheduled_sinr(slot_v)
                rs_net = RandSchedNet(wsn, cont_f=999, pairs=99, Q=0.1,
                        slot_t=2, VB=False, until=1e9)
                slots[k, i, 2] = max(rs_net.schedule())
        np.savez(file_name, slots=slots, uncon=uncon)
    elif action == 1:
        rslt = np.load(file_name + '.npz')
        slots = rslt['slots'].mean(axis=0)
        uncon = rslt['uncon'].mean(axis=0)
        # Comparison of slot number of BF vs RandSched
        g = Pgf()
        g.add(r'node density $\rho$', r'number of slots $M$')
        g.opt('ybar', 'bar width=10pt', 'ymin=0', 'enlarge x limits=0.2')
        g.opt(r'legend style={at={(1.02,0.5)}, anchor=west }')
        g.opt(r'xtick={' + ', '.join([str(i) for i in xrange(len(rho_v))]) 
                                     + r'}')
        g.opt(r'xticklabels={' + ', '.join([str(i) for i in rho_v]) + r'}' )
        s = (slots / n_nodes.reshape((3,1)))[:,:3]
        #pdb.set_trace()         # 
        g.plot(range(3), s[:, 0], 'BF2', 'fill=black!90')
        g.plot(range(3), s[:, 1], 'BF3', 'fill=black!40')
        g.plot(range(3), s[:, 2], 'RandSched', 'fill=black!10')
        g.add(r'node density $\rho$', 'fraction of unconnected nodes')
        g.opt('ymode=log')
        g.plot(rho_v, uncon[:,0], 'BF2')
        g.plot(rho_v, uncon[:,1], 'BF3')
        g.save()
    elif action == 2:
        # Create plots for presentation
        rslt = np.load(file_name + '.npz')
        slots = rslt['slots'].mean(axis=0)
        uncon = rslt['uncon'].mean(axis=0)  
        # Comparison of slot number of BF vs RandSched
        plt.close('all')
        fig = [plt.figure(i, figsize=(3,2)) for i in xrange(2)]
        ax = [f.add_axes([.2, .3, .7, .5]) for f in fig]
        ax[0].set_ylabel('schedule size')
        ax[0].set_xlabel('node density')
        label = ('Breadth First', 'BF3', 'RandSched')
        choi = 0, 2
        bar(ax[0], rho_v, slots[:,choi], [label[i] for i in choi], True)
        ax[0].legend(loc=2)
        ax[1].set_title('Unduly scheduled nodes')
        ax[1].set_xlabel('node density')
        marker = ['>-','v-','<-','o-']
        for v, m in zip(uncon.T, marker):
            print(v)
            ax[1].semilogy (rho_v, v, m)
        ax[1].set_ybound(1e-6,.3)
        ax[1].legend(('BF2', 'BF3'), loc='lower right')
        #plt.close(1)
        # Customize fonts and save
        font1 = FontProperties(size=14, family='Calibri')
        font2 = FontProperties(size=8)
        tit_font = FontProperties(size=14, weight='bold')
        for a in ax[0],:
            for x in ['x','y']:
                # Set tick label fonts
                for l in getattr(a,'get_{0}ticklabels'.format(x))():
                    l.set_fontsize(8) #properties(font2)
            # Set font of legends    
            for l in a.legend_.get_texts():
                l.set_fontproperties(font1)
        #    a.title.set_fontproperties(tit_font)
        fig[0].savefig('{0}_pres00.png'.format(file_name), dpi=300)
        plt.show()
    else:
       raise Exception, "2nd parameter must be 'compute' or 'plot'"
def graphRandSched2(tst_nr=1, action=2):
    '''Plot size of RandSched's schedule for different node densities,
    numbers of pairs of slots, and lengths of the contention window.

    Parameters to call this script with:
    * tst_nr:
            0: fast test
            1: parameters for publication

    * action: 
        0: compute the results and store them in a file
        1: plot all the results
        2: (RECOMENDED): plot a reduced version for publication

    '''
    file_name = "{0}_{1:02d}".format(sys._getframe().f_code.co_name, tst_nr)
    file_name = "graphRandSched2_{0:02d}".format(tst_nr)
    tx_rg = PhyNet(tx_p=1e-6, BW=256e3, sinr=20, d0=100, PL0=1e8, p_exp=3.5,
                   shadow=8.0).tx_range
    x, y = np.array([[2,4],[3,2]][tst_nr]) * tx_rg
    # Node density
    rho_v = np.array([[7,14,28,40], [7,14,28,56]] [tst_nr])
    # cont_f = how many times is the contention window longer than Q
    cont_f = np.array([[2,8],[2,4,8,16,32]][tst_nr], int)
    # Pairs of slots for the resolution of the hidden terminal problem
    pairs = np.array([[2,32],[2,4,8,16,32,64]][tst_nr])
    repetitions = [1, 402][tst_nr] 
    n_nodes = np.array((rho_v * x * y / np.pi / tx_rg**2).round(), int)
    # r is the matrix that will contain the results
    r = np.zeros((n_nodes.size, len(cont_f), len(pairs),
                  repetitions)) 
    # Set graphical properties
    font1 = FontProperties(size=8,family='serif')
    font2 = dict(fontsize=8)
    font3 = FontProperties(size=10,family='serif')
    fs = (3,9)
    dpi = 200
    def pub(figure_number):
        plt.savefig('{0}_{1}.png'.format(file_name, figure_number), dpi=dpi)
        plt.savefig('{0}_{1}.eps'.format(file_name, figure_number), dpi=dpi)
    if action == 0:
        for h in xrange(repetitions):
            print("Iteration =  {0}".format(h))
            np.random.seed(h)
            for i, c in enumerate(n_nodes):
                wsn = PhyNet(c=c, x=x, y=y, n_tries=50)
                for j, c_f in enumerate(cont_f):
                    for k, p in enumerate(pairs):
                        try:
                            rs_net = sc.RandSchedNet(wsn, cont_f=c_f,
                                pairs=p, Q=0.1, slot_t=2, VB=False)
                            rs_net.run_su(until=1e9)
                            r[i,j,k,h] = max(rs_net.schedule())
                        except sc.NoProgressError:
                            # Do not change the value --> error indication
                            pass 
        np.save(file_name, r)
    elif action == 1:
        r = np.load(file_name + '.npy') 
        plt.ioff()
        plt.close('all')
        figure_number = 1
        fig = plt.figure(figure_number, figsize=fs)
        for i, rho in enumerate(rho_v):
            ax = fig.add_subplot(len(rho_v), 1, i + 1)
            m = 0 # max
            for j, c_f in enumerate(cont_f):
                x = np.array([[p, a.mean()] 
                              for p, a in zip(pairs, r[i,j]) if a.all()])
                if x.size:
                    ax.plot(np.log2(x[:,0]), x[:,1], '-o', 
                            label="contF = {0}".format(c_f))
                    m = max(m, max(x[:,1]))
            ax.set_title(r'Node density $\rho$ = {0}'.format(rho), font2)
            ax.set_ylabel('Number of slots', font2)
            ax.set_ybound(0, m)
            plt.legend(loc='lower left', prop=font1)
            plt.setp(ax.get_xticklabels() + ax.get_yticklabels(), 
                    fontproperties=font1)
        ax.set_xlabel('log2(pairs)', font2)
        fig.subplots_adjust(left=0.2, hspace=0.5)
        plt.show()
        pub(1)
    elif action == 2:
        r = np.load(file_name + '.npy') 
        g = Pgf()
        g.add(r'$\log_2(L)$', r'number of slots $M$')
        for q, i in enumerate((2,3)):
            m = 0 # max
            for k, j in enumerate((0,4)):
                x = np.array([[p, a.mean()] 
                              for p, a in zip(pairs, r[i,j]) if a.all()])
                g.plot(np.log2(x[:,0]), x[:,1],
                       r"$\rho={0}, \gamma = {1}$"
                       .format(rho_v[i], cont_f[j]))
        g.save()
    else:
        raise Exception, "Unexpected action"
def graphRandSched3(tst_nr=1, action=1):
    '''Plot size of RandSched's schedule for different node densities,
    numbers of pairs of slots, and lengths of the contention window.

    Parameters to call this script with:
    * tst_nr:
            0: fast test
            1: parameters for publication
    * action: 
        0: compute the results and store them in a file
        1: plot all the results
    '''
    x, y = np.array([[2,4],[4,4]][tst_nr]) * tx_rg1
    rho_v = np.array([[7,14,28,40], [7,18,29]] [tst_nr])
    pairs = np.array([[2,32],[4,6,8,10,12]][tst_nr])
    repetitions = [1, 500][tst_nr] 
    n_nodes = np.array((rho_v * x * y / np.pi / tx_rg1**2).round(), int)
    n_slots = np.zeros((repetitions, n_nodes.size, len(pairs))) 
    if action == 0:
        for h in xrange(repetitions):
            print("Iteration =  {0}".format(h))
            for i, c in enumerate(n_nodes):
                print_nodes(c, h)
                wsn = PhyNet(c=c, x=x, y=y, n_tries=50)
                for j, p in enumerate(pairs):
                    rs_net = RandSchedNet(wsn, cont_f=40, pairs=p,
                            Q=0.1, slot_t=2, VB=False, until=1e9)
                    n_slots[h,i,j] = max(rs_net.schedule())
        save_npz('n_slots')
    r = load_npz()
    g = Pgf()
    g.add("contention pairs", "relative schedule increase \%")
    for rho, n_slots in zip(rho_v, r['n_slots']):
        idx = np.nonzero(n_slots > 0)
        x = pairs[idx]
        y = n_slots[idx]
        y = 100. *(y - y[-1]) / float(y[-1])
        g.plot(pairs[idx], y, r"$\bar{{\rho}}$ = {0}".format(rho))
    g.save(plot=action)
def graphRandSched4(tst_nr=1, action=1, plot=0):
    '''Plot M/N (schedule size / number of nodes in the network).

    Parameters to call this script with:
    * tst_nr:
            0: fast test
            1: parameters for publication

    * action: 
        0: compute the results and store them in a file
        1: plot all the results
    '''
    xv = np.array([[4, 6, 8, 10, 12],[4, 6, 8, 10, 12]][tst_nr]) * tx_rg1
    rho = 7
    repetitions = [1, 10][tst_nr] 
    n_nodes = np.array((rho * xv**2 / np.pi / tx_rg1**2).round(), int)
    n_slots = np.zeros((repetitions, n_nodes.size)) 
    if action == 0:
        for h in xrange(repetitions):
            print("Iteration =  {0}".format(h))
            for i, (c, x) in enumerate(zip(n_nodes, xv)):
                print_nodes(c, h)
                wsn = PhyNet(c=c, x=x, y=x, n_tries=50, **net_par1)
                rs_net = RandSchedNet(wsn, cont_f=40, pairs=6,
                                      Q=0.1, slot_t=2, VB=False, until=1e9)
                n_slots[h,i] = max(rs_net.schedule())
        save_npz('n_slots')
    norm = load_npz()['n_slots'] / n_nodes
    g = Pgf()
    g.add("number of node in the network $M$", "$M/N$")
    g.plot(n_nodes, norm)
    g.save(plot=action)
def tst_FlexiTP2():
    np.random.seed(0)
    wsn = PhyNet(c=8,x=3*tx_rg1, y=1*tx_rg1, **net_par1)
    print("The tree is {0}".format(wsn.f))
    n1 = FlexiTPNet2(wsn, VB=True)
    n1.complete_convergecast()
    n1.print_dicts()
    print(n1.dismissed())
def graphFlexiDensity(tst_nr=-1, repetitions=1, action=0, plot=0):
    """Dependence on the node density.

    tsn_nr:seconds per iteration:
    ee-moda2: 2:1235
    ee-moda:  2:623
    """
    x, y = np.array(((3,3), (3,3), (3,3))[tst_nr]) * tx_rg1
    rho_v = np.array(((7,10), (7,11,15), (7,11,15,19,22)) [tst_nr])
    n_nodes = np.array((rho_v * x * y / np.pi / tx_rg1**2).round(), int)
    print('Number of nodes to be tested: {0}'.format(n_nodes))
    o = dict(
        # number of nodes unduly dismissed as unreachable
        dismi = np.zeros((repetitions, n_nodes.size, 4)),
        #  slots per FTS used in the initialization phase of FlexiTP
        nadv1 = np.zeros((repetitions, n_nodes.size, 2)),
        # number of slots used in FlexiTP's setup to propagate schedule
        slosu=np.zeros((repetitions, n_nodes.size, 3)),
        )
    # Number of slots per FTS used in the transmission phase of FlexiTP
    if action == 1:
        for k in xrange(repetitions):
            print_iter(k, repetitions)
            for i, c in enumerate(n_nodes):
                print_nodes(c, k)
                wsn = PhyNet(c=c, x=x, y=y, **net_par1)
                nets = []
                for j in xrange(5):
                    np.random.seed(k)
                    print("Constructing net {0}".format(j))
                    if j in (0, 1):
                        nets.append(FlexiTPNet(wsn, fw=j + 2, n_exch=70)) 
                    elif j == 2:
                        nets.append(ACSPNet(wsn, cont_f=100, pairs=40))
                    else:
                        nets.append(FlexiTPNet2(wsn, fw=j - 1)) 
                o['slosu'][k,i,:] = [n.n_slots() for n in nets[:3]]
                o['dismi'][k,i,:] = [nets[h].dismissed() for h in (0,1,3,4)]
                o['nadv1'][k,i,:] = [n.nadve for n in nets[:2]]
        savedict(**o)
    # Plot schedule sizes
    r = load_npz()
    x_t = r'node density $\bar{\rho}$'
    leg = ['FlexiTP2', 'FlexiTP3', 'ACSPNet']
    g = Pgf()
    g.add(x_t, r'number of slots $M$')
    g.mplot(rho_v, r['slosu'], leg)
    g.opt(r'legend style={at={(1.02,0.5)}, anchor=west }')
    # Plot dismissal probability. 
    g.add(x_t, r'fraction of unduly dismissed $p_d$')
    g.mplot(rho_v, r['dismi'], leg[:2]+ ["fw=2", "fw=3"])
    # Plot number of slots necessary to communicate schedule
    g.add(x_t, r"slots per exchange in FlexiTP's init")
    g.mplot(rho_v, r['nadv1'], leg[:3])
    g.save(plot=plot)
def debgraphFlexiDensity():
    """Dependence on the node density.
    """
    tst_nr=2
    repetitions=1
    action=1
    plot=1
    x, y = np.array(((3,3), (3,3), (3,3))[tst_nr]) * tx_rg1
    c = 63
    print_nodes(c, 9)
    wsn = PhyNet(c=c, x=x, y=y, **net_par1)
    np.random.seed(8)
    n1 = FlexiTPNet(wsn, fw=2, n_exch=70)
def graphFlexiSds(tst_nr=0, repetitions=1, action=0, plot=False):
    """Dependence on the number of SDS tones. 

    tsn_nr:seconds per iteration, 1:2700@ee-modalap

    DEBUGGING:
    
    It breaks down for iteration 962, SDS = 0, simulating for 20 nodes.
    """
    bitrate = 19.2e3
    packet_size = 56 * 8 # bits
    slot_t = packet_size / bitrate
    sglt = 20.0 / bitrate # 10 bits
    x, y = np.array([[2, 2],[3,3]][tst_nr]) * tx_rg1
    rho_v = np.array(((7,8,9), (7,11,15,19,22), 
                      (7,10,13,16,19,21))[tst_nr])
    # Indices of rho_v selected for the table
    ind_rho_table = ((0, 1, 2), (0,2,4))[tst_nr]
    # Since the number of expelled nodes per incorporation is 0.05, the
    # minimum number of repetitions should be 200.
    n_nodes = np.array((rho_v * x * y / np.pi / tx_rg1**2).round(), int)
    sdsl = ((0,1,2), (0,1,2))[tst_nr]
    fltfw = (2, 3) # values of fw used in FlexiTP
    nnew = 2 # Number of new nodes to add 
    o = dict(
        attem=np.zeros((repetitions, len(rho_v), len(sdsl))),
        dismi=np.zeros((repetitions, len(rho_v), 1+len(fltfw))),
        energ=np.zeros((repetitions, len(rho_v), len(sdsl))),
        expe1=np.zeros((repetitions, len(rho_v), len(sdsl)+len(fltfw))),
        expe2=np.zeros((repetitions, len(rho_v), len(sdsl)+len(fltfw))),
        laten=np.zeros((repetitions, len(rho_v), len(sdsl)+len(fltfw))),
        lates=np.zeros((repetitions, len(rho_v), len(sdsl)+len(fltfw))),
        losse=np.zeros((repetitions, len(rho_v), len(sdsl))),
        nadv1=np.zeros((repetitions, len(rho_v), len(fltfw))),
        nadv2=np.zeros((repetitions, len(rho_v), len(fltfw))),
        natre=np.zeros((repetitions, len(rho_v))),
        pkets=np.zeros((repetitions, len(rho_v))),
        sloov=np.zeros((repetitions, len(rho_v), len(sdsl) + len(fltfw))),
        slosu=np.zeros((repetitions, len(rho_v), 1+len(fltfw))),
        )
    print(n_nodes)
    if action == 1:
        for k in xrange(repetitions):
            print_iter(k, repetitions)
            for i, c in enumerate(n_nodes):
                print_nodes(c, k)
                wsn = PhyNet(c=c, x=x, y=y, **net_par1)
                hu = sum(x for x in (wsn.tier(w) for w in xrange(wsn.c))
                         if x < TIER_MAX)
                o['pkets'][k,i] = hu
                nt = [ACSPNet(wsn, cont_f=100., Q=1/bitrate, slot_t=slot_t,
                        pairs=40, VB=0)]
                for fw in fltfw:
                    nt.append(FlexiTPNet(wsn, fw=fw, n_exch=70))
                for r, n in enumerate(nt):
                    o['slosu'][k, i, r] = n.n_slots()
                    o['dismi'][k, i, r] = n.dismissed()
                    if r > 0:
                        o['nadv1'][k,i,r-1] = n.nadve
                wsn.mutate_network(c - nnew, nnew)
                var = ('attem', 'expe1', 'expe2', 'laten', 'lates', 'losse')
                for q, nsds in enumerate(sdsl):
                    np.random.seed(k)
                    netc = copy.deepcopy(nt[0])
                    print("Updating with SDS = {0}".format(nsds))
                    netc.adjust_schedule_in_tx_phase(nsds=nsds,sglt=sglt,
                                                     cap=1,mult=8)
                    if q == 0:
                        o['natre'][k, i] = netc.natre
                    for va in var:
                        o[va][k,i,q] = netc.record[va]
                    o['energ'][k, i, q] = netc.incorp_ener()
                    o['sloov'][k, i, q] = (c * netc.nsds * netc.n_frames()
                                           + 2 * netc.record['losse'])
                for (u, nx) in enumerate(nt[1:]):
                    np.random.seed(k)
                    nx.adjust_schedule_in_tx_phase()
                    o['nadv2'][k,i,u] = nx.nadve 
                    o['sloov'][k,i,len(sdsl)+u] = c*nx.n_frames() * nx.nadve
                    o['laten'][k,i,len(sdsl)+u] = nx.record['laten']
                    o['lates'][k,i,len(sdsl)+u] = nx.record['lates']
                    expe12 = nx.expe12()
                    o['expe1'][k,i,len(sdsl)+u] = expe12[0]
                    o['expe2'][k,i,len(sdsl)+u] = expe12[1]
         # wsn.plot_tree()
        savedict(**o)
    r = load_npz()
    # Plot schedule length. Takeaway: FlexiTP requires more slots when the
    # network changes.
    x_t = r'node density $\bar{\rho}$'
    legsu = ['ACSP', 'Flt fr=2', 'Flt fr=3']
    legsds = ['SDS = {0}'.format(i) for i in sdsl]
    leg2 = ['$\\rho = {0}$'.format(rho_v[i]) for i in ind_rho_table]
    legflt = ["Flt fltfw = {0}".format(f) for f in fltfw]
    g = Pgf(extra_preamble='\\usepackage{plotjour1}\n')
    # Plot the packets destroyed by incorporations in ACSP
    g.add(x_t, r'attem: attempts per natre')
    g.mplot(rho_v, r['attem'], legsds)
    g.add(x_t, r'dismi')
    g.mplot(rho_v, r['dismi'], legsu)
    g.add(x_t, r'energ: per natreq (mJ)')
    g.mplot(rho_v, r['energ'] * 1e3, legsds)
    g.add(x_t, r'expe1: expelled  per natreq')
    g.mplot(rho_v, r['expe1'], legsds + legflt)
    g.add(x_t, 'expe1 + expe2')
    g.mplot(rho_v, r['expe1'] + r['expe2'], legsds + legflt)
    g.add(x_t, r'expe2/expe1')
    quot = r['expe2'] / r['expe1'] # some elements may be NaN (divide by 0)
    quot = np.where(np.isnan(quot), 0, quot) # Replace NaN by 0
    g.mplot(rho_v, quot, legsds + legflt)
    #g.opt(r'legend style={at={(1.02,0.5)}, anchor=west }')
    g.add(x_t, r'laten: per natreq') 
    g.mplot(rho_v, r['laten'], legsds + legflt)
    g.add(x_t, r'losse: packets ruined per incorporation in ACSP')
    g.mplot(rho_v, r['losse'], legsds)
    g.plot(rho_v, np.zeros(len(rho_v)), 'FlexiTP')
    g.opt(r'ylabel style={yshift = 3mm}')
    g.add(x_t, r"nadv1: slots per exchange in FlexiTP's setup")
    g.mplot(rho_v, r['nadv1'], legflt)
    g.add(x_t, r"nadv2: slots per exchange in FlexiTP's tx")
    g.mplot(rho_v, r['nadv2'], legflt)
    g.add(x_t, "natre: naturally required slots")
    g.plot(rho_v, r['natre'])
    g.add(x_t, "sloov: overhead in slots")
    g.mplot(rho_v, r['sloov'], legsds + legflt)
    g.add(x_t, "slosu: number of slot in the setup phase")
    g.mplot(rho_v, r['slosu'], legsu)
    ######################################
    # Postponement
    postpone = np.zeros((len(rho_v), len(sdsl+2)))
    for i, rho in enumerate(rho_v):
        for j, k in enumerate(0, 2): 

            postpone[i,j] = ((r['losse'][i,k]+r['lates'][i,k])
                             / Ts / r['pkets'][i])
        postpone[i,2] = (r['lates'][i,3] * r['laten'][i,3] / r['laten'][i,0] 
                         / Ts / r['pkets'][i])
        postpone[i,3] = (r['lates'][i,3] + r['laten'][i,3] / r['laten'][i,2]
                         / Ts / r['pkets'][i])
    g.add(x_t, "postponements")
    g.mplot(rho_v, postpone) 
    ############## Normalized gain as a function of the T_s
    # Energy consumed by FlexiTP
    eflex = np.zeros((len(rho_v), len(fltfw)))
    E_v_sutp_0 = np.zeros((len(rho_v), len(sdsl)))
    for i in xrange(len(rho_v)):
        for j in xrange(len(fltfw)):
            eflex[i, j] = STATES['tx'] * 1000 * r['nadv2'][i,j] * slot_t
        for j in xrange(len(sdsl)):
            E_v_sutp_0[i, j] = r['energ'][i,j] * 1000.
    print("eflex = {0}".format(eflex))
    E_f_sutp =  STATES['rx'] * 1000 * sglt * 1
    Ts = np.arange(5, 20, 2) # frames between natreq slots
    Gbar = np.zeros((len(Ts), len(rho_v), len(sdsl), len(fltfw)))
    for s, sds in enumerate(sdsl):
        for i, rho in enumerate(rho_v):
            E_t_sutp = (E_f_sutp + E_v_sutp_0[i, s] / Ts / n_nodes[i])
            print("E_t_sutp = {0}".format(E_t_sutp))
            for h, fw in enumerate(fltfw):
                x = (eflex[i,h] / r['laten'][i,s]
                     * r['laten'][i,len(sdsl) + h])
                if x < 0.1:
                    pdb.set_trace()
                Gbar[:, i, s, h] = x / E_t_sutp
    fwi = 0
    sdsi = 0
    g.add('Period between changes $T_s$', 
          'barG FlexiTP{0}/SDS{1}'.format(fltfw[fwi],sdsl[sdsi]))
    g.mplot(Ts, Gbar[:,ind_rho_table,0,0], leg2)
    #################
    g.extra_body.append(r"""
\begin{verbatim}
""")
    # E_f is the fixed energy consumption, i.e. the energy consumed even
    # when no node has the need to obtain a slot.  E_f is the fixed energy
    # per node.
    g.extra_body.append("|%15s|%6s|%15s|%15s|\n" %("Protocol", "rho",
          "E_f", "E_g"))
    for j, fw in enumerate(fltfw):
        for i in ind_rho_table:#index of node density
            g.extra_body.append("|%15s|%6d|%15f|%15f|\n" %
                                (("", "FlexiTP{0}".format(fw))[not i], 
                                 rho_v[i], eflex[i, j], 0))
    for nsds in sdsl:
        for i, j in enumerate(rho_v):#index of node density
            g.extra_body.append("|%15s|%6d|%15f|%15f|\n" %
             (("", "SDS=%d$" % nsds)[not i],
              rho_v[i],
              STATES['rx'] * 1000 * sglt * (nsds+1),
              r['energ'][i,nsds] * 1000.))
    g.extra_body.append(r"""
\end{verbatim}
""")
    g.save(plot=plot)
    #nw.adjust_schedule_in_tx_phase(until=200)
def debugGraphFlexiSds():
    """Dependence on the number of SDS tones. 
    tsn_nr:seconds per iteration, 1:2700@ee-modalap
    """
    bitrate = 19.2e3
    packet_size = 56 * 8 # bits
    slot_t = packet_size / bitrate
    sglt = 20.0 / bitrate # 10 bits
    x = 3 * tx_rg1
    nnew = 2 # Number of new nodes to add 
    c = 20
    k = 962
    print_nodes(c, 962)
    wsn = PhyNet(c=c, x=x, y=x, **net_par1)
    net1 = ACSPNet(wsn, cont_f=100., Q=1/bitrate, slot_t=slot_t,
            pairs=40, VB=0)
    net2 = FlexiTPNet(wsn, fw=2, n_exch=70)
    wsn.mutate(c - nnew, nnew)
    np.random.seed(k)
    print("*******Beginning schedule update*************")
    # net1.VB = 1
    net1.adjust_schedule_in_tx_phase(nsds=0, sglt=sglt, cap=1, mult=8)
def testFlexiSds():
    """Dependence on the number of channel change cycles. """
    x, y = np.array([3,3]) * tx_rg1
    c = int(np.round(7 * x * y / np.pi / tx_rg1**2))
    sds = 0, 1, 2, 3
    nnew = 2 # Number of new nodes to replace 
    print_nodes(c, 7)
    wsn = PhyNet(c=c, x=x, y=y, **net_par1)
    net = ACSPNet(wsn, Q=0.01, cont_f=100, pairs=40, VB=0)
    wsn.mutate(c - nnew, nnew)
    np.random.seed(7)
    netc = copy.deepcopy(net)
    print("Updating with SDS = {0}".format(0))
    netc.adjust_schedule_in_tx_phase(nsds=0, cap=1, mult=8)
def test_flexitp_update():
    nw = test_net3()
    print(nw.f)
    f = FlexiTPNet(nw, VB=True, until=500)
    nw.mutate(2, 1)
    print(nw.f)
    f.adjust_schedule_in_tx_phase()
    f.print_dicts()
def test_pgf2():
    x = np.arange(0, 2 * np.pi)
    g1 = pgfplot(x, np.sin(x), lgnd='sin')
    g2 = pgfplot(x, np.cos(x), lgnd='cos')
    pgfsave(2, 1, g1, g2)
net_par1 = dict(tx_p=1e-6, BW=256e3, sinr=20, d0=100, PL0=1e8, p_exp=3.5,
               shadow=8.0)
tx_rg1 = PhyNet(**net_par1).tx_range
class LossNode(list):
    def __init__(z, id):
        z.id = id
    def __repr__(z):
        z.sort(key=lambda x: x[0])
        return "Node %d: %s;" % (z.id, list.__repr__(z))
    def ret_post_order(z, node_list):
        for n in z.ch:
            n.ret_post_order(node_list)
        node_list.append(z)
    def print(z):
        print(z)
class LossTree(list):
    def __init__(z, fv, ps, size, VB=False):
        z.size = size
        z.VB = VB
        z[:] = [LossNode(i) for i in xrange(len(fv))]
        for i, n in enumerate(z):
            n.f = z[fv[i]] if i else -1
            n.ch = [z[j] for j, k in enumerate(fv) if k==i]
            n.ps = ps[i]
            n.ancestors = [] # includes the sink
            n.subtree = []
        for n in z[1:]:
            ancestor = n.f
            while ancestor != -1:
                ancestor.subtree.append(n)
                n.ancestors.append(ancestor)
                ancestor = ancestor.f
        z.postorder_list = [] # does not include the sink
        for n in z[0].ch:
            n.ret_post_order(z.postorder_list)
    def prnt(z, *args):
        if z.VB:
            print(*args)
    def _discard(z, n, type, rate):
        if n.id:
            if type < 4:
                discard_type = (0, 0, 1, 2)[type]
            elif rate < min(x.ps for x in [n] + n.ancestors[:-1]) * 1.1:
                discard_type = 0
            else:
                discard_type = 1
            f1 = lambda x:(x[0] % len(z.count) in n.gen, x[0])
            n.sort(key=[itemgetter(0), itemgetter(1,0), f1][discard_type])
            n[0:-z.size] = []
    def optimum(z):
        """Maximum data rate transmittable assuming infinite buffers and
        perfect node synchronization."""
        return sum(min(q.ps for q in [n] + n.ancestors) for n in z[1:])
    def simulate_it(z, iterations, rate, type):
        """ Implementation wherein the packet from the oldest sensing time
        is transmitted.

        rate is a number between 0 and 1. """
        assert type in range(5)
        z.iterations = iterations
        old = -1
        # if type == 3:
        #     rate = float(len(z.count)) / z.frames
        #    pdb.set_trace()
        for frame in xrange(iterations):
            for old in xrange(old + 1, int(frame * rate) + 1):
                for n in z[1:]:
                    n.append((old, 1))
                    z._discard(n, type, rate)
            z.prnt("Starting frame %d" %frame)
            for n in z.postorder_list:
                if len(n) and np.random.rand() < n.ps: # Success
                    if type < 4:
                        select_type = (0, 1, 1, 2) [type]
                    elif type == 4:
                        if (rate < min(x.ps for x in [n] + n.ancestors[:-1])
                            * 1.1):
                            select_type = 0
                        else:
                            select_type = 1
                    if select_type == 0:
                        n.sort(key=itemgetter(0))
                    elif select_type == 1:
                        n.sort(key=itemgetter(0))
                        n.sort(key=itemgetter(1), reverse=True)
                    elif select_type == 2:
                        n.sort(key=lambda x: (x[0] % len(z.count) not in
                            n.gen, x[0]))
                       # if len(n) > 4:
                       #     pdb.set_trace()
                    t, k = n.pop(0)
                    z.prnt("Node %d tx (%d, %d) to %d" %(n.id, t, k, n.f.id))
                    d = dict(n.f[:])
                    d[t] = d.get(t, 0) + k
                    n.f[:] = d.items()
                    z._discard(n.f, type, rate)
        d = dict(z[0])
        z.results = np.array([d.get(i, 0) for i in xrange(old + 1)])
    def add_frame(z, node, frame):
        assert node.q > 0
        node.q -= 1
        node.gen.append(frame)
        if frame >= len(z.count):
            z.count.append(0)
        z.count[frame] += 1
        z.prnt("Node %d gained frame %d" % (node.id, frame))
    def find_schedule(z, frames, source_min):
        """ Stamp the nodes with the list gen of sampling.

        Input:
        + frames: number of frames per hyperframe
        + source_min: the minimum number of sources per sensing period
        Output:
        + node.gen: list of slots for which node generates data

        The quality of the schedule is given by:
          
          max(n.gen for n in z)

        """
        z.frames = frames
        z.count = []
        for n in z[1:]:
            n.q = min(int(a.ps * frames) for a in [n]+n.ancestors)
            n.gen = []
        for frame in xrange(9999):
            tree_list = []
            for n in z[1:]:
                tree = [n] + n.ancestors[:-1]
                if not all(x.q > 0 for x in tree):
                    break
                stack = collections.deque(n.ch) # for preorder traversal
                while stack and len(tree) < source_min:
                    x = stack.pop()
                    if x.q > 0:
                        tree.append(x)
                        stack.extend(x.ch)
                if len(tree) >= source_min:
                    tree_list.append(tree)
            if len(tree_list) < 1:
                break
            tree = max(tree_list, key=lambda x: len(x[0].ancestors))
            z.prnt("***Subtree of node %d selected" % tree[0].id)
            for n in tree:
                z.add_frame(n, frame)
        else:
            raise Error,  "too many frames generated"
        assert len(z.count) > 0, "source_min is too small"
        # The number of frames is now fixed.  If possible, use remaining q's
        # to increase the counts.
        z[0].gen = range(len(z.count))
        stack = collections.deque(z[0].ch[:])
        while stack:
            x = stack.pop()
            stack.extend(x.ch)
            available = list(set(x.f.gen) - set(x.gen))
            while x.q and available:
                available.sort(key=lambda x: z.count[x])
                z.add_frame(x, available.pop(0))
                available = list(set(x.f.gen) - set(x.gen))
        assert not any(x.q for x in z[1:])
        z.show_schedule()
    def show_schedule(z):
        z.prnt("****Show computation results****")
        z.prnt([i.q for i in z[1:]])
        for n in z[1:]: 
            n.gen.sort()
            z.prnt("%d: %s; q=%d" % (n.id, n.gen, n.q))
def test_rate2():
    fv = [-1, 0, 1, 1, 1 , 1]
    ps = [1, 0.2, 0.5, 0.5, 0.5, 0.5]
    size = 30
    repetitions = 2000
    rate_v = np.arange(1, 0.1, -0.1)[-1::-1]
    for type in (0, 1, 2, 3):
        print("****** Executing %d" % type)
        print("%8s %8s %8s %8s" % ("rate", "mean", "std", "sum"))
        for i, rate in enumerate(rate_v):
            np.random.seed(0)
            t = LossTree(fv, ps, size)
            if type == 3:
                t.find_schedule(50, 3)
            t.simulate_it(repetitions, rate, type)
            print("%8s %8.2f %8.2f %8.2f" % (rate, t.results.mean(),
                t.results.std(), t.results.sum()))
    opt = t.optimum() 
    print("The optimum fraction of packets is %8.2f" %opt) 
    print("The optimal sum is %8.2f" % (opt * repetitions)) 
    print("The max without failures is %d" % ((len(fv)-1) * repetitions)) 
    # Now compute the optimum
def graphRate1(tst_nr=0, repetitions=2, action=0, plot=0):
    """ Some plots

    plot: 0=do not compile, 1: compile, 2: compile and display
    """
    tst_nr = int(tst_nr)
    repetitions = int(repetitions)
    action = int(action)
    plot = int(plot)
    if tst_nr == 0:
        # file:/home/ornediaz/py1/graphRate1_00_000010.pdf
        # T0 >= T1 = T2
        # At its peak, T3 is the best in both sum and pmin
        ps = [1, 0.8, 0.4, 0.4, 0.4, 0.4]
        # The optimum rate for type=3 is 0.75

        # In order to see the advantage of the scheduled approach, there must
        # be some node with two children with lower success probability than
        # itself.
        fv = [-1, 0, 1, 1, 1, 1]
        frames=8
    elif tst_nr == 1:
        # file:/home/ornediaz/py1/graphRate1_01_000010.pdf
        # file:/home/ornediaz/py1/graphRate1_01_000010.tex
        # Using a homogeneous success probability the advantage of type 3
        # cannot be seen.  T0 = T3 >> {T1, T2}
        ps = [1, 0.4, 0.4, 0.4, 0.4, 0.4]
        fv = [-1, 0, 1, 1, 1 , 1]
        frames = 8
    elif tst_nr == 2:
        # file:/home/ornediaz/py1/graphRate1_02_000010.pdf
        ps = [1, 0.4, 0.8, 0.8, 0.8, 0.8]
        fv = [-1, 0, 1, 1, 1 , 1]
        frames = 8
    elif tst_nr == 3:
        # file:/home/ornediaz/py1/graphRate1_03_000010.1pdf
        # file:/home/ornediaz/py1/graphRate1_03_000010.tex
        # Type 
        ps = [1, 0.6, 0.6, 0.6, 0.2, 0.2]
        # Using a homogeneous success probability the advantage of type 3
        # cannot be seen.
        fv = [-1, 0, 1, 1, 1 , 1]
        frames = 10
    elif tst_nr == 4:
        # file:/home/ornediaz/py1/graphRate1_04_000010.pdf
        # file:/home/ornediaz/py1/graphRate1_04_000010.tex
        ps = [1, 0.3, 0.6, 0.6, 0.2, 0.2]
        fv = [-1, 0, 1, 1, 1 , 1]
        frames = 10
    elif tst_nr == 5:
        # file:/home/ornediaz/py1/graphRate1_05_000010.pdf
        # file:/home/ornediaz/py1/graphRate1_05_000010.tex

        # This test case highlights a situation wherein discard_type=1 (type
        # 2).  It achieves a significant advantage over the other types
        # (except 3).

        fv = [-1, 0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5]
        ps = np.ones(12) * 0.3
        frames = 10
    elif tst_nr == 6:
        # file:/home/ornediaz/py1/graphRate1_06_000010.pdf
        # file:/home/ornediaz/py1/graphRate1_06_000010.tex

        # This test case highlights a situation wherein discard_type=1 (type
        # 2) achieves a significant advantage over the other types
        # (except 3).

        fv = [-1, 0, 1, 2, 3, 4, 5]
        #plot_logical(fv)
        ps = np.ones(7) * 0.3
        frames = 10
    elif tst_nr == 7:
        # file:/home/ornediaz/py1/graphRate1_07_000010.pdf
        # file:/home/ornediaz/py1/graphRate1_07_000010.tex
        fv = [-1, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
        ps = np.ones(11) * 0.8
        frames = 10
    elif tst_nr == 8:
        # file:/home/ornediaz/py1/graphRate1_08_000010.pdf
        # file:/home/ornediaz/py1/graphRate1_08_000010.tex
        fv = [-1, 0, 0, 1, 1, 1, 2, 2, 2]
        ps = np.ones(11) * 0.8
        frames = 10
    elif tst_nr == 9:
        # file:/home/ornediaz/py1/graphRate1_09_000010.pdf
        # file:/home/ornediaz/py1/graphRate1_09_000001.tex
        fv = [-1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 9, 10, 11, 12] 
        ps = [0.8 if i < 9 else 0.4 for i in xrange(len(fv))]
        frames = 30
    elif tst_nr == 10:
        # file:/home/ornediaz/py1/graphRate1_10_000010.pdf
        # file:/home/ornediaz/py1/graphRate1_10_000001.tex
        fv = [-1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 9, 10, 11, 12] 
        ps = [0.4 if i < 9 else 0.8 for i in xrange(len(fv))]
        frames = 30
    elif tst_nr == 11:
        # file:/home/ornediaz/py1/graphRate1_10_000010.pdf
        # file:/home/ornediaz/py1/graphRate1_10_000001.tex
        fv = [-1, 0, 1, 1, 1, 0, 5, 6, 7, 8] 
        ps = [0.4 if i < 5 else 0.8 for i in xrange(len(fv))]
        frames = 30
    else:
        raise Error("tst_nr is invalid")
    rate_v = np.arange(1.5, 0.1, -0.05)[-1::-1]
    plot_logical3(fv, ps, 'pdf', plot)
    size = 30
    types = (0, 1, 2, 3, 4)
    metrics = 'mean', 'std','sum'
    iterations = 2000
    mean = np.zeros((repetitions, len(rate_v), len(types)))
    std = np.zeros((repetitions, len(rate_v), len(types)))
    sum = np.zeros((repetitions, len(rate_v), len(types)))
    pmin = np.zeros((repetitions, len(rate_v), len(types)))
    threshold = 3
    if action == 1:
        for k in xrange(repetitions):
            print_iter(k,repetitions)
            for j, rate in enumerate(rate_v):
                for i, type in enumerate(types):
                    np.random.seed(k)
                    t = LossTree(fv, ps, size)
                    if type == 3:
                        t.find_schedule(frames, threshold)
                        if k == j == 0:
                            opt = np.array(float(len(t.count))/t.frames)
                    t.simulate_it(iterations, rate, type)
                    mean[k, j, i] = t.results.mean() 
                    std[k, j, i] = t.results.std() / float(iterations)
                    sum[k, j, i] = t.results.sum() / float(iterations)
                    pmin[k,j,i] = (t.results >= threshold).mean()
        save_npz('mean', 'std', 'sum', 'pmin', 'opt')
    r = load_npz()
    print("================")
    leg = ('0', '1', '2', '3=%f' %r['opt'], '4')
    g = Pgf(extra_preamble='\\usepackage{plotjour1}\n')
    g.add('rate', r'total') 
    g.mplot(rate_v, r['sum'], leg)
    g.add('rate', r'mean') 
    g.mplot(rate_v, r['mean'], leg)
    g.add('rate', r'std') 
    g.mplot(rate_v, r['std'], leg)
    g.add('rate', r'pmin') 
    g.mplot(rate_v, r['pmin'], leg) 
    g.extra_body.append('\n\includegraphics[scale=0.4]{ztree.pdf}\n')
    g.save(plot=plot)
    t = LossTree(fv, ps, size)
    t.find_schedule(8, threshold)
def gitls(content=''):
    l1 = subprocess.Popen(['git', 'ls-files'],
            stdout=subprocess.PIPE).communicate()[0].split()
    return [i for i in l1 if content in i]
def gitrm(fname):
    if type(fname) is not list:
        fname = [fname]
    subprocess.call(['git', 'rm'] + fname)
def rate1_clean():
    l2 = gitls('graphRate1' in i)
    print(l2)
    subprocess.call(['git', 'rm', f])
def test_rate3():
    fv = [-1, 0, 1, 1, 1 , 1]
    ps = np.ones(len(fv)) * 0.3
    size = 50
    repetitions = 2000
    rate_v = np.arange(0.05, 1.0, 0.1)
    d = collections.namedtuple('d', 'mean', 'std', 'sum')
    types = (0, 1, 2)
    d1 = dict((t, d(*[np.zeros(len(rate_v)) if i else rate_v for i in
                xrange(len(d._fields))])) for t in types)
    for t in types:
        print("****** Executing %d" % t)

        print("%8s %8s %8s %8s" % ("rate", "mean", "std", "sum"))
        for i, rate in enumerate(rate_v):
            np.random.seed(0)
            t = LossTree(fv, ps, size)
            t.simulate_it(repetitions, rate, type)
            print("%8s %8.2f %8.2f %8.2f" % (rate, t.results.mean(),
                t.results.std(), t.results.sum()))
    opt = t.optimum() 
    print("The optimum fraction of packets is %8.2f" %opt) 
    print("The optimal sum is %8.2f" % (opt * repetitions)) 
    print("The max without failures is %d" % ((len(fv)-1) * repetitions)) 
def test_find_schedule(test, plot):
    if test == 0:
        fv = [-1, 0, 1, 1, 1 , 1]
        if plot: plot_logical(fv)
        ps = np.ones(len(fv)) * 0.8
        t = LossTree(fv, ps, size=50, VB=True)
        t.find_schedule(frames=4, source_min=2)
    if test == 1:
        fv = [-1, 0, 0, 1, 1 , 2, 2, 3, 3]
        if plot: plot_logical(fv)
        ps = np.ones(len(fv)) * 0.3
        t = LossTree(fv, ps, size=50, VB=True)
        t.find_schedule(frames=9, source_min=2)
    if test == 3:
        fv = [-1, 0, 0, 1, 1 , 2, 2, 3, 3]
        if plot: plot_logical(fv)
        ps = [1, 0.5, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2]
        t = LossTree(fv, ps, size=50, VB=True)
        t.find_schedule(frames=10, source_min=2)
    if test == 4:
        fv = [-1, 0, 0, 1, 1 , 2, 2, 3, 3]
        if plot: plot_logical(fv)
        ps = [1, 0.5, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2]
        size = 50
        t = LossTree(fv, ps, size, VB=True)
        t.find_schedule(frames=10, source_min=2)
        t.simulate_it(iterations=10, rate=1, type=2)
        print(t.results)
def test_round5():
    n_frames = 10
    fv = [-1, 0, 0, 1, 1, 2, 2, 3, 3]
    ps = [1, 0.55, 0.35, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
    t = LossTree(fv, ps, 3)
    t.round5(n_frames, n_hyperframes=10, source_min=2)
def test_opt():
    t = LossTree([-1, 0, 0, 1, 1], [1, 0.3, 0.3, 0.1, 0.2], 3)
    print(t.optimum())
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
def g():
    print(locals())
def run(*args):
    # Remove the files that do not correspond to any graph function.
    lst = []
    for a in args:
        try:
            lst.append(int(a))
        except ValueError:
            try:
                lst.append(float(a))
            except ValueError:
                lst.append(a)
    print("Executing ", lst)
    print("Execution started on {0}.".format(time_module.asctime()))
    global start_time
    start_time = time_module.time()
    try:
        globals()[lst[0]](*lst[1:])
    finally:
        print("Execution ended on {0}.".format(time_module.asctime()))
        duration = time_module.time() - start_time
        print("Execution lasted for {0} seconds.".format(duration))
for file_name in glob.glob('graph*'):
    for fun in dir():
        if fun == file_name.partition('_')[0]:
            break
    else:
        print('{0} to be removed'.format(file_name))
        os.remove(file_name)
if __name__ == '__main__':
    # g()
    # Execute function requested in the command line
    if len(sys.argv) > 1:
        run(*sys.argv[1:])
    #doctest.testmod()
    pass
