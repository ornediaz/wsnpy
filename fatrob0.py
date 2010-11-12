''' Compute and plot the node failure tolerance of the iterative
implementation of the FAT method.

Arguments to pass:
1: tst_nr
2: 'compute' or 'plot'          

'''
import sys
import numpy as ny
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pdb
import net

tst_nr = int(sys.argv[1])
action = sys.argv[2]
assert action in ('compute', 'plot')

file_name = "%s_%02d" %(__file__[0:__file__.rfind('.')],tst_nr)
outputdir = ('./', 'c:/root/latex/fat/')

def save(figure_number):
    f = plt.figure(figure_number)
    for o in outputdir:
        f.savefig('%s%s_%d.pdf' %(o, file_name, figure_number))

# tst_nr chooses simulation parameters. The execution times are 1:7',
   # 2:28', 3:7h, 4:16h, 5:17h, 6:35.6h
# Select simulation parameters
tx_rg = 60
# Number of Monte Carlo simulations
REPEAT = [4, 20, 100, 500, 400, 1600, 3200][tst_nr]
# Distance to the sink normalized by transmission range
vDist = ny.array([[2,6], [2,6,12], [4,6,10], [4,7,10], [4,7,10],
       [4,7,10], [4,7,10]][tst_nr])
# node density=c*pi*tx_rg^2/x/y. Greatly increases the execution time.
rhoV = ny.array([(8,14), (7,13,15), (7,13,15), (7,11,15),
   (7,14,21,28), (7,14,28), (7,14,28)][tst_nr])
failP = ny.array([(.05,.2,.5), (.2,), (.1,), (.01,.05,.09,.13,.17), (.09,), (.05), (.05,)][tst_nr])


if action == 'compute':
    # Compute and store the probability of timely delivery with FAT for
    # different node densities and event locations.
    x = max(vDist) * tx_rg # Width of the sensing area square
    y = 4 * tx_rg # Height of the sensing area
    # Compute number of nodes based on the area and the node density.
    nV = ny.array((rhoV * x * y / ny.pi / tx_rg **2).round(), int)
    # Initialize the vector that will contain the results:
    r = ny.zeros((3, rhoV.size, vDist.size, failP.size, REPEAT)) 
    ny.random.seed(4)
    for k in xrange(REPEAT):
        print "Iteration %d" %k
        for i,c in enumerate(nV):
            # Try a certain number of times to obtain a reasonably
            # connected network
            w = WSN(c=c, x=x, y=y, tx_rg=tx_rg, ixRg=2.1*tx_rg)
            for h, fp in enumerate(failP):
                working = ny.random.rand(c) > fp
                for j, d in enumerate(vDist * tx_rg):
                    print "EXECUTION (%d,%d)" %(i,j)
                    src = Src(p=z.p, f=z.f, loc=[x,y/2], nSrc=nSrc)
                    r[:,i,j,h,k] = Fat(w, src=src,
                            working=working.copy())[1]
    # Save the simulation results into a file
    ny.savez(file_name + '.npz', r=r)
elif action == 'plot':
    '''Load and plot simulation results.'''
    d = ny.load(file_name + '.npz')
    for u in d.files:
       exec "%s = d[\"%s\"]" %(u,u)
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
    # savefig("%s%02d.eps" %(file_name,tst_nr))
    if tst_nr == 6:
        if True: # new plot
            fpIndex = 0 # index of failP to choose
            x = 1 - r.mean(4)[:,:,:,fpIndex] # Type, density, distance
            P, Q = x.shape[1:3] # P number of densities, Q of distances
            lgd = ["SPT", "FAT", "optimal"]
            g = net.Pgf()
            for i, rho in enumerate(rhoV):
                g.add("normalized event-sink distance, $d/r_t$",
                      "source isolation probability")
                #g.options(title="rho".format(rho))
                tit = "title={{rho={0}}}".format(rho)
                # print tit
                # pdb.set_trace()
                g.opt(tit)
                g.mplot(vDist, x[:, i, :].T, lgd)
            g.save(f_name='fatrob0_06.npz')
            pdb.set_trace()
        else: #old plot
            fpIndex = 0 # index of failP to choose
            x = 1 - r.mean(4)[:,:,:,fpIndex] # Type, density, distance
            P, Q = x.shape[1:3] # P number of densities, Q of distances
            W = .8 # Bar width in bar plot
            D = (1 - W) / 2 # Space between bars in bar plot
            font = { 'fontname':'Times New Roman','fontsize':9}
            font2 = FontProperties(size=10,family='serif')
            f = plt.figure(2, figsize=(3.2,2))
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
               enumerate(rhoV)]) + '; f_0=%0.3f$' %failP[fpIndex]
            ax.text(3., .7, t, **font)
            # Plot the hop number
            t = ny.arange(P) * (Q + 1) + float(P) / 2
            ax.set_xticks(t)
            ax.set_xticklabels(["%d" %i for i in vDist], **font)
            labels = plt.getp(plt.gca(), 'yticklabels')
            plt.setp(labels, **font)
            plt.setp(labels, fontsize=8)
            plt.xlabel(r'event-sink distance normalized by TX range $d/r_t$',
             **font)
            plt.ylabel(r'Prob. not finding path $f$', **font) 
            plt.xlim(-1, P * (Q + 1))
            plt.ylim((0, x.max() * 1.8))
            plt.show()
            plt.close(1)
            save(2)
