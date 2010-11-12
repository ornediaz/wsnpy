''' Probability of timely delivery using the FAT method.

'''
import sys
import numpy as ny
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
# Import ny last so that I have access to the savez and load function
# from ny. plt has a different load function that I want to override.

Infi = 9999
def dijk(P,funLength = None):
   ''' Return next hop towards sink computed using Dijktra's algorithm. 

   Arguments:
   P -- Nx2 ndarray indicating location of N nodes. P[0] refers to sink
   funLength(P,x,y) -- function returning the distance from x to y

   '''
   if funLength == None:
      def funLength(P,x,y):
         # If we raise the norm to third power we find a different tree
         return sum((P[x]-P[y])**2)
   N = len(P)
   dst = ny.ones(N) * Infi # Each node's smallest distance to the sink
   dst[0] = 0 # The source is at distance 0 from itself
   previous = -ny.ones(N,dtype='int32') # parent of each node
   processed = ny.zeros(N,dtype=bool)
   # If any node does not have neighbors, it will never be processed.
   while True:
      x = (dst + processed * Infi * 2 ).argmin()
      if processed[x]: break
      processed[x] = True
      for y in ny.r_[0:x,x+1:N]:
         alt = dst[x] + funLength(P,x,y) 
         if alt < Infi and alt < dst[y]:
            dst[y] = alt
            previous[y] = x
   return previous


class WSN:
   ''' WSN containing node location, neighbors and the FAT protocol.'''
   def __init__(z,n=100,x=200,y=200,txRg=50,ixRg=150):
      z.n = n # number of nodes
      z.x = x # Length of monitored area
      z.y = y # Height of monitored area
      z.txRg = txRg
      z.ixRg = ixRg
      z.p = ny.random.rand(z.n,2)*[z.x,z.y] # Location of the nodes
      z.p[0] = [0,y/2] # The sink is located in the southwest corner
      def funLength(P,x,y):
         '''Return the communication cost between two nodes.'''
         dist = sum((P[x]-P[y])**2)**.5
         if dist < txRg:
            return 1
         else:
            return Infi
      z.funLength = funLength
      z.f = dijk(z.p,z.funLength)
      z.ch = [[] for i in xrange(z.n)] # All nodes initially childless
      # Compute the tier number of each node.
      z.tr = ny.ones(z.n,dtype='int32')*99   
      for i in xrange(z.n):
         p = i
         n = 0
         while p != -1 and p!=0: # 0 is the sink
            n = n+1
            p = z.f[p]
         if p == 0: # If p==-1, the node cannot reach the gateway
            z.tr[i] = n # Sentinel value
      # Compute neighbors within tx & ix range & nodes in the upper tier.
      # Initialize list of lists of neighbors within TX range.
      z.txL = [[] for i in xrange(z.n)] 
      # Initialize list of lists neighbors within IX but TX range.
      z.ixL = [[] for i in xrange(z.n)] 
      # Initialize list of lists of neighbors in the next tier.
      z.up = [[] for i in xrange(z.n)] 
      for i in xrange(z.n):
         for j in ny.r_[0:i,i+1:z.n]:
            di = funLength(z.p,i,j)
            if di < txRg:
               z.txL[i].append(j)
               if z.tr[j] == z.tr[i] -1:
                  z.up[i].append(j)
            elif di < ixRg:
               z.ixL[i].append(j)

   def Fat(z,dist,nSrc,working):
      ''' Return fraction of sources enjoying small delays in FAT.

      INPUTS
      dist -- distance of the event center from the sink
      nSrc -- number of sources
      working -- boolean vector indicating whether each node works
      ========Output============
      Array with elements:
       * dumb (# of sources whose default path works)
       * success (# of sources that find a path using the FAT method)
       * connect (# of sources for which a path to the gateway exists)

      ========Discussion
      The method does not simulate packet exchange. It is an iterative
      implementation. As such, it does not estimate T_tier. However, the
      tree constructed should be similar to the one with FAT.
      
      Geometry: The monitored area is a rectangle of length x and height
      y. The origin of coordinates is in the southewestern corner and the
      coordinates of the diagonally opposed corner are (x,y). The sink is
      in (0,y/2) and the event occurs in (d,y/2), where d is the distance
      from the sink.

      '''
      assert sum(z.f > 0) >= nSrc, 'Sources exceed connected nodes'
      # Select the nSrc elements closest to the event center.
      eventLocation = ny.array([dist,z.y/2])
      d = ny.sum(abs(z.p - eventLocation)**2, axis=1)
      src = ny.argsort(d)[0:nSrc] # 'sort' arranges smallest first.
      working[0] = True # Ensure the sink is working
      working[src] = True # Ensure sources are working
      # Initialize each node's number of children, which will be used in
      # choosing a parent.
      z.chN = ny.zeros(z.n, dtype='int32') 
      # Initialize number of sources in complete subtree
      off = ny.zeros(z.n, dtype='int32') 
      # Initialize vector 'actvtd', that will list all the nodes that have
      # looked for children, which are those that received activation
      # tones. I use this list to compute energy consumption at the end
      # of the algorithm.
      actvtd = ny.array([]) 
      # Execute FAT method starting from the tier of the sources further
      # away from the sink and ending in Tier 1.
      for tier in xrange(max(z.tr[src]),0,-1):
         # Determine nodes looking for parent in current tier
         aux = ny.union1d(z.chN.nonzero()[0],src)
         active = aux[z.tr[aux]==tier]
         if VERB:
            print "Nodes active in tier %d: " %tier, active
         ny.random.shuffle(active) 
         for i in active:
            # Simulate the effect of transmitting an activation tone by
            # adding neighbors within TX or IX range in next tier to the
            # 'actvtd' list.
            aux = ny.array(z.txL[i]+z.ixL[i])
            if aux.size == 0:
               continue
            aux = aux[z.tr[aux] == tier-1]
            actvtd = ny.union1d(actvtd,aux)
            # Choose as parent the working node in the next tier with
            # the greatest number of children.
            aux = ny.array(z.up[i])
            if aux.size == 0: 
               continue
            aux = aux[working[aux]]
            if aux.size == 0:
               continue
            f = aux[z.chN[aux].argmax()]    # chosen parent
            z.ch[f].append(i)  # Set itself as a child 
            z.chN[f] += 1
            off[f] += off[i] + (i in src)
      # Execute Dijkstra's algorithm to determine whether a path exist to
      # the gateway. Pass only functional nodes to the method.
      f = dijk(z.p[working],z.funLength)
      # Compute the indices of the sources in the array without
      # dysfunctional nodes.
      src2 = src - ny.cumsum(-working)[src]
      # Compute the number of sources for which, without the tiered
      # architecture constraint, a path towards the gateway exists.
      connected = sum(f[src2] >= 0)
      # off[0] nodes find a path with the FAT method
      success = off[0]
      # Compute the success ratio using the default parent
      dumb = 0
      for u in src:
         while u != 0: # 0 is the sink
            u = z.f[u]
            if not working[u] or u < 0:
               break
         else:
            dumb += 1
      # Handle division by zero error
      print "success = % f" %success
      return ny.array([dumb, success, connected],dtype=float) / nSrc

   def Plot(z):        
      ''' Plot the default tree and label the nodes.'''
      plot(z.p[:,0],z.p[:,1],'o')
      for i,p in enumerate(z.p):
         text(p[0]+z.x*0.02,p[1],i)
      hold(True)
      # Plot the line between each node and its parent. 
      for u,v in enumerate(z.f):
         if v<0: # Ignore orphans (nodes whose parent is v=-1)
            continue
         p = z.p[[u,v]]
         plot(p[:,0],p[:,1])
      axis([0,z.x*1.05,0,z.y*1.05])
      show()
VERB = False

def Average(tstNr,txRg, vDist, rhoV, failP, nSrc,REPEAT):
   # Size of the sensing area square
   x = max(vDist) * txRg 
   y = 4 * txRg
   nV = ny.array((rhoV*x*y/(ny.pi*txRg**2)).round(),dtype='int32') 
   # Inititialize the vector that will contain the results:
   # - [dumb, fat, Dijkstra]
   # - rhoV = node density
   # - vDist: distance from the event to the sink
   # - failP: probability of node failure
   # - REPEAT: repetitions
   r = ny.zeros((3,rhoV.size,vDist.size,failP.size,REPEAT),'float64') 
   ny.random.seed(4)
   for k in xrange(REPEAT):
      print "Iteration %d" %k
      for i,n in enumerate(nV):
         # Try a certain number of times to obtain a reasonably
         # connected network
         for j in xrange(40):
            w = WSN(n=n,x=x,y=y,txRg=txRg,ixRg=2.1*txRg)
            if sum(w.f > 0) > 0.9 * n:
               break
         else:
            raise Exception, 'Insufficient density for connectivity'
         for h, fp in enumerate(failP):
            working = ny.random.rand(n) > fp
            for j, d in enumerate(vDist * txRg):
               print "EXECUTION (%d,%d)" %(i,j)
               r[:,i,j,h,k] = w.Fat(dist=d, nSrc=nSrc, 
                     working=ny.copy(working)) 
   # Save the simulation results into a file
   ny.savez('%s%02d' %(fileName,tstNr), r=r)
if __name__ == "__main__":
   # The first argument shows which part to execute.
   # Depending on the value of tst
   assert sys.argv[1] in ('0','1','2','3','4','5','6','7','8')
   tstNr = int(sys.argv[1])
   action = sys.argv[2]
   fileName = 'fatProb'
   assert action in ('0','1','2','3')

   if tstNr == 0:
      txRg = 60 
      vDist = ny.array([2,6]) #distance to the sink
      rhoV = ny.array([8,14]) # node density: n*pi*txRg^2/x^2
      failP = ny.array([0.05,.2,.5]) # Probability of node failure 
      nSrc = 5
      REPEAT = 4 # Number of Monte Carlo simulations
   elif tstNr == 1: # 7 minutes of execution
      txRg = 60 
      vDist = ny.array([2,6,12]) #distance to the sink
      rhoV = ny.array([7,13,15]) # node density: n*pi*txrg^2/x^2
      failP = ny.array(0.2) # probability of node failure 
      nSrc = 5
      REPEAT = 20
   elif tstNr  == 2: # 28 minutes of execution
      txRg = 60 
      vDist = ny.array([4,6,10]) #distance to the sink
      rhoV = ny.array([7,13,15]) # node density: n*pi*txrg^2/x^2
      failP = ny.array(0.1) # probability of node failure 
      nSrc = 5
      REPEAT = 100
   elif tstNr  == 3: # 6 hours of execution
      txRg = 60 
      vDist = ny.array([4,7,10]) #distance to the sink
      rhoV = ny.array([7,11,15]) # node density: n*pi*txrg^2/x^2
      failP = ny.array([.01,.05,.09,.13,.17]) # probability of node failure 
      nSrc = 5
      REPEAT = 500
   elif tstNr  == 4: # About 16 hours?
      txRg = 60 
      vDist = ny.array([4,7,10]) #distance to the sink
      rhoV = ny.array([7,14,21,28]) # node density: n*pi*txrg^2/x^2
      failP = ny.array([.05,.09,.13,.17]) # probability of node failure 
      nSrc = 5
      REPEAT = 400
   elif tstNr  == 5: # 17 hours of execution
      txRg = 60 
      vDist = ny.array([4,7,10]) #distance to the sink
      rhoV = ny.array([7,14,28]) # node density: n*pi*txrg^2/x^2
      failP = ny.array([.09]) # probability of node failure 
      nSrc = 5
      REPEAT = 1600
   elif tstNr  == 6: # 35.6 hours of execution
      txRg = 60 
      vDist = ny.array([4,7,10]) #distance to the sink
      rhoV = ny.array([7,14,28]) # node density: n*pi*txrg^2/x^2
      failP = ny.array([.05]) # probability of node failure 
      nSrc = 5
      REPEAT = 3200
   if action == '0':

      '''Sample network for which ratio = 0.5

      #THIS CODE IS DEPRECATED
      # 
      #EXPLANATION OF RESULT

      # Nodes 3 and 8 are sources in Tier 2.  z.up[3] = [9] z.up[8] = [4,9]
      # Node 4 is working, whereas node 9 is not. Therefore, the FAT method
      # allows Node 8 but not Node 3 to reach the gateway. Without the
      # tiered architecture, Node 3 can reach the gateway through Node 8.
      # Therefore,ratio = 1/2.
      # 
      '''
      #seed(17)
      #n = 12
      #w = WSN(n=n,x=200,y=200,txRg=120,ixRg=200)
      ##w.Plot()
      #working = rand(n) > .5
      #ratio = w.Fat(dist=150,nSrc=2,working=working)   
      #print "Fraction = %f " % ratio
   elif action == '1':
      '''Create probability of timely delivery graphs with FAT.
    
      Simulate for different node densities and event locations.
      
      '''
      Average(tstNr,txRg,vDist,rhoV,failP,nSrc,REPEAT)
   elif action == '2':
      '''Load and plot simulation results.'''
      d = ny.load('fat_prob%02d.npz' %tstNr)
      for u in d.files:
         exec "%s = d[\"%s\"]" %(u,u)
 
      plt.close('all')
      f1 = plt.figure(1, figsize=(6, 3 * len(failP)))
      frmt = ('rp--','<b-.','>g:','^k-')
      for k, f in enumerate(failP):
         for g, x in enumerate(r):
            s = plt.subplot(len(failP), 3, 1 + g + (3 * k))
            for i, rho in enumerate(rhoV):
               plt.plot(vDist, x.mean(3)[i,:,k], frmt[i], 
                     label=r"$\rho$ = %d" %rho)
            plt.legend(loc='lower left')
            plt.axis([min(vDist),max(vDist),-.1,1.1])
            plt.xlabel('Normalized event distance')
            plt.title('failP = %f' %f)
            if g == 0:
               plt.ylabel('E[timely delivery]')
            elif g == 1:
               s.set_yticklabels([])
      # savefig("%s%02d.eps" %(fileName,tstNr))
      plt.show()
      if tstNr == 6 and True:
         fpIndex = 0 #index of failP to choose
         x = 1 - mean(r,4)[:,:,:,fpIndex] #Type, density, distance
         P, Q = x.shape[1:3] # P number of densities, Q of distances
         W = .8
         D = (1 - W) / 2
         font = { 'fontname':'Times New Roman','fontsize':9}
         font2 = FontProperties(size=7,family='serif')
         f = plt.figure(2, figsize=(3.2,2))
         a = plt.axes([.15,.20,.8,.7])

         for i, g in enumerate(x): # different techniques
            for j, h in enumerate(g): # different densities
               for k, n in enumerate(h): # different distances
                  b, = plt.bar(k * (Q + 1) + j + D,
                        n, width=W, 
                        color=str(float(i)/2))
                  if i == 0:
                     plt.text(k * (Q + 1) + j + .5 , 
                           n + .02,
                           r'$\rho_%d$' %j,
                           ha='center',**font)
                  if j == 0 and k == 0:
                     b.set_label(
                           ('T1: SPT=  using default parent',
                              'T2: FAT\'s normal construction',
                              'T3: FAT\'s backup construction') [i])
         #text(2.9,.55,'$f_p = %0.3f$' %fp, **font)
         #l = legend()
         #l.prop.set_name('Times New Roman')
         #l.prop.set_size(7)
         #l = legend(loc=2,prop=l.prop)
         l = plt.legend(loc=2, prop=font2)
         t = r'$'
         for i, r in enumerate(rhoV):
            t += r'\rho_%d = %d' % (i, r)
            if i != (len(rhoV) - 1):
               t += ', '
         t += '$'
         plt.text(0,.52,t,**font)
         #for i, r in enumerate(rhoV):
         #   text(8,.75 - i *.08, r'$\rho_%d=%d$' %(i,r),**font)
         plt.text(0,.42,r'$f=%0.3f$' %failP[fpIndex], **font)
         # Plot the hop number
         plt.xticks(arange(P) * (Q + 1) + float(P) / 2, vDist, **font)
         labels = getp(gca(), 'yticklabels')
         setp(labels, **font)#fontsize=7,fontname='Times New Roman')
         setp(labels, fontsize=8)#fontsize=7,fontname='Times New Roman')
         plt.xlabel(r'event-sink distance normalized by TX range: d/r', **font)
      
         plt.ylabel('Prob. not finding path',**font) # (P[Ti], i \in {1,2,3})',**font)
         plt.xlim(-1, P * (Q + 1))
         plt.ylim((0, x.max() * 1.8))
         plt.show()
         plt.close(1)
         plt.savefig('fat_prob%02d.eps' % tstNr, dpi=600)


