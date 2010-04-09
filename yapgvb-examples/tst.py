import yapgvb
import subprocess
g = yapgvb.Digraph("tree")
n = [g.add_node(str(i)) for i in xrange(4)]
e = [ i >> j for i, j in zip(n[1:],n[:-1])]
e[0].label = 'cat'
    
# for j in xrange(3):
#     g.add_edge(str(i+1), str(i))
g.layout(yapgvb.engines.dot)
file = 'tree.png'
g.render('tree.png')
subprocess.Popen(['display', file]) 

