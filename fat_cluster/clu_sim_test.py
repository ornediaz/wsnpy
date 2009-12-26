""" A testing routine for CircularNet.

I do not remember what it does and it does not seem to work.

 """

import numpy as np
import clu
import unittest

class TestCircularNet(unittest.TestCase):
    # Visual test to confirm the distribution is correct
    #plt.ioff()
    #fig = plt.figure(1)
    #n_points = 500
    #P = dist(n_points,3,5)
    #
    #ax = fig.add_subplot(1,1,1)
    #ax.plot(P[:,0],P[:,1],'*')
    #plt.show()

    def testsize(self):
        # The output has correct size
        npoints = 30
        P = clu.dist(npoints,3,5,np.pi/2)
        self.assertEqual(P.shape, (npoints,2))
    
    def test_exc(self):
        class Net(clu.CircularNet):
            p = np.array([[2.0, 0.0], [5.0, 0.0]])
        self.assertRaises(
                clu.UnsufficientDensity,
                clu.CircularNet.__init__, Net)
        
class TestClusterize(unittest.TestCase):
    pass

if __name__ == '__main__':
    unittest.main()


