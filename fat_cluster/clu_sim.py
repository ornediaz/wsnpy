''' Test the algorithm clusterize of net_sinr.

Simulation of my cluster-based topology based on simulations rather than my
formulas.

'''
import numpy as np
import matplotlib.pyplot as plt
import net_sinr as net

INF = 9e9

class UnsufficientDensity(Exception): pass

class IncorrectError(Exception): pass


