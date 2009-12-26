from SimPy.Simulation import *
class Message(Process):
    def go(self): 
        for i in self.go2(): yield i
    def go2(self):
        print now(), "Starting"
        yield hold,self,100.0
        print now(), "In the midle"
        yield hold,self,100.0
        print now(), "Arrived"
initialize()
m = Message()
activate(m,m.go())
simulate(until=200)
print 'Current time is', now() # Will print 106.0
