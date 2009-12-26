"""Demonstrate how a SimPy node can interrupt another node."""


import SimPy.Simulation as sim


class Bus(sim.Process):
    
    """Bus that travels until it reaches a destination."""
    
    def operate(self, repairduration, triplength): 
        """Travel until the trip is finished.
        
        The bus may break down down several times during the trip.
        
        Parameters:
        repairduration -- time to recover from a breakdown and continue
        triplength     -- duration of the trip
         
        """
        tripleft = triplength
        # "tripleft"is the driving time to finish trip
        # if there are no further breakdowns
        while tripleft > 0:
            yield sim.hold, self, tripleft # try to finish the trip
            # if a breakdown intervenes
            if self.interrupted():
                print self.interruptCause.name, 'at %s' % sim.now()
                tripleft = self.interruptLeft
                # update driving time to finish
                # the trip if no more breakdowns
                self.interruptReset() 
                # end self interrupted state
                #update next breakdown time
                sim.reactivate(br, delay=repairduration)
                #impose delay for repairs on self
                yield sim.hold, self, repairduration
                print 'Bus repaired at %s' % sim.now()
            else: # no breakdowns intervened, so bus finished trip
                break
        print 'Bus has arrived at %s' % sim.now()
        
        
class Breakdown(sim.Process):
    
    """Creator of periodic interruptions to break down the bus class."""
    
    def __init__(self, myBus):
        sim.Process.__init__(self, name='Breakdown ' + myBus.name)
        self.bus = myBus
        
    def break_bus(self, interval):
        """Interrupt periodically self.myBus."""
        while True:
            #driving time between breakdowns
            yield sim.hold, self, interval 
            if self.bus.terminated():
                break
            # signal "self.bus" to break itself down
            self.interrupt(self.bus)

            
sim.initialize()
bus = Bus('Bus')
sim.activate(bus, bus.operate(repairduration=20, triplength=1000))
#create a breakdown object "br" for bus "bus", and
br = Breakdown(bus)
#activate it with driving time between breakdowns equal to 300
sim.activate(br, br.break_bus(300))
sim.simulate(until=4000)
print 'SimPy: No more events at time %s' % sim.now()
