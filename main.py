from AlgoAPI import AlgoAPIUtil, AlgoAPI_Backtest

class AlgoEvent:
    
    # Open Jake Liu
    def # to be modified
    
    
    
    
    # High HZ
    def # to be modified
    
    
    
    
    # Low Zhou yAng
    def # to be modified
    
    
    
    # Close KS dog
    def # to be modified
    
    
    
    def __init__(self):
        pass
        
    def start(self, mEvt):
        self.evt = AlgoAPI_Backtest.AlgoEvtHandler(self, mEvt)
        self.evt.start()
        
    def on_bulkdatafeed(self, isSync, bd, ab):
        pass

    def on_marketdatafeed(self, md, ab):
        pass

    def on_newsdatafeed(self, nd):
        pass

    def on_weatherdatafeed(self, wd):
        pass
    
    def on_econsdatafeed(self, ed):
        pass
        
    def on_corpAnnouncement(self, ca):
        pass

    def on_orderfeed(self, of):
        pass

    def on_dailyPLfeed(self, pl):
        pass

    def on_openPositionfeed(self, op, oo, uo):
        pass
