from AlgoAPI import AlgoAPIUtil, AlgoAPI_Backtest

class AlgoEvent:
    
    # Open Price
    def getOpenPrice(self, instrument, days, endtime):
        contract = {"instrument": instrument}
        res = self.evt.getHistoricalBar(contract, days, 'D', endtime)
        openprices = []
        for t in res:
            timestamp = t
            lastprice = res[t]['o']
            openprices.append((timestamp, lastprice))
        return openprices
    
    # High Price
    def getHighPrice(self, instrument, days, endtime):
        contract = {"instrument": instrument}
        res = self.evt.getHistoricalBar(contract, days, 'D', endtime)
        highprices = []
        for t in res:
            timestamp = t
            lastprice = res[t]['h']
            highprices.append((timestamp, lastprice))
        return highprices
    
    # Low Price
    def getLowPrice(self, instrument, days, endtime):
        contract = {"instrument": instrument}
        res = self.evt.getHistoricalBar(contract, days, 'D', endtime)
        lowprices = []
        for t in res:
            timestamp = t
            lastprice = res[t]['l']
            lowprices.append((timestamp, lastprice))
        return lowprices
    
    # Close Price
    def getClosePrice(self, instrument, days, endtime):
        contract = {"instrument": instrument}
        res = self.evt.getHistoricalBar(contract, days, 'D', endtime)
        closeprices = []
        for t in res:
            timestamp = t
            lastprice = res[t]['c']
            closeprices.append((timestamp, lastprice))
        return closeprices
    
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
