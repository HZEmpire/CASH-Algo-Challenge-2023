from AlgoAPI import AlgoAPIUtil, AlgoAPI_Backtest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

class AlgoEvent:
    def __init__(self):
        self.last_grid = 0
        self.last_change_grid = [0, 0]
        self.ref = 0
        self.R_prediction = 0
        self.credit = 1

        # Use to store yesturday's status
        self.yesterday = None
        pass

    def getClosePrice(self, instrument, days, endtime):
        contract = {"instrument": instrument}    
        res = self.evt.getHistoricalBar(contract, days, 'D', endtime)
        closeprices = []
        timestamps = []
        for t in res:
            timestamp = t
            lastprice = res[t]['c']
            closeprices.append(lastprice)
            timestamps.append(timestamp)
        return closeprices, timestamps
    
    def doit(self, instrument, buysell, Ref, vol):
        order = AlgoAPIUtil.OrderObject(
            instrument = instrument,
            orderRef = Ref,
            openclose = 'open', 
            buysell = buysell,    #1=buy, -1=sell
            ordertype = 0,  #0=market, 1=limit
            volume = vol
        )
        self.evt.sendOrder(order)

    def start(self, mEvt):
        self.myinstrument = mEvt['subscribeList'][0]
        self.evt = AlgoAPI_Backtest.AlgoEvtHandler(self, mEvt)
        self.evt.start()
        
    def on_bulkdatafeed(self, isSync, bd, ab):
       pass

    def on_marketdatafeed(self, md, ab):
        todayclose, today = self.getClosePrice(self.myinstrument, 2, None)
        # Compare the prediction with the real close price
        self.evt.consoleLog('Rbreaker prediction: ' + str(self.R_prediction))
        self.evt.consoleLog('Real close price: ' + str(todayclose))
        if self.R_prediction == 1 and todayclose[-1] > todayclose[-2]:
            self.evt.consoleLog('True prediction')
            self.credit += 0.01
        elif self.R_prediction == -1 and todayclose[-1] < todayclose[-2]:
            self.evt.consoleLog('True prediction')
            self.credit += 0.01
        elif self.R_prediction == 0:
            # No prediction
            self.evt.consoleLog('No prediction till now')
        else:
            self.evt.consoleLog('False prediction')
            self.credit -= 0.01
        self.evt.consoleLog(' ')
    
        Rsignal = 0
        #stragegy for R-breaker        
        #yesterday = timestamp - timedelta(hours=24)
        #today = timestamp
        contract = {"instrument": self.myinstrument}

        res = self.evt.getHistoricalBar(contract, 3, 'D')
        days = []
        for each in res:
            days.append(each)
            self.evt.consoleLog(str(each))  
        today = days[-1]
        yesterday = days[-2]
        C = res[today]['c']
        #last_C = res[yesterday - timedelta(hours=24)]['c']
        H = res[yesterday]['h']
        L = res[yesterday]['l']
        HT = res[today]['h']
        LT = res[today]['l']
        P = (H + C + L) / 3
        TLP = H + 2 * P - 2 * L
        OSP = P + H - L
        RSP = 2 * P - L
        RLP = 2 * P - H
        OLP = P - (H - L)
        TSP = L - 2 * (H - P)

        # Print corresponding data
        self.evt.consoleLog('TLP: ' + str(TLP))
        self.evt.consoleLog('OSP: ' + str(OSP))
        self.evt.consoleLog('RSP: ' + str(RSP))
        self.evt.consoleLog('RLP: ' + str(RLP))
        self.evt.consoleLog('OLP: ' + str(OLP))
        self.evt.consoleLog('TSP: ' + str(TSP))
        self.evt.consoleLog('C: ' + str(C))

        pos, osOrder, pendOrder = self.evt.getSystemOrders()
        position = pos[self.myinstrument]["netVolume"]
        self.evt.consoleLog('Position: ' + str(position))

        if position == 0:
            if C > TLP:
                #long signal
                Rsignal = 1
            
            elif C < TSP:
                #short signal
                Rsignal = -1

        
        else:
            if position > 0:
                if HT > OSP and C < RSP:
                    #short signal
                    Rsignal = -1
            else:
                if LT < OLP and C > RLP:
                    # long signal
                    Rsignal = 1
        self.R_prediction = Rsignal
        self.evt.consoleLog('Rbreaker signal: ' + str(Rsignal))

        return
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
        
    
