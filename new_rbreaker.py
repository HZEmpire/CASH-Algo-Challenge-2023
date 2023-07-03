from AlgoAPI import AlgoAPIUtil, AlgoAPI_Backtest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np



###setting###
#2023.1-2023.5
#capital: 250000
#Enable short sell
#00520HK

class AlgoEvent:
    def __init__(self):
        self.last_grid = 0
        self.last_change_grid = [0, 0]
        self.ref = 0
        self.R_prediction = 0
        self.credit = 1
        self.count = 0

        # Use to store yesturday's status
        self.yesterday = None

        # Initialize reference number
        self.ref = 0
        self.coun = 1

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
        #self.doit(self.myinstrument, 1, self.ref, 30)
        self.evt = AlgoAPI_Backtest.AlgoEvtHandler(self, mEvt)
        self.evt.start()
        
    def on_bulkdatafeed(self, isSync, bd, ab):
       pass

    def on_marketdatafeed(self, md, ab):
        self.R_prediction = 0
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
            self.evt.consoleLog('No prediction')
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
        C = res[yesterday]['c']
        H = res[yesterday]['h']
        L = res[yesterday]['l']
        HT = res[today]['h']
        LT = res[today]['l']
        TC = res[today]['c']
        P = (H + C + L) / 3
        TLP = H + 2 * P - 2 * L # 突破买入价
        OSP = P + H - L # 观察卖出价
        RSP = 2 * P - L # 反转卖出价
        RLP = 2 * P - H # 反转买入价
        OLP = P - (H - L) # 观察买入价
        TSP = L - 2 * (H - P) # 突破卖出价

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
        if self.coun == 1:
            self.doit(self.myinstrument, 1, self.ref, 5)
            self.coun += 1
        

        if position == 0:
            if TC > TLP:
                #long signal
                Rsignal = 1
                self.count += 1
                self.doit(self.myinstrument, 1, self.ref, 30)
                self.ref += 1
            
            elif TC < TSP:
                #short signal
                Rsignal = -1
                self.count += 1
                self.doit(self.myinstrument, -1, self.ref, 30)
        
        else:
            if position > 0:
                if HT > OSP and TC < RSP:
                    #short signal
                    Rsignal = -1
                    self.doit(self.myinstrument, -1, self.ref, 30)
            else:
                if LT < OLP and TC > RLP:
                    # long signal
                    Rsignal = 1
                    self.doit(self.myinstrument, 1, self.ref, 30)
        self.R_prediction = Rsignal
        self.evt.consoleLog('Rbreaker signal: ' + str(Rsignal))
        self.evt.consoleLog('Count: ' + str(self.count))

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
        
    
