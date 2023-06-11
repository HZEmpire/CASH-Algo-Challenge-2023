from AlgoAPI import AlgoAPIUtil, AlgoAPI_Backtest
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd

class AlgoEvent:
    
    # Open Price
    def getOpenPrice(self, instrument, days, endtime):
        contract = {"instrument": instrument}
        res = self.evt.getHistoricalBar(contract, days, 'D', endtime)
        openprices = []
        timestamps = []
        for t in res:
            timestamp = t
            lastprice = res[t]['o']
            openprices.append(lastprice)
            timestamps.append(timestamp)
        return openprices, timestamps
    
    # High Price
    def getHighPrice(self, instrument, days, endtime):
        contract = {"instrument": instrument}
        res = self.evt.getHistoricalBar(contract, days, 'D', endtime)
        highprices = []
        timestamps = []
        for t in res:
            timestamp = t
            lastprice = res[t]['h']
            highprices.append(lastprice)
            timestamps.append(timestamp)
        return highprices, timestamps
    
    # Low Price
    def getLowPrice(self, instrument, days, endtime):
        contract = {"instrument": instrument}
        res = self.evt.getHistoricalBar(contract, days, 'D', endtime)
        lowprices = []
        timestamps = []
        for t in res:
            timestamp = t
            lastprice = res[t]['l']
            lowprices.append(lastprice)
            timestamps.append(timestamp)
        return lowprices, timestamps
    
    # Close Price
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
    
    # ARIMA model for close price pridiction
    def getARIMA(self, instrument, closeprices):
        low_AIC = np.inf
        best_ar = 1
        best_ma = 0
        best_diff = 0
        for ar in range(1, 8):
            for ma in range(5):
                for diff in range(2):
                    try:
                        model = ARIMA(closeprices, order=(ar, diff, ma))
                        model_fit = model.fit()
                        if model_fit.aic < low_AIC:
                            low_AIC = model_fit.aic
                            best_ar = ar
                            best_ma = ma
                            best_diff = diff
                    except:
                        continue
        params = (best_ar, best_diff, best_ma)
        return params
    
    def __init__(self):
        # Get initial past 200 days close price data
        self.openprices = None
        self.highprices = None
        self.lowprices = None
        self.closeprices = None

        # Initialize ARIMA model
        self.ARIMAparams = None
        
        # Initialize the ARIMA prediction for next day close price
        self.ARIMA_prediction = None
        
    def start(self, mEvt):
        self.evt = AlgoAPI_Backtest.AlgoEvtHandler(self, mEvt)

        # Get initial past 200 days close price data
        self.closeprices, timestamps = self.getClosePrice('00001HK', 200, None)
                
        # Initialize ARIMA model
        self.ARIMAparams = self.getARIMA('00001HK', self.closeprices)
        self.evt.consoleLog('ARIMA parameters: ' + str(self.ARIMAparams))
        
        # Initialize the first ARIMA prediction
        model = ARIMA(self.closeprices, order=self.ARIMAparams)
        model_fit = model.fit()
        self.ARIMA_prediction = model_fit.forecast(steps=1)[0]

        self.evt.start()
        
    def on_bulkdatafeed(self, isSync, bd, ab):
        # Add new close price data
        todayclose, today = self.getClosePrice('00001HK', 1, None)
        self.closeprices = self.closeprices + todayclose

        # --------------------------------------------
        # ARIMA prediction
        # Compare the prediction with the real close price
        self.evt.consoleLog('ARIMA prediction: ' + str(self.ARIMA_prediction))
        self.evt.consoleLog('Real close price: ' + str(todayclose))
        if self.ARIMA_prediction > self.closeprices[-2] and todayclose[0] > self.closeprices[-2]:
            self.evt.consoleLog('True prediction')
        elif self.ARIMA_prediction < self.closeprices[-2] and todayclose[0] < self.closeprices[-2]:
            self.evt.consoleLog('True prediction')
        else:
            self.evt.consoleLog('False prediction')
        self.evt.consoleLog(' ')

        # Fit data into the ARIMA model and do prediction
        model = ARIMA(self.closeprices, order=self.ARIMAparams)
        model_fit = model.fit()

        # Do one step prediction
        ARIMA_prediction = model_fit.forecast(steps=1)[0]

        # Update the prediction
        self.ARIMA_prediction = ARIMA_prediction
        # --------------------------------------------

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
