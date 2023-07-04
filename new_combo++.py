from AlgoAPI import AlgoAPIUtil, AlgoAPI_Backtest
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
#from pandas.core.Frame import DataFrame
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # One time step
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # out.size() --> 100, 10
        out = self.fc(out) 

        # map out to 0~1
        out = torch.sigmoid(out)

        return out

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

        # Initialize the LSTM model
        self.LSTM_model = None

        # Initialize the LSTM prediction for next day close price
        self.LSTM_prediction = None
        self.LSTM_credit = 10
        
        # the credit rating for the AI model
        self.credit = 1
        
        # Initialize reference number
        self.ref = 0
    
        self.R_prediction = 0
        self.R_credit = 10
        self.count = 0

        # Use to store yesturday's status
        self.yesterday = None

    def start(self, mEvt):
        self.evt = AlgoAPI_Backtest.AlgoEvtHandler(self, mEvt)
        self.myinstrument = mEvt['subscribeList'][0]
        self.evt.consoleLog('myinstrument = '+str(self.myinstrument))

        # Get initial past 200 days close price data
        #self.closeprices, timestamps = self.getClosePrice(self.myinstrument, 200, None)
        
        # Initialize ARIMA model
        #self.ARIMAparams = self.getARIMA(self.myinstrument, self.closeprices)
        #self.evt.consoleLog('ARIMA parameters: ' + str(self.ARIMAparams))
        
        # Initialize the first ARIMA prediction
        #model = ARIMA(self.closeprices, order=self.ARIMAparams)
        #model_fit = model.fit()
        #self.ARIMA_prediction = model_fit.forecast(steps=1)[0]

        
        open_1, t = self.getOpenPrice(self.myinstrument, 300, None)
        #self.evt.consoleLog('open price: ')
        #self.evt.consoleLog(open_1, t)
        high_1, t = self.getHighPrice(self.myinstrument, 300, None)
        #self.evt.consoleLog('high price: ')
        #self.evt.consoleLog(high_1, t)
        low_1, t = self.getLowPrice(self.myinstrument, 300, None)
        #self.evt.consoleLog('low price: ')
        #self.evt.consoleLog(low_1, t)
        close_1, t = self.getClosePrice(self.myinstrument, 300, None)
        #self.evt.consoleLog('close price: ')
        #self.evt.consoleLog(close_1, t)
        data_len = len(close_1)
        data_all = {'open' : open_1[0:data_len],
                    'high' : high_1[0:data_len],
                    'low'  : low_1[0:data_len],
                    'close': close_1[0:data_len],
                    # target: from 1 to 300 if close price is higher than previous day, 0 otherwise
                    'target': [1 if close_1[i] > close_1[i-1] else 0 for i in range(1, data_len)] + [0]
        }
        df_main = pd.DataFrame(data_all)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        sel_col = ['open', 'high', 'low', 'close']
        for col in sel_col:
            df_main[col] = scaler.fit_transform(df_main[col].values.reshape(-1,1))
            
        
        df_main = df_main.astype(np.float32)
        data_raw = df_main
        seq = 10
        test_set_size = 0
        # generate train & test dataset
        #feat,target = create_seq_data(data_raw,seq)
        
        data_feat,data_target = [],[]
        for index in range(len(data_raw - 1) - seq):
            data_feat.append(data_raw[['open', 'high', 'low', 'close']][index: index + seq].values)
            data_target.append(data_raw['target'][index:index + seq])
        data_pred = data_raw[['open', 'high', 'low', 'close']][-seq:].values
        feat = np.array(data_feat)
        target = np.array(data_target)
        
        #trainX,trainY,testX,testY = train_test(feat,target,test_set_size,seq)
        train_size = feat.shape[0] - (test_set_size) 
        trainX = torch.from_numpy(feat[:train_size].reshape(-1,seq,4)).type(torch.Tensor)
        trainY = torch.from_numpy(target[:train_size].reshape(-1,seq,1)).type(torch.Tensor)
        
        self.evt.consoleLog('x_train.shape = '+str(trainX.shape))
        self.evt.consoleLog('y_train.shape = '+str(trainY.shape))
        
        n_steps = seq
        batch_size = 259
        num_epochs = 150
        
        train = torch.utils.data.TensorDataset(trainX,trainY)
        train_loader = torch.utils.data.DataLoader(dataset=train, 
                                           batch_size=batch_size, 
                                           shuffle=False)
        
        input_dim = 4
        hidden_dim = 20
        num_layers = 2 
        output_dim = 1
        self.model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

        # Binary cross entropy loss
        loss_fn = nn.MSELoss()

        optimiser1 = torch.optim.Adam(self.model.parameters(), lr=0.1)
        optimiser2 = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.evt.consoleLog(self.model)
        self.evt.consoleLog(len(list(self.model.parameters())))
        for i in range(len(list(self.model.parameters()))):
            self.evt.consoleLog(list(self.model.parameters())[i].size())
         
        hist = np.zeros(num_epochs)
        seq_dim = seq
        for t in range(num_epochs):
            # Initialise hidden state
            # Don't do this if you want your LSTM to be stateful
            #model.hidden = model.init_hidden()
    
            # Forward pass
            y_train_pred = self.model(trainX)

            loss = loss_fn(y_train_pred, trainY)
            if t % 10 == 0 and t !=0:
                self.evt.consoleLog("Epoch ", t, "MSE: ", loss.item())
            hist[t] = loss.item()

            # Zero out gradient, else they will accumulate between epochs
            if t < 70:
                optimiser = optimiser1
            else:
                optimiser = optimiser2
            optimiser.zero_grad()

            # Backward pass
            loss.backward()

            # Update parameters
            optimiser.step()
        
        
        
        # predict tomorrow
        last_seq = data_pred
        last_seq = torch.from_numpy(last_seq.reshape(-1,seq,4)).type(torch.Tensor)
        last_seq_pred = self.model(last_seq)
        # print last 9 true values
        self.evt.consoleLog('true: '+str(data_all["target"][-10:-1]))
        # print 9 predicted values
        self.evt.consoleLog('pred: '+str(last_seq_pred.detach().numpy()[:,:-1,0]))

        last_seq_pred = last_seq_pred.detach().numpy()[:,-1,0]

        self.evt.consoleLog('last_seq_pred: '+str(last_seq_pred))
        self.LSTM_prediction = last_seq_pred

        self.evt.start()
        
        
    def on_bulkdatafeed(self, isSync, bd, ab):
        # --------------------------------------------
        # LSTM
        todayclose, today = self.getClosePrice(self.myinstrument, 2, None)
        # Compare the prediction with the real close price
        self.evt.consoleLog('LSTM prediction: ' + str(self.LSTM_prediction))
        self.evt.consoleLog('Real close price: ' + str(todayclose))
        if self.LSTM_prediction >= 0.5 and todayclose[-1] > todayclose[-2]:
            self.evt.consoleLog('True prediction')
            self.LSTM_credit += 1
        elif self.LSTM_prediction < 0.5 and todayclose[-1] < todayclose[-2]:
            self.evt.consoleLog('True prediction')
            self.LSTM_credit += 1
        else:
            self.evt.consoleLog('False prediction')
            if self.LSTM_credit > 1:
                self.LSTM_credit -= 1
        self.evt.consoleLog(' ')
        if todayclose[-1] > todayclose[-2]:
            self.yesterday = 1
        else:
            self.yesterday = 0
        
        open, t = self.getOpenPrice(self.myinstrument, 30, None)
        high, t = self.getHighPrice(self.myinstrument, 30, None)
        low, t = self.getLowPrice(self.myinstrument, 30, None)
        close, t = self.getClosePrice(self.myinstrument, 30, None)
        data_len = len(close)
        data_all = {'open' : open[0:data_len],
                    'high' : high[0:data_len],
                    'low'  : low[0:data_len],
                    'close': close[0:data_len],
                    'target': [1 if close[i] > close[i-1] else 0 for i in range(1, data_len)] + [0]
        }
        df_main = pd.DataFrame(data_all)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        sel_col = ['open', 'high', 'low', 'close']
        for col in sel_col:
            df_main[col] = scaler.fit_transform(df_main[col].values.reshape(-1,1))

        df_main = df_main.astype(np.float32)
        data_raw = df_main
        seq = 10
        # No need to test, fit all data
        test_set_size = 0

        data_feat,data_target = [],[]
        for index in range(len(data_raw - 1) - seq):
            data_feat.append(data_raw[['open', 'high', 'low', 'close']][index: index + seq].values)
            data_target.append(data_raw['target'][index:index + seq])
        data_pred = data_raw[['open', 'high', 'low', 'close']][-seq:].values
        feat = np.array(data_feat)
        target = np.array(data_target)

        trainX = torch.from_numpy(feat.reshape(-1,seq,4)).type(torch.Tensor)
        trainY = torch.from_numpy(target.reshape(-1,seq,1)).type(torch.Tensor)

        n_steps = seq
        batch_size = 259
        num_epochs = 10 # avoid overfitting

        train = torch.utils.data.TensorDataset(trainX,trainY)
        train_loader = torch.utils.data.DataLoader(dataset=train,
                                             batch_size=batch_size,
                                                shuffle=False)
        
        loss_fn = nn.BCELoss()
        optimiser = torch.optim.Adam(self.model.parameters(), lr=0.01)
        input_dim = 4
        hidden_dim = 20
        num_layers = 2
        output_dim = 1

        hist = np.zeros(num_epochs)
        seq_dim = seq
        for t in range(num_epochs):
            y_train_pred = self.model(trainX)
            
            loss = loss_fn(y_train_pred, trainY)
            if t % 10 == 0 and t !=0:
                self.evt.consoleLog("Epoch ", t, "Cross Entropy: ", loss.item())
            hist[t] = loss.item()

            optimiser.zero_grad()

            loss.backward()

            optimiser.step()

        # predict tomorrow
        last_seq = data_pred
        last_seq = torch.from_numpy(last_seq.reshape(-1,seq,4)).type(torch.Tensor)
        last_seq_pred = self.model(last_seq)
        last_seq_pred = last_seq_pred.detach().numpy()[:,-1,0]
        self.LSTM_prediction = last_seq_pred

        self.evt.consoleLog('last_seq_pred: '+str(last_seq_pred))

        # --------------------------------------------
        # end of LSTM


        #-----------------------R-breaker

        self.R_prediction = 0
        todayclose, today = self.getClosePrice(self.myinstrument, 2, None)
        # Compare the prediction with the real close price
        self.evt.consoleLog('Rbreaker prediction: ' + str(self.R_prediction))
        self.evt.consoleLog('Real close price: ' + str(todayclose))
        if self.R_prediction == 1 and todayclose[-1] > todayclose[-2]:
            self.evt.consoleLog('True prediction')
            self.R_credit += 1
        elif self.R_prediction == -1 and todayclose[-1] < todayclose[-2]:
            self.evt.consoleLog('True prediction')
            self.R_credit += 1
        elif self.R_prediction == 0:
            # No prediction
            self.evt.consoleLog('No prediction')
        else:
            self.evt.consoleLog('False prediction')
            if self.R_credit > 1:
                self.R_credit -= 1
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
        
        # Test for buy-sell function
        self.doit(self.myinstrument, 1, self.ref, 1 )
        self.doit(self.myinstrument, -1, self.ref, 1 )
        
        pos, osOrder, pendOrder = self.evt.getSystemOrders()
        position = pos[self.myinstrument]["netVolume"]
        self.evt.consoleLog('Position: '+str(position))
        
        if position == 0:
            if TC > TLP:
                #long signal
                Rsignal = 1
                        
            elif TC < TSP:
                #short signal
                Rsignal = -1
                        
        else:
            if position > 0:
                if HT > OSP and TC < RSP:
                    #short signal
                    Rsignal = -1
            else:
                if LT < OLP and TC > RLP:
                    # long signal
                    Rsignal = 1
                    
        self.R_prediction = Rsignal
        self.evt.consoleLog('Rbreaker signal: ' + str(Rsignal))

        # Combine the two
        LSTM_score = (self.LSTM_prediction - 0.5) * 2
        R_score = self.R_prediction
        Predict_score = (LSTM_score * self.LSTM_credit + R_score * self.R_credit) / (self.LSTM_credit + self.R_credit)
        self.evt.consoleLog('Predict score: ' + str(Predict_score))

        if position == 0:
            if LSTM_score > 0.5:
                self.evt.consoleLog('Initial Buy')
                self.doit(self.myinstrument, 1, self.ref, 50)
            elif LSTM_score < -0.5:
                self.evt.consoleLog('Initial Sell')
                self.doit(self.myinstrument, -1, self.ref, 50)

        else:
            if HT > OSP and TC < RSP and Predict_score < - 0.7:
                self.evt.consoleLog('Sell')
                if position > 30:
                    self.doit(self.myinstrument, -1, self.ref, abs(position) * 0.9)
                else:
                    self.doit(self.myinstrument, -1, self.ref, 50)
            elif LT < OLP and TC > RLP and Predict_score > 0.7:
                self.evt.consoleLog('Buy')
                if position < -30:
                    self.doit(self.myinstrument, 1, self.ref, abs(position) * 0.9)
                else:
                    self.doit(self.myinstrument, 1, self.ref, 50)
            
            # Case R-breaker is not accurate as LSTM and did not predict
            if R_score == 0:
                if Predict_score > 0.6:
                    self.evt.consoleLog('Buy')
                    if position > 30:
                        self.doit(self.myinstrument, 1, self.ref, abs(position) * 0.5)
                    else:
                        self.doit(self.myinstrument, 1, self.ref, 50)
                elif Predict_score < -0.6:
                    self.evt.consoleLog('Sell')
                    if position < -30:
                        self.doit(self.myinstrument, -1, self.ref, abs(position) * 0.5)
                    else:
                        self.doit(self.myinstrument, -1, self.ref, 50)
                      
            
    # 当过去两天涨幅大于5%,平掉所有仓位止盈
        if position < 0 and todayclose[-1]/ todayclose[-2] > 1.05:
            self.doit(self.myinstrument, 1, self.ref, abs(position))
           
    # 当时间为周五并且跌幅大于5%时,平掉所有仓位止损
        if position > 0 and todayclose[-1] / todayclose[-2] < 0.95 :
            self.doit(self.myinstrument, -1, self.ref, abs(position))
            

        # --------------------------------------------
        # ARIMA prediction
        # Compare the prediction with the real close price
        #self.evt.consoleLog('ARIMA prediction: ' + str(self.ARIMA_prediction))
        #self.evt.consoleLog('Real close price: ' + str(todayclose))
        #if self.ARIMA_prediction > self.closeprices[-2] and todayclose[0] > self.closeprices[-2]:
            #self.evt.consoleLog('True prediction')
        #elif self.ARIMA_prediction < self.closeprices[-2] and todayclose[0] < self.closeprices[-2]:
            #self.evt.consoleLog('True prediction')
        #else:
            #self.evt.consoleLog('False prediction')
        #self.evt.consoleLog(' ')

        # Fit data into the ARIMA model and do prediction
        #model = ARIMA(self.closeprices, order=self.ARIMAparams)
        #model_fit = model.fit()

        # Do one step prediction
        #ARIMA_prediction = model_fit.forecast(steps=1)[0]

        # Update the prediction
        #self.ARIMA_prediction = ARIMA_prediction
        # --------------------------------------------
    def doit(self, instrument, buysell, Ref, vol):
        order = AlgoAPIUtil.OrderObject(
            instrument = instrument,
            orderRef = Ref,
            openclose = 'open', 
            buysell = buysell,    #1=buy, -1=sell
            ordertype = 0,  #0=market, 1=limit
            volume = vol,
        )
        self.evt.sendOrder(order)
        self.ref += 1


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

        
