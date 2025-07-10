import numpy as np
import pandas as pd
import torch
import os
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
import torch.optim as optim
import time
torch.manual_seed(101)


os.chdir(os.path.dirname(os.path.abspath(__file__)))

def create_dataframe(dataset,date_cols=[],rename_columns=False,
                   remove_cols=False,cols_to_remove=[],
                   datecol_to_group=None,break_time_cols=False,agg_func='mean',
                   change_to_cat=False,cols_to_cat=[]):
    
    new_dataset = dataset.copy()
    #change col to categorical applying one hot encode for categorical cols
    if change_to_cat:
        ohe = OneHotEncoder().fit(new_dataset.loc[:,cols_to_cat])
        for col in cols_to_cat:
            new_dataset[f"{col}"] = pd.Categorical(new_dataset[f"{col}"])
        new_cat_cols = [f"is_{c}" for c in ohe.categories_[0]]
        new_dataset.loc[:,new_cat_cols] = ohe.transform(new_dataset.loc[:,cols_to_cat]).toarray()
    #remove cols
    if remove_cols:
        new_dataset = new_dataset.drop(labels=cols_to_remove,axis=1)
    #convert the date columns to timestamp and create a new column just for date
    if len(date_cols)>0:
        for col in date_cols:
            new_dataset[f"{col}_date"] = [datetime.fromisoformat(d).date() for d in new_dataset[f"{col}"]]
    #break the date by year,month and day
    if break_time_cols:
        for col in date_cols:
            new_dataset[f"{col}_year"] = [d.year for d in new_dataset[f"{col}_date"]]
            new_dataset[f"{col}_month"] = [d.month for d in new_dataset[f"{col}_date"]]
            new_dataset[f"{col}_day"] = [d.day for d in new_dataset[f"{col}_date"]]
    #remove blank spaces in columns and replace by "_"
    if rename_columns:
        new_c = {c:c.replace(" ","_") for c in new_dataset.columns.to_list()}
        new_dataset = new_dataset.rename(new_c,axis="columns")
    
    #create dataset with timestamp index
    new_dataframe = new_dataset.groupby([f"{datecol_to_group}_date"]).agg(agg_func,numeric_only=True)
    #rename index col
    if new_dataframe.index.name != "datetime":
        new_dataframe.index = new_dataframe.index.rename("datetime")
    
    
    return new_dataframe

def preprocessing(dataframe):
    new_dataframe = dataframe.copy()
    scaler = StandardScaler()
    #applying StandardScaler
    if len(new_dataframe.shape)>1:
        scaler.fit(new_dataframe)
        new_values = scaler.transform(new_dataframe)
        new_dataframe.loc[:,new_dataframe.columns.to_list()] = new_values
        
        #remove cols with constant values
        thr = int(dataframe.shape[0]*0.8)
        rm_cols = []
        for col in new_dataframe.columns:
            values,counts = np.unique(new_dataframe[col],return_counts=True)
            if (len(values)==1) or (np.any(counts>thr)):
                rm_cols.append(col)
                
        new_dataframe = new_dataframe.drop(labels=rm_cols,axis="columns")
        
    else:
        scaler.fit(new_dataframe.values.reshape(-1,1))
        new_values = scaler.fit_transform(new_dataframe.values.reshape(-1,1))
        new_dataframe = new_values
        
    return new_dataframe

 #remove cols that are in test but not in train and vice-versa
def rm_unseen_cols(train_df,test_df):
    cols_not_in_train = [col for col in test_df.columns if col not in train_df.columns]
    cols_not_in_test = [col for col in train_df.columns if col not in test_df.columns]
    new_X_test_df = test_df.drop(labels=cols_not_in_train,axis="columns")
    new_X_train_df = train_df.drop(labels=cols_not_in_test,axis="columns")
    
    return new_X_train_df, new_X_test_df

def create_dataset(train_dataframe,test_dataframe,target,n_input,n_out):
    # convert history into inputs and outputs
    def to_supervised(dataframe,target, n_input=n_input, n_out=n_out):
        """
        This functions creates a history from the dataframe values and target values.
        For the X value we're going to separate the values from dataframe by n_input days.
        For the Y value we're going to get the values from the last day of input (in_end) till
        the n_out (that is the number of outputs).
        Args:
        Dataframe: pd.Dataframe. the dataframe with all the values including the values from the target
        Target: string. Name of the column that we're going to forecast
        N_input: int. The n_input days that are going to by our history. By default is 7.
        N_out: int. The size of sequence that we're going to forecast. By defeaut is 7.
        Returns:
        X,Y: np.array,np.array: The X vector has the history values from the dataset and the Y contains the history values
        that we're going to predicted. 
    """
        X, y = list(), list()
        in_start = 0
        # step over the entire history one time step at a time
        for _ in range(dataframe.shape[0]):
            # define the end of the input sequence
            in_end = in_start + n_input
            out_end = in_end + n_out
            # ensure we have enough data for this instance
            if out_end <= dataframe.shape[0]:
                x_input = dataframe.iloc[in_start:in_end,:].values
                X.append(x_input)
                y.append(dataframe[target].iloc[in_end:out_end].values)
            # move along one time step
            in_start += 1
        return np.array(X), np.array(y)
    
    xtrain,ytrain = to_supervised(train_dataframe,target)
    xtest,ytest = to_supervised(test_dataframe,target)
  
    train_dataset = TensorDataset(torch.Tensor(xtrain).to(device),torch.Tensor(ytrain).unsqueeze(2).to(device))
    test_dataset = TensorDataset(torch.Tensor(xtest).to(device),torch.Tensor(ytest).unsqueeze(2).to(device))
    
    return train_dataset,test_dataset

def split_dataset(dataframe):
    # split into standard weeks
    sp_df = np.array(np.split(dataframe.values, dataframe.shape[0]/7))
    return sp_df

class LSTMModel(nn.Module):
    def __init__(self,input_dim,hidden_dim,layer_dim,output_dim,dropout_prob,is_bidirectional=False):
        super(LSTMModel,self).__init__()
        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.bidirectional = is_bidirectional

        #dimension for LSTM or BiLSTM
        if self.bidirectional:
            self.D = 2
        else:
            self.D = 1

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True,
            bidirectional=self.bidirectional, dropout=dropout_prob
        )
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim*self.D,output_dim)

    def forward(self,x):

        h0 = torch.zeros(self.layer_dim*self.D, x.size(0), self.hidden_dim,device=device).requires_grad_()
        c0 = torch.zeros(self.layer_dim*self.D, x.size(0), self.hidden_dim,device=device).requires_grad_()
        
        #we need to detach since we're doing backpropagatio through time 
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        
        return out
    
#Create Optimization Class
class Optimization:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
    
    def train_step(self, x, y):
        # x.to(device)
        # y.to(device)
        # print(x.is_cuda, self.model.is_cuda)
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        train_preds = self.model(x.to(device))

        # Computes loss
        loss = self.loss_fn(y, train_preds)

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()
    
    def train(self, train_loader,batch_size=64,n_features=1, n_epochs=50):
        model_path = f'./models/{self.model}.pth'
#         print("model_path", model_path)
        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)
            

            if (epoch <= 10) | (epoch % 50 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}"
                )

#             if not os.path.exists(model_path):
#                 print("here")
#                 torch.save(self.model.state_dict(), model_path)
            model = self.model
        return model 
                
    # make a forecast
    def forecast(self,model, history, n_seq):
        history = np.array(history)
        data = history.reshape(history.shape[0]*history.shape[1], history.shape[2])
        # retrieve last observations for input data
        input_x = data[-n_seq:, :]
        # reshape into [1, n_input, 1]
        input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))#.to(device)
        # forecast the next week
        with torch.no_grad():
            input_tensor = torch.tensor(input_x, dtype=torch.float32).to(device)
            pred = model(input_tensor).cpu().numpy()

        
        return pred[0]
    def evaluate_forecasts(self,actual, predicted):
        scores = list()
        if len(predicted.shape) > 2:
            predicted = predicted.squeeze(axis=2)
        # calculate an RMSE score for each day
        for i in range(actual.shape[1]):
            # calculate mse
            mse = mean_squared_error(actual[:, i], predicted[:, i])
            
            # calculate rmse
            rmse = sqrt(mse)
            # store
            scores.append(rmse)
        # calculate overall RMSE
        s = 0
        for row in range(actual.shape[0]):
            for col in range(actual.shape[1]):
                s += (actual[row, col] - predicted[row, col])**2
                score = sqrt(s / (actual.shape[0] * actual.shape[1]))
            return score, scores
    
    def evaluate_model(self,train, test, n_seq,model):
        # history is a list of weekly data
        history = [x_train for x_train in train]
        # walk-forward validation over each week
        predictions = list()
        for i in range(len(test)):
            # predict the week
            pred_sequence = self.forecast(model, history, n_seq)
            # store the predictions
            predictions.append(pred_sequence)
            # get real observation and add to history for predicting the next week
            history.append(test[i, :])
            # evaluate predictions days for each week
        predictions = np.array(predictions)
        score, scores = self.evaluate_forecasts(test[:, :, test.shape[2]-1], predictions)
        return score, scores, predictions 
        
    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()
        
    # summarize scores
    def summarize_scores(self,name, score, scores):
        s_scores = ', '.join(['%.1f' % s for s in scores])
        print('%s: [%.3f] %s' % (name, score, s_scores))

 #inverse transform to results 

def inverse_transform(base_values, to_transform_values):
    scaler = StandardScaler()
    scaler.fit(base_values)
    new_values = scaler.inverse_transform(to_transform_values)
    return new_values


def format_predictions(predictions, values, idx_test):
    df_res = pd.DataFrame(data={"total_load_predicted_values":predictions,
                            "total_load_real_values":values},index=idx_test)
    return df_res

def plot_multiple_time_series(index,real_values,predicted_values,name_model):
    plt.figure(figsize=(20,10))
    plt.plot(index,real_values,".-y",label="real",linewidth=2)
    plt.plot(index,predicted_values,".-.r",label="predicted",linewidth=1)
    plt.legend()
    plt.xticks(rotation = 45)
    plt.title(f"{name_model} - Real x Predicted 7 Days Load Forecast");
    
def subplots_time_series(index,real_values,predicted_values,name_model):
    fig,ax = plt.subplots(2,1,sharex=True,figsize=(20,10))
    ax[0].plot(index,real_values,".-y",label="real",linewidth=1)
    ax[1].plot(index,predicted_values,".-.r",label="predicted",linewidth=1)
    
    ax[0].legend()
    ax[1].legend()
    plt.xticks(rotation = 45)
    plt.suptitle(f"{name_model} - Real and Predicted 7 Days Load Forecast");


if __name__ == "__main__":
    ########################################## 1 #############################################
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    #loading the data
    df_energy= pd.read_csv("./data/energy_dataset.csv")
    df_weather_features = pd.read_csv("./data/weather_features.csv")

    print("shape of energy dataset:",df_energy.shape)
    print("shape of weather features:",df_weather_features.shape)

    rm_cols_enrg = ["generation hydro pumped storage aggregated","forecast solar day ahead","forecast wind offshore eday ahead",
                "forecast wind onshore day ahead", "total load forecast","price day ahead","price actual"]
    rm_cols_wth = ["city_name","weather_id", "weather_description","weather_icon"]

    df_enrg = create_dataframe(df_energy,date_cols=["time"],rename_columns=True,
                               datecol_to_group="time",remove_cols=True,cols_to_remove=rm_cols_enrg,break_time_cols=False)

    df_wth = create_dataframe(df_weather_features,date_cols=["dt_iso"],remove_cols=True,cols_to_remove=rm_cols_wth,
                              datecol_to_group="dt_iso",break_time_cols=False,change_to_cat=True,cols_to_cat=["weather_main"])
    
    all_years_wth = np.array([i_wth.year for i_wth in df_wth.index])
    years_wth,count_values_wth = np.unique(all_years_wth,return_counts=True)

    all_years_enrg = np.array([i_enrg.year for i_enrg in df_enrg.index])
    years_enrg,count_values_enrg = np.unique(all_years_enrg,return_counts=True)

    ########################################## 2 #############################################
    df = df_enrg.join(df_wth,on="datetime",how="inner")
    
    begin_train_date = datetime.strptime("2015-01-01","%Y-%m-%d").date() 
    end_train_date = datetime.strptime("2017-12-29","%Y-%m-%d").date()
    end_test_date = datetime.strptime("2018-12-29","%Y-%m-%d").date()

    X_train = df.loc[(df.index > begin_train_date)&(df.index < end_train_date),~df.columns.isin(["total_load_actual"])]
    y_train = df.loc[(df.index > begin_train_date)&(df.index < end_train_date),"total_load_actual"]
    X_test = df.loc[(df.index>end_train_date)&(df.index<end_test_date),~df.columns.isin(["total_load_actual"])]
    y_test =df.loc[(df.index>end_train_date)&(df.index<end_test_date),"total_load_actual"]

    print("preprocessing data...")
    new_X_train = preprocessing(X_train)
    new_y_train = preprocessing(y_train)

    new_X_test = preprocessing(X_test)
    new_y_test = preprocessing(y_test)

    new_X_train, new_X_test =rm_unseen_cols(new_X_train,new_X_test)

    #Creating train and test dataframes with new preprocessing values 
    #train dataframe
    train_dataframe = new_X_train
    #our target 
    train_dataframe["total_load_actual"] = new_y_train

    #test dataframe
    test_dataframe = new_X_test
    #our target 
    test_dataframe["total_load_actual"] = new_y_test

    sequence_length = 7
    batch_size = 20

    features = [col for col in train_dataframe.columns if col != "total_load_actual"]
    target = "total_load_actual"

    train_dataset,test_dataset = create_dataset(train_dataframe,test_dataframe,target,n_input=7,n_out=7)

    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=False)

    X_train, y_train = next(iter(train_loader))

    #split the dataframe by weeks  
    week_train_dataset = split_dataset(train_dataframe)
    week_test_dataset = split_dataset(test_dataframe)

    ################################ 4 ########################################
    input_dim = X_train.shape[2]
    n_seq = 7
    batch_size = 16
    output_dim = 1
    hidden_dim = 32
    n_epochs = 500
    learning_rate = 1e-4
    weight_decay = 1e-6

    model_params = {'input_dim': input_dim,
                'hidden_dim' : hidden_dim,
                'output_dim' : output_dim,
                }

    models = {
        "vanilla_lstm":LSTMModel(**model_params,layer_dim=1,is_bidirectional=False,dropout_prob=0).to(device),
        "stacked_lstm":LSTMModel(**model_params,layer_dim=2,is_bidirectional=False,dropout_prob=0.2).to(device),
        "model_bilstm":LSTMModel(**model_params,layer_dim=5,is_bidirectional=True,dropout_prob=0).to(device)
    }
    loss_fn = nn.MSELoss(reduction="mean")
    opts = []
    for name_model,model in models.items():
        print(name_model)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
        opts.append((f"{name_model}",opt))
    print(models)
    if not os.path.exists("./models"):
        os.makedirs("./models")

    predictions_by_model = {
        "vanilla_lstm":[],
        "stacked_lstm":[],
        "model_bilstm":[]
    }

    for name_model,opt in opts:
        print(f"===== training {name_model} =====")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time_start = time.perf_counter()
        model = opt.train(train_loader, n_epochs=n_epochs,batch_size=batch_size,n_features=input_dim)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time_end = time.perf_counter()
        print(f"Training time for {name_model}: {time_end - time_start:.4f} seconds")
        mean_time_per_epoch = (time_end - time_start) / n_epochs
        print(f"Mean time per epoch for {name_model}: {mean_time_per_epoch:.4f} seconds")
        model_path = f'./models/{name_model}.pth'
        print("model_path", model_path)
        torch.save(model, model_path)
        print(f"==== plot losses - {name_model} ====== ")
        opt.plot_losses()
        score, scores,predictions = opt.evaluate_model(week_train_dataset, week_test_dataset, n_seq,model)
        predictions_by_model[name_model].append(predictions)
        # summarize scores
        print(f"===== scores for {name_model} ====")
        opt.summarize_scores(name_model, score, scores)

    pred_vanilla_lstm_values = predictions_by_model["vanilla_lstm"][0].squeeze(2)
    vanilla_lstm_values = np.ravel(inverse_transform(y_test.values.reshape(-1,1),pred_vanilla_lstm_values))
    
    df_vanilla_lstm_values = format_predictions(vanilla_lstm_values,y_test,y_test.index)

    print(f"RMSE for Vanilla LSTM: ",mean_squared_error(df_vanilla_lstm_values["total_load_real_values"], 
                                                        df_vanilla_lstm_values["total_load_predicted_values"]))
    
    # Predicitons for Stacked LSTM with two layers
    pred_stacked_lstm_values = predictions_by_model["stacked_lstm"][0].squeeze(2)
    stacked_lstm_values = np.ravel(inverse_transform(y_test.values.reshape(-1,1),pred_stacked_lstm_values))

    df_stacked_lstm_values = format_predictions(stacked_lstm_values,y_test,y_test.index)

    # Predictions for Bi LSTM
    pred_bilstm_values = predictions_by_model["model_bilstm"][0].squeeze(2)
    bilstm_values = np.ravel(inverse_transform(y_test.values.reshape(-1,1),pred_bilstm_values))

    df_bilstm_values = format_predictions(bilstm_values,y_test,y_test.index)
    print(f"RMSE for BILSTM: ",mean_squared_error(df_bilstm_values["total_load_real_values"],
                                                  df_bilstm_values["total_load_predicted_values"]))
    
