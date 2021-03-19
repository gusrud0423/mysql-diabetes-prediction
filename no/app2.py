import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import h5py
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import pickle

from sklearn.ensemble import RandomForestClassifier
import joblib 
def main() :
    
    
    model =  joblib.load('data/best_model.pk1')

    diabetes_df =  pd.read_csv('data/diabetes.csv')
    st.dataframe(diabetes_df)

    new_data =  np.array([3,88,58,11,54,24,0.26,22]) # 
    new_data = new_data.reshape(1,-1)
    print(new_data)

    st.write(model.predict(new_data))

if __name__ == '__main__' :
    main()

    #$ nohup srteamlit rn app.py & 하면 전세계에서 누구나 접속 가능 

    # $ ps -ef|grep streamlit streamlit
    ubuntu 28371....

    # kill 28371