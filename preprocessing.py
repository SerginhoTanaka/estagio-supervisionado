#sklearn methods
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MaxAbsScaler
#streamlit
import streamlit as st
#pandas
import pandas as pd

class Preprocessing:
    def __init__(self, data):
        self.data = data
        self.preprocessing_method = None
        self.scaler = None

    def select_preprocessing_method(self):
        self.preprocessing_method = st.sidebar.selectbox(
            'Selecione um método de normalização:',
            ('MinMaxScaler', 'StandardScaler', 'RobustScaler', 'Normalizer', 'MaxAbsScaler')
        )

    def apply_preprocessing(self):
        scaler_map = {
            'MinMaxScaler': MinMaxScaler(),
            'StandardScaler': StandardScaler(),
            'RobustScaler': RobustScaler(),
            'Normalizer': Normalizer(),
            'MaxAbsScaler': MaxAbsScaler()
        }
        
        self.scaler = scaler_map.get(self.preprocessing_method)

        if self.scaler:
            self.normalized_data = self.scaler.fit_transform(self.data)
            st.sidebar.write(f"Método selecionado: {self.preprocessing_method}")