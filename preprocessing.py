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


class Preprocessing():
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.scaler = None
        
    def _clean_data(self, data: pd.DataFrame, columns) -> pd.DataFrame:
        for column in columns:
            data[column] = data[column].dropna()
        return data
    def _select_preprocessing_method(self) -> None:
        preprocessing_method = st.sidebar.selectbox(
            'Selecione um método de normalização:',
            ('MinMaxScaler', 'StandardScaler', 'RobustScaler', 'Normalizer', 'MaxAbsScaler')
        ) 
        self.__apply_preprocessing(preprocessing_method)

    def __apply_preprocessing(self,preprocessing_method: str) -> None:
        scaler_map = {
            'MinMaxScaler': MinMaxScaler(),
            'StandardScaler': StandardScaler(),
            'RobustScaler': RobustScaler(),
            'Normalizer': Normalizer(),
            'MaxAbsScaler': MaxAbsScaler()
        }
        
        self.scaler = scaler_map.get(preprocessing_method)
        
        clenaned_data = self._clean_data(self.data, self.data.columns.tolist())
        categorical_columns = clenaned_data.select_dtypes(include=['object']).columns
        
        numerical_df = clenaned_data.drop(columns=categorical_columns)

        if self.scaler and numerical_df is not None:
            normalized_data = self.scaler.fit_transform(numerical_df)
            print(normalized_data)
            st.sidebar.write(f"Método selecionado: {preprocessing_method}")