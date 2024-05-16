#sklearn methods
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import OneHotEncoder
#streamlit
import streamlit as st
#pandas
import pandas as pd

from typing import List
from scipy.signal import savgol_filter
from scipy import stats
import numpy as np  
import itertools
import re 
class Preprocessing():
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.scaler = None
        
    def _clean_data(self, data_to_clean: pd.DataFrame, columns:List[str],cleaned_methods:List[str]) -> pd.DataFrame:
        for column, cleaned_method in itertools.product(columns, cleaned_methods):
                try:
                    if cleaned_method == 'Remover linhas com valores nulos':
                        print(f'NUlOS Inicialmente da coluna {data_to_clean[column].name} : {data_to_clean[column].isnull().sum()}')
                        # data_to_clean[column] = data_to_clean[column].apply(lambda x: re.sub(r'[!@#$%¨&*(){}\[\]]', '', str(x)) if not isinstance(x, bool) else x)
                        data_to_clean = data_to_clean.dropna(subset=[column])
                        print(f'NUlOS finais da coluna {data_to_clean[column].name} : {data_to_clean[column].isnull().sum()}')
                    if cleaned_method == 'Remover linhas duplicadas':
                        data_to_clean = data_to_clean.drop_duplicates()
                    if cleaned_method == 'Remover rendundantes':
                        numeric_data = pd.to_numeric(data_to_clean[column], errors='coerce')  # Convert to numeric, coerce errors
                        print(numeric_data)
                        clean_data = numeric_data[(np.abs(stats.zscore(numeric_data)) < 3).all(axis=1)]
                        print(clean_data)
                        

                except Exception as e:
                    print(data_to_clean[column].name)
                    print(e)
                    continue
        
        return data_to_clean
    
    def select_clenning_method(self) -> str:
        cleaning_method = st.sidebar.multiselect(
            'Selecione um método de limpeza:',
            ('Remover linhas com valores nulos', 'Remover linhas duplicadas', 'Remover rendundantes')
        )
        return cleaning_method
    
    def _select_columns(self) -> List[str]:
        columns = st.sidebar.multiselect(
            "Selecione as colunas:",
            self.data.columns.tolist()
        )
        return columns
        
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
        
        selected_columns = self._select_columns()
        st.write(len(self.data))
        categorical_values = self.data.select_dtypes(include=['object'])
        
        clened_method = self.select_clenning_method()
        numerical_df = self.data.drop(columns=categorical_values.columns.tolist())
        
        cleaned_numerical_df = self._clean_data(numerical_df, numerical_df.columns.tolist(), clened_method)
        st.write(len(cleaned_numerical_df))
        numerical_scaled = self.scaler.fit_transform(cleaned_numerical_df)
        # cleaned_data = pd.concat([numerical_scaled, categorical_values], axis=1)
        # describe_data = numerical_scaled.describe()
        # #conversar com o ricardo 