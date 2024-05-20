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
 
    #     return data_to_clean
    # def _clean_data(self, data_to_clean: pd.DataFrame, columns:List[str],cleaned_methods:List[str]) -> pd.DataFrame:
    #         try:
    #             if cleaned_method == 'Remover linhas com valores nulos':
    #                 print(f'NUlOS Inicialmente da coluna {data_to_clean[column].name} : {data_to_clean[column].isnull().sum()}')
    #                 # data_to_clean[column] = data_to_clean[column].apply(lambda x: re.sub(r'[!@#$%¨&*(){}\[\]]', '', str(x)) if not isinstance(x, bool) else x)
    #                 data_to_clean = data_to_clean.dropna(subset=[column])
    #                 print(f'NUlOS finais da coluna {data_to_clean[column].name} : {data_to_clean[column].isnull().sum()}')
    #             if cleaned_method == 'Remover linhas duplicadas':
    #                 print(data_to_clean.duplicated().sum())
    #                 data_to_clean = data_to_clean.drop_duplicates()
    #             if cleaned_method == 'Remover rendundantes':#ruidos
    #                 clean_data = data_to_clean[(np.abs(stats.zscore(data_to_clean)) < 3).all(axis=1)]
    #                 print(clean_data)
    #         except Exception as e:
    #             print(data_to_clean[column].name)
    #             print(e)
    #         return data_to_clean
    
    
    def select_clenning_method(self) -> str:
        cleaning_method = st.sidebar.multiselect(
            'Selecione um método de limpeza:',
            ('Remover linhas com valores nulos', 'Remover linhas duplicadas', 'Remover ruidos')
        )
        return cleaning_method
    
        
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
        
        
        clened_method = self.select_clenning_method()
        df = None
        for method in clened_method:
            if method == 'Remover linhas com valores nulos':
                df = self._clean_null(self.data)
            if method == 'Remover linhas duplicadas':
                df = self._clean_duplicates(self.data)
            if method == 'Remover ruidos':
                categorical_values = self.data.select_dtypes(include=['object'])
                numerical_df = self.data.drop(columns=categorical_values.columns.tolist())
                bool_columns = numerical_df.select_dtypes(include=["bool"]).columns.tolist()
                numerical_df[bool_columns] = numerical_df[bool_columns].astype(int)
                df = self._clean_noise(numerical_df)
                df = pd.concat([df, categorical_values], axis=1)
        if df is not None:
            st.write("Quantidade de valores em cada coluna antes da limpeza:")
            st.table(self.data.count())

            st.write("Quantidade de valores em cada coluna depois da limpeza:")
            st.table(df.count())
            
            scaled_data = self.scaler.fit_transform(df[['cdn_cliente']])     
            st.write("cdn_cliente normalizado:")
            st.write(scaled_data)
        # cleaned_numerical_ = self._clean_data(numerical_df, numerical_df.columns.tolist(), clened_method)
        
        # numerical_scaled = self.scaler.fit_transform(cleaned_numerical_df)
        # cleaned_data = pd.concat([numerical_scaled, categorical_values], axis=1)

    def _clean_null(self,df:pd.DataFrame) -> pd.DataFrame:
        st.write('Valores nulos antes da limpeza:', df.isnull().sum())
        return df.dropna()
    def _clean_duplicates(self,df:pd.DataFrame) -> pd.DataFrame:
        st.write('Valores duplicados antes da limpeza:', df.duplicated().sum())
        return df.drop_duplicates()
    def _clean_noise(self, df:pd.DataFrame) -> pd.DataFrame:
        st.write('Valores ruidosos antes da limpeza:', (np.abs(stats.zscore(df)) > 3).sum())
        return df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]