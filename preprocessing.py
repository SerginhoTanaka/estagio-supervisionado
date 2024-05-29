# sklearn methods
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import OneHotEncoder

# streamlit
import streamlit as st

# pandas
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
    
    def select_cleaning_method(self) -> List[str]:
        cleaning_method = st.sidebar.multiselect(
            'Selecione um método de limpeza:',
            ('nenhum', 'Remover linhas com valores nulos', 'Remover linhas duplicadas', 'Remover ruidos'),
            default='nenhum'
        )
        return cleaning_method
    
    def _select_preprocessing_method(self) -> None:
        preprocessing_method = st.sidebar.selectbox(
            'Selecione um método de normalização:',
            ('nenhum', 'MinMaxScaler', 'StandardScaler', 'RobustScaler', 'Normalizer', 'MaxAbsScaler'),
            index=0
        ) 
        self.__apply_preprocessing(preprocessing_method)

    def __apply_preprocessing(self, preprocessing_method: str) -> None:
        scaler_map = {
            'MinMaxScaler': MinMaxScaler(),
            'StandardScaler': StandardScaler(),
            'RobustScaler': RobustScaler(),
            'Normalizer': Normalizer(),
            'MaxAbsScaler': MaxAbsScaler()
        }
        
        self.scaler = scaler_map.get(preprocessing_method)
        
        cleaning_methods = self.select_cleaning_method()
        
        # Exibir apenas as contagens e a tabela se nenhum método for selecionado
        if 'nenhum' in cleaning_methods and preprocessing_method == 'nenhum':
            st.write("Nenhum método de limpeza ou normalização selecionado")
            st.write("Quantidade de valores em cada coluna:")
            st.table(self.data.count())
            st.write("Tabela original:")
            st.write(self.data)
            return
        
        df = self.data.copy()
        if 'nenhum' not in cleaning_methods:
            for method in cleaning_methods:
                if method == 'Remover linhas com valores nulos':
                    df = self._clean_null(df)
                elif method == 'Remover linhas duplicadas':
                    df = self._clean_duplicates(df)
                elif method == 'Remover ruidos':
                    categorical_values = df.select_dtypes(include=['object'])
                    numerical_df = df.drop(columns=categorical_values.columns.tolist())
                    bool_columns = numerical_df.select_dtypes(include=["bool"]).columns.tolist()
                    numerical_df[bool_columns] = numerical_df[bool_columns].astype(int)
                    df = self._clean_noise(numerical_df)
                    df = pd.concat([df, categorical_values], axis=1)
        
        st.write("Quantidade de valores em cada coluna antes da limpeza:")
        st.table(self.data.count())

        st.write("Quantidade de valores em cada coluna depois da limpeza:")
        st.table(df.count())
        
        if preprocessing_method != 'nenhum':
            scaled_data = self.scaler.fit_transform(df[['cdn_cliente']])
            normalizer_data = pd.DataFrame(scaled_data, columns=['cdn_cliente'])
            
            st.write("cdn_cliente normalizado:")
            st.write(f"Normalizado pelo método {preprocessing_method}:")
            st.write(normalizer_data)
        else:
            st.write("Nenhum método de normalização selecionado")

    def _clean_null(self, df: pd.DataFrame) -> pd.DataFrame:
        st.write('Valores nulos antes da limpeza:', df.isnull().sum())
        return df.dropna()

    def _clean_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        st.write('Valores duplicados antes da limpeza:', df.duplicated().sum())
        return df.drop_duplicates()

    def _clean_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        st.write('Valores ruidosos antes da limpeza:', (np.abs(stats.zscore(df)) > 3).sum())
        return df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]