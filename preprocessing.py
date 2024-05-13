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

import re 
class Preprocessing():
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.scaler = None
        
    def _clean_data(self, data_to_clean: pd.DataFrame, columns:List[str]) -> pd.DataFrame:
        for column in columns:
                try:
                    data_to_clean[column] = data_to_clean[column].apply(lambda x: re.sub(r'[!@#$%¨&*(){}\[\]]', '', str(x)) if not isinstance(x, bool) else x)
                    data_to_clean = data_to_clean.dropna(subset=[column])
                    # data_to_clean[column] = data_to_clean[column].drop_duplicates() 
                    # TODO retirar dados rendeundantes 
                    if pd.api.types.is_numeric_dtype(data_to_clean[column]):
                        data_to_clean[column] = savgol_filter(data_to_clean[column], 5, 2) #aplicando filtro de dados ruidosos
                    # TODO dados inconsistentes 
                except Exception as e:
                    print(data_to_clean[column].name)
                    print(e)
                    continue
        
        return data_to_clean
    
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
        

        numerical_df = self.data.drop(columns=categorical_values.columns.tolist())
        
        cleaned_numerical_df = self._clean_data(numerical_df, numerical_df.columns.tolist())
        st.write(len(cleaned_numerical_df))
        numerical_scaled = self.scaler.fit_transform(cleaned_numerical_df)
        # cleaned_data = pd.concat([numerical_scaled, categorical_values], axis=1)
        # describe_data = numerical_scaled.describe()
        # #conversar com o ricardo 