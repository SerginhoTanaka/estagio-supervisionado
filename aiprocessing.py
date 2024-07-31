from typing import List, Optional
import pandas as pd
import numpy as np
import streamlit as st
from preprocessing import Preprocessing 

# Classificação
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Regressão
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class AiProcessing:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.ai = None
        self.normalized_data = None

    def run(self) -> None:
        """
        Método principal para executar o fluxo de pré-processamento e seleção do modelo de IA.
        """
        method = st.selectbox('Selecione um modelo de IA:', ('', 'Regressão', 'Classificação'))
        preprocessing_option = st.selectbox('Deseja pré-processar os dados?', ('', 'Sim', 'Não'))
        
        if preprocessing_option == 'Sim':
            self.normalized_data = self.__preprocessing()
 
        if method == 'Regressão':
            self.__regression()
        elif method == 'Classificação':
            self.__classification()

    def __preprocessing(self) -> pd.DataFrame:
        """
        Método para executar o pré-processamento dos dados.
        """
        preprocessing = Preprocessing(self.data)
        processed_data = preprocessing.run()
        st.write('Dados pré-processados!')
        return processed_data

    def __regression(self) -> None:
        """
        Método para executar a regressão.
        """
        st.subheader('Modelos de Regressão Disponíveis:')
        self.ai = st.selectbox('Selecione um modelo de regressão:', ('', 'Linear Regression', 'SVR', 'Random Forest'))
        
        regression_option = {
            'Linear Regression': self.__run_linear_regression,
            'SVR': self.__run_svr,
            'Random Forest': self.__run_random_forest_regression
        }
        
        if self.ai:
            regression_option[self.ai]()
    
    def __classification(self) -> None:
        """
        Método para executar a classificação.
        """
        st.subheader('Modelos de Classificação Disponíveis:')
        self.ai = st.selectbox('Selecione um modelo de classificação:', ('', 'Logistic Regression', 'KNN', 'Random Forest', 'Decision Tree'))
        
        classification_option = {
            'Logistic Regression': self.__run_logistic_regression,
            'KNN': self.__run_knn,
            'Random Forest': self.__run_random_forest_classification,
            'Decision Tree': self.__run_decision_tree
        }
        
        if self.ai:
            classification_option[self.ai]()
    
    def __run_linear_regression(self) -> None:
        """
        Método para executar a regressão linear.
        """
        X = self.normalized_data.iloc[:, :-1]
        y = self.normalized_data.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        mse = mean_squared_error(y_test, predictions)
        st.write(f'MSE: {mse}')
        st.write('Predições:', predictions)
    
    def __run_svr(self) -> None:
        """
        Método para executar o SVR.
        """
        X = self.normalized_data.iloc[:, :-1]
        y = self.normalized_data.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = SVR()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        mse = mean_squared_error(y_test, predictions)
        st.write(f'MSE: {mse}')
        st.write('Predições:', predictions)
    
    def __run_random_forest_regression(self) -> None:
        """
        Método para executar o Random Forest para regressão.
        """
        X = self.normalized_data.iloc[:, :-1]
        y = self.normalized_data.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        mse = mean_squared_error(y_test, predictions)
        st.write(f'MSE: {mse}')
        st.write('Predições:', predictions)
    
    # Métodos de classificação ainda precisam ser implementados
    def __run_logistic_regression(self) -> None:
        """
        Método para executar a regressão logística.
        """
        st.write('Em desenvolvimento...')  # Implemente a lógica para regressão logística aqui
    
    def __run_knn(self) -> None:
        """
        Método para executar o KNN para classificação.
        """
        st.write('Em desenvolvimento...')  # Implemente a lógica para KNN aqui
    
    def __run_random_forest_classification(self) -> None:
        """
        Método para executar o Random Forest para classificação.
        """
        st.write('Em desenvolvimento...')  # Implemente a lógica para Random Forest de classificação aqui
    
    def __run_decision_tree(self) -> None:
        """
        Método para executar a árvore de decisão para classificação.
        """
        st.write('Em desenvolvimento...')  # Implemente a lógica para árvore de decisão aqui