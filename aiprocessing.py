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
from sklearn.metrics import mean_squared_error, accuracy_score

class AiProcessing:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.ai = None
        self.normalized_data = None
        self.target_column = None

    def run(self) -> None:
        """
        Método principal para executar o fluxo de pré-processamento e seleção do modelo de IA.
        """
        method = st.selectbox('Selecione um modelo de IA:', ('', 'Regressão', 'Classificação'))
        preprocessing_option = st.selectbox('Deseja pré-processar os dados?', ('', 'Sim', 'Não'))
        
        if preprocessing_option == 'Sim':
            self.normalized_data = self.__preprocessing()
        else:
            numerical_df = self.data.select_dtypes(exclude=['object'])
            preprocessing = Preprocessing(self.data)
            final_data = preprocessing.preprocess_categorical_data(self.data,numerical_df)
            self.data = final_data
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
        Método genérico para executar a regressão.
        """
        st.subheader('Modelos de Regressão Disponíveis:')
        self.target_column = st.selectbox('Selecione a coluna alvo para a regressão:', self.data.columns)
        self.ai = st.selectbox('Selecione um modelo de regressão:', 
                               ('', 'Linear Regression', 'SVR', 'Random Forest'))

        if self.ai:
            model = self.__get_model()
            self.__train_and_evaluate(model, regression=True)
    
    def __classification(self) -> None:
        """
        Método genérico para executar a classificação.
        """
        st.subheader('Modelos de Classificação Disponíveis:')
        self.target_column = st.selectbox('Selecione a coluna alvo para a classificação:', self.data.columns)
        self.ai = st.selectbox('Selecione um modelo de classificação:', 
                               ('', 'Logistic Regression', 'KNN', 'Random Forest', 'Decision Tree'))

        if self.ai:
            model = self.__get_model()
            self.__train_and_evaluate(model, regression=False)
    
    def __get_model(self):
        """
        Retorna o modelo de IA baseado na escolha do usuário.
        """
        models = {
            'Linear Regression': LinearRegression(),
            'SVR': SVR(),
            'Random Forest': RandomForestRegressor(),
            'Logistic Regression': LogisticRegression(),
            'KNN': KNeighborsClassifier(),
            # 'Random Forest Classifier': RandomForestClassifier(),
            'Decision Tree': DecisionTreeClassifier()
        }
        return models[self.ai]

    def __split_data(self) -> (pd.DataFrame, pd.Series):
        """
        Método auxiliar para separar as colunas X e y do DataFrame.
        """
        data_to_use = self.normalized_data if self.normalized_data is not None else self.data
        X = data_to_use.drop(columns=[self.target_column])
        y = data_to_use[self.target_column]
        return X, y

    def __train_and_evaluate(self, model, regression: bool) -> None:
        """
        Método para treinar e avaliar o modelo selecionado.
        """
        X, y = self.__split_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Exibe métricas de desempenho
        if regression:
            mse = mean_squared_error(y_test, predictions)
            st.write(f'MSE: {mse}')
        else:
            accuracy = accuracy_score(y_test, predictions)
            st.write(f'Acurácia: {accuracy}')
        
        # Exibe tabela com y_test e as previsões
        results_df = pd.DataFrame({'Real': y_test, 'Predição': predictions})

        # Redefinindo o índice para removê-lo
        results_df.reset_index(drop=True, inplace=True)

        st.write('Resultados:')
        st.dataframe(results_df, use_container_width=True)
