import streamlit as st
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, func
from models import TBAiActions, TBPrimaryActions
import pandas as pd
from typing import List, Tuple

# Configuração do banco de dados
engine = create_engine('sqlite:///actions.db')
Session = sessionmaker(bind=engine)
session = Session()

class ReportsDashboard:
    """
    Classe para visualizar e analisar os relatórios salvos no banco de dados.
    """

    @staticmethod
    def __get_report_data() -> Tuple[int, int, List[Tuple[str, int]]]:
        """
        Obtém dados do relatório do banco de dados.

        Returns:
            total_preprocessing (int): Total de pré-processamentos.
            total_ai_processing (int): Total de processamentos com IA.
            preprocessing_details (List[Tuple[str, int]]): Detalhes dos pré-processamentos.
        """
        try:
            total_preprocessing = session.query(func.count(TBPrimaryActions.id)).filter(TBPrimaryActions.is_ai == False).scalar()
            total_ai_processing = session.query(func.count(TBPrimaryActions.id)).filter(TBPrimaryActions.is_ai == True).scalar()

            preprocessing_details = session.query(TBPrimaryActions.dataset_name, func.count(TBPrimaryActions.id))\
                                           .filter(TBPrimaryActions.is_ai == False)\
                                           .group_by(TBPrimaryActions.dataset_name).all()

            return total_preprocessing, total_ai_processing, preprocessing_details
        except Exception as e:
            st.error(f"Ocorreu um erro ao obter os dados do relatório: {e}")
            return 0, 0, []

    @staticmethod
    def __get_ai_actions_by_dataset(dataset_name: str) -> List[TBAiActions]:
        """
        Obtém as ações de IA para um dataset específico.

        Args:
            dataset_name (str): Nome do dataset.

        Returns:
            List[TBAiActions]: Lista de ações de IA para o dataset selecionado.
        """
        try:
            ai_actions = session.query(TBAiActions).join(TBPrimaryActions)\
                             .filter(TBPrimaryActions.dataset_name == dataset_name).all()
            return ai_actions
        except Exception as e:
            st.error(f"Ocorreu um erro ao obter as ações de IA para o dataset '{dataset_name}': {e}")
            return []

    def run(self) -> None:
        """
        Executa o dashboard Streamlit para visualizar os relatórios de ações de dados.
        """
        try:
            total_preprocessing, total_ai_processing, preprocessing_details = self.__get_report_data()

            st.title('Relatório de Ações de Dados')
            st.subheader('Total de Ações')
            st.write(f"Total de Pré-processamentos: {total_preprocessing}")
            st.write(f"Total de Processamentos com IA: {total_ai_processing}")

            st.subheader('Detalhes dos Pré-processamentos')
            if preprocessing_details:
                df = pd.DataFrame(preprocessing_details, columns=['Dataset', 'Quantidade'])
                st.dataframe(df)

                dataset_options = [detail[0] for detail in preprocessing_details]
                selected_dataset = st.selectbox('Escolha o dataset', dataset_options)

                if selected_dataset:
                    ai_actions = self.__get_ai_actions_by_dataset(selected_dataset)

                    if ai_actions:
                        ai_actions_data = [{
                            'Paradigma': action.paradigm,
                            'Modelo': action.model,
                            'Coluna Alvo': action.target_column,
                            'MSE': action.metrics.get('mse', ''),
                            'Accuracy': action.metrics.get('accuracy', '')
                        } for action in ai_actions]

                        metrics = [{
                            'Predictions': action.metrics.get('predictions', None),
                            'Real Values': action.metrics.get('real_values', None)
                        } for action in ai_actions]

                        df_ai = pd.DataFrame(ai_actions_data)
                        df_metrics = pd.DataFrame(metrics)
                        st.subheader('Detalhes das Ações de IA')
                        st.dataframe(df_ai)

                        st.subheader('Métricas')
                        st.dataframe(df_metrics)

                        from main import Dashboard
                        Dashboard().download_spreadsheet(df_metrics, 'ai_actions.csv')
        except Exception as e:
            st.error(f"Ocorreu um erro ao carregar o dashboard: {e}")