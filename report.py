import streamlit as st
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine,func
from models import TBAiActions, TBPrimaryActions 
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
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
        
        total_preprocessing = session.query(func.count(TBPrimaryActions.id)).filter(TBPrimaryActions.is_ai == False).scalar()
        total_ai_processing = session.query(func.count(TBPrimaryActions.id)).filter(TBPrimaryActions.is_ai == True).scalar()
        
        preprocessing_details = session.query(TBPrimaryActions.dataset_name, func.count(TBPrimaryActions.id)).filter(TBPrimaryActions.is_ai == False).group_by(TBPrimaryActions.dataset_name).all()
        
        
        return total_preprocessing, total_ai_processing, preprocessing_details
    
    @staticmethod
    def __get_ai_actions_by_dataset(dataset_name: str) -> List[TBAiActions]:
        """
        Obtém as ações de IA para um dataset específico.
        
        Args:
            dataset_name (str): Nome do dataset.
        
        Returns:
            List[TBAiActions]: Lista de ações de IA para o dataset selecionado.
        """
        
        ai_actions = session.query(TBAiActions).join(TBPrimaryActions).filter(TBPrimaryActions.dataset_name == dataset_name).all()
        
        
        return ai_actions

    def run(self) -> None:
        """
        Executa o dashboard Streamlit para visualizar os relatórios de ações de dados.
        """
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
                        'MSE': action.metrics['mse'] if 'mse' in action.metrics else '',
                        'Accuracy': action.metrics['accuracy'] if 'accuracy' in action.metrics else '',
                    } for action in ai_actions]

                    metrics = [
                        {
                            'Predictions': action.metrics['predictions'] if action.metrics and 'predictions' in action.metrics else None,
                            'Real Values': action.metrics['real_values'] if action.metrics and 'real_values' in action.metrics else None
                        } 
                        for action in ai_actions
                    ]
                    
                    df_metrics = pd.DataFrame(metrics)
                    df_ai = pd.DataFrame(ai_actions_data)
                    
                    st.subheader('Detalhes das Ações de IA')
                    st.dataframe(df_ai)
                    
                    st.subheader('Métricas')
                    st.dataframe(df_metrics)
                    from main import Dashboard

                    Dashboard().download_spreadsheet(df_ai, 'ai_actions.xlsx')
                        
                else:
                    st.write("Nenhuma ação de IA encontrada para o dataset selecionado.")
        else:
            st.write("Nenhum pré-processamento encontrado.")