import streamlit as st
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from models import DBAiActions, DBPrimaryActions 
import pandas as pd

engine = create_engine('sqlite:///actions.db')
Session = sessionmaker(bind=engine)
session = Session()
class ReportsDashboard:
    """
    Classe para visualizar e analisar os relatórios salvos no banco de dados.
    """


    def run(self) -> None:
        """
        Executa o dashboard de relatórios.
        """
        st.sidebar.title("Relatórios")
        selected_option: str = st.sidebar.radio(
            "Selecione uma opção",
            ["Listar Relatórios", "Visualizar Relatório"]
        )

        if selected_option == "Listar Relatórios":
            self.__list_reports()
        elif selected_option == "Visualizar Relatório":
            self.__view_report()

    def __list_reports(self) -> None:
        """
        Lista todos os relatórios salvos no banco de dados e permite ao usuário escolher um relatório para visualizar.
        """
        # Consulta todas as ações primárias
        primary_actions = session.query(DBPrimaryActions).all()

        # Cria uma lista para armazenar os IDs e nomes das ações primárias
        action_options = [(action.id, action.action_name) for action in primary_actions]

        # Adiciona a opção de selecionar "Todos"
        action_options.insert(0, (None, "Todos"))

        # Permite ao usuário selecionar uma ação primária ou "Todos"
        selected_action_id, selected_action_name = st.selectbox(
            "Selecione uma Ação Primária",
            action_options,
            format_func=lambda option: f"{option[1]} (ID: {option[0]})"
        )

        # Filtra os relatórios com base na ação primária selecionada
        if selected_action_id is None:
            reports = session.query(DBAiActions).all()
        else:
            reports = session.query(DBAiActions).filter(DBAiActions.primary_action_id == selected_action_id).all()

        # Conta os relatórios de pré-processamento e processamento com IA
        preprocessing_count = sum(1 for report in reports if not report.is_ai)
        ai_processing_count = sum(1 for report in reports if report.is_ai)

        st.write(f"Total de Relatórios de Pré-processamento: {preprocessing_count}")
        st.write(f"Total de Relatórios de Processamento com IA: {ai_processing_count}")

        if reports:
            st.write("Relatórios Salvos:")
            report_options = [(report.id, f"Modelo: {report.model}, Paradigma: {report.paradigm}, Data: {report.timestamp}") for report in reports]
            
            selected_report_id = st.selectbox(
                "Selecione um Relatório para Visualizar",
                [option[0] for option in report_options],
                format_func=lambda id: next(option[1] for option in report_options if option[0] == id)
            )
            
            if selected_report_id:
                report = session.query(DBAiActions).filter(DBAiActions.id == selected_report_id).first()
                if report:
                    st.write(f"**Modelo:** {report.model}")
                    st.write(f"**Paradigma:** {report.paradigm}")
                    st.write(f"**Data:** {report.timestamp}")
                    st.write(f"**Métricas e Resultados:**")
                    
                    metrics = report.metrics
                    st.write(f"**Métricas:** {metrics.get('mse', metrics.get('accuracy', 'N/A'))}")
                    
                    # Exibir valores reais e predições se disponíveis
                    if 'real_values' in metrics and 'predictions' in metrics:
                        results_df = pd.DataFrame({
                            'Real': metrics['real_values'],
                            'Predição': metrics['predictions']
                        })
                        st.write('Resultados:')
                        st.dataframe(results_df, use_container_width=True)
                else:
                    st.error("Relatório não encontrado.")
        else:
            st.write("Nenhum relatório encontrado para a seleção.")

    def __view_report(self) -> None:
        """
        Visualiza um relatório específico selecionado pelo usuário filtrado pela ação primária.
        """
        primary_actions = session.query(DBPrimaryActions).all()
        selected_action_id = st.selectbox("Selecione uma Ação Primária", [action.id for action in primary_actions], format_func=lambda id: f"ID: {id}")
        
        if selected_action_id:
            report_ids = [report.id for report in session.query(DBAiActions).filter(DBAiActions.primary_action_id == selected_action_id).all()]
            selected_id = st.selectbox("Selecione um relatório", report_ids)

            if selected_id is not None:
                report = session.query(DBAiActions).filter(DBAiActions.id == selected_id).first()
                if report:
                    st.write(f"**Modelo:** {report.model}")
                    st.write(f"**Paradigma:** {report.paradigm}")
                    st.write(f"**Data:** {report.timestamp}")
                    st.write(f"**Métricas e Resultados:**")
                    
                    metrics = report.metrics
                    st.write(f"**Métricas:** {metrics.get('mse', metrics.get('accuracy', 'N/A'))}")
                    
                    # Exibir valores reais e predições se disponíveis
                    if 'real_values' in metrics and 'predictions' in metrics:
                        results_df = pd.DataFrame({
                            'Real': metrics['real_values'],
                            'Predição': metrics['predictions']
                        })
                        st.write('Resultados:')
                        st.dataframe(results_df, use_container_width=True)
                else:
                    st.error("Relatório não encontrado.")
        else:
            st.write("Nenhuma ação primária selecionada.")