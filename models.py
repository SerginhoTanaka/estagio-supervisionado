from sqlalchemy import create_engine, Column, Integer, String, Boolean, ForeignKey, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime

# Configuração básica do SQLAlchemy
Base = declarative_base()
engine = create_engine('sqlite:///actions.db')
Session = sessionmaker(bind=engine)
session = Session()

# Definição das tabelas
class TBPrimaryActions(Base):
    __tablename__ = 'tb_primary_actions'
    id = Column(Integer, primary_key=True, autoincrement=True)
    action_name = Column(String, nullable=False)  # Nome da ação principal
    dataset_name = Column(String, nullable=True)  # Nome do dataset utilizado
    is_ai = Column(Boolean, default=False)  # Indica se a ação envolve IA
    timestamp = Column(DateTime, default=datetime.utcnow)  # Momento da ação
    ai_actions = relationship("DBAiActions", back_populates="primary_action", cascade="all, delete-orphan")  # Relação 1:N com AiActions

class TBAiActions(Base):
    __tablename__ = 'tb_ai_actions'
    id = Column(Integer, primary_key=True, autoincrement=True)
    paradigm = Column(String, nullable=False)  # Pode ser 'Classification' ou 'Regression'
    model = Column(String, nullable=False)  # Nome do modelo utilizado (e.g., 'Random Forest')
    target_column = Column(String, nullable=False)  # Nome da coluna alvo
    metrics = Column(JSON, nullable=True)  # Métricas do modelo em formato JSON
    primary_action_id = Column(Integer, ForeignKey('db_primary_actions.id'), nullable=False)  # Chave estrangeira para PrimaryActions
    primary_action = relationship("DBPrimaryActions", back_populates="ai_actions")  # Relação N:1 com PrimaryActions


Base.metadata.create_all(engine)
