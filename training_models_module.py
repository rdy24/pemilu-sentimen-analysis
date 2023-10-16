from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class TrainingModel(Base):
    __tablename__ = 'training'

    id = Column(Integer, primary_key=True, autoincrement=True, unique=True)
    teks = Column(String)
    label = Column(String)
    sosmed = Column(String)
