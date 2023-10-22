from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class TrainingModel(Base):
    __tablename__ = 'training'

    id = Column(Integer, primary_key=True, autoincrement=True, unique=True)
    teks = Column(String)
    label = Column(String)
    sosmed = Column(String)

class PreprocessingModel(Base):
    __tablename__ = 'preprocessing'

    id = Column(Integer, primary_key=True, autoincrement=True, unique=True)
    teks = Column(String)
    hasil = Column(String)
    label = Column(String)
    sosmed = Column(String)

class TFIDFModel(Base):
    __tablename__ = 'tfidf'

    id = Column(Integer, primary_key=True, autoincrement=True, unique=True)
    document_id = Column(Integer)
    term = Column(String)
    tf = Column(Float)
    idf = Column(Float)
    tfidf = Column(Float)
