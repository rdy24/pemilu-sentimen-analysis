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

class UserModel(Base):
    __tablename__ = 'user'

    id = Column(Integer, primary_key=True, autoincrement=True, unique=True)
    nama = Column(String)
    email = Column(String)
    password = Column(String)
    role = Column(String, default='user')

    # Implementasikan metode is_active
    def is_active(self):
        return True  # Gantilah dengan logika aktivasi pengguna yang sesuai

    # Implementasikan metode get_id
    def get_id(self):
        return str(self.id)

    def is_authenticated(self):
        return True  # Atau sesuaikan dengan logika autentikasi yang sesuai

    # Implementasikan metode __str__ (opsional, untuk debugging)
    def __str__(self):
        return f"User ID: {self.id}, Username: {self.username}, Role: {self.role}"

class KlasifikasiTrainingModel(Base):
    __tablename__ = 'klasifikasi_training'

    id = Column(Integer, primary_key=True, autoincrement=True, unique=True)
    teks = Column(String)
    label = Column(String)
    hasil_klasifikasi = Column(String)

class TestingModel(Base):
    __tablename__ = 'testing'

    id = Column(Integer, primary_key=True, autoincrement=True, unique=True)
    teks = Column(String)
    preprocessing = Column(String)
    hasil_klasifikasi = Column(String)

class KlasifikasiTestingModel(Base):
    __tablename__ = 'klasifikasi_testing'

    id = Column(Integer, primary_key=True, autoincrement=True, unique=True)
    teks = Column(String)
    hasil_klasifikasi = Column(String)

class PrepocessingTestingModel(Base):
    __tablename__ = 'preprocessing_testing'

    id = Column(Integer, primary_key=True, autoincrement=True, unique=True)
    teks = Column(String)
    hasil = Column(String)

class ScrapingModel(Base):
    __tablename__ = 'scraping_tweet'

    id = Column(Integer, primary_key=True, autoincrement=True, unique=True)
    created_at = Column(String)
    teks = Column(String)
    preprocessing = Column(String)
    hasil_klasifikasi = Column(String)
