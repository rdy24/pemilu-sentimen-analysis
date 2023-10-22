#!/usr/bin/env python
from flask import Flask, url_for, render_template, request, jsonify, redirect
import jinja2.exceptions
import pandas as pd
import mysql.connector
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from training_models_module import TrainingModel, PreprocessingModel, TFIDFModel
from preprocessing import cleaning_text, case_folding, tokenizing, stopword_removal, stemming, normalisasi
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Database connection
db_user = 'root'
db_password = ''
db_host = 'localhost'
db_name = 'pemilu-sentimen-analysis'

# Membuat koneksi ke database MySQL dengan mysql-connector-python
db_connection = mysql.connector.connect(
    host=db_host,
    user=db_user,
    password=db_password,
    database=db_name
)

engine = create_engine('mysql+mysqlconnector://', creator=lambda: db_connection)
Session = sessionmaker(bind=engine)
session = Session()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/training-data')
def training_data():
    data = session.query(TrainingModel).all()
    return render_template('trainingData.html', data=data)


@app.route('/upload-data-training', methods=['POST'])
def upload_data_training():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'File CSV tidak ditemukan'})

        file = request.files['file']

        if file and file.filename.endswith('.csv'):
            try:
                session.execute(text('TRUNCATE TABLE training'))
                session.execute(text('TRUNCATE TABLE preprocessing'))
                session.commit()
                df = pd.read_csv(file)

                if 'teks' in df.columns and 'label' in df.columns and 'sosmed' in df.columns:
                    for _, row in df.iterrows():
                        df.fillna('', inplace=True)
                        # Membuat objek model dan menyimpannya ke database
                        data = TrainingModel(teks=row['teks'], label=row['label'], sosmed=row['sosmed'])
                        session.add(data)

                    session.commit()
                    return redirect(url_for('training_data'))
                else:
                    return jsonify({'error': 'Kolom "teks", "label", dan "sosmed" diperlukan dalam file CSV'})
            except Exception as e:
                return jsonify({'error': f'Error: {str(e)}'})
        else:
            return jsonify({'error': 'File harus berformat CSV'})
    else:
        return jsonify({'error': 'Metode HTTP tidak valid, hanya mendukung POST'})


@app.route('/preprocessing-proses')
def preprocessing_proses():
    data = session.query(TrainingModel).all()
    session.execute(text('TRUNCATE TABLE preprocessing'))
    session.commit()
    # Melakukan preprocessing pada data
    for item in data:
        cleaned_text = cleaning_text(item.teks)
        lowercased_text = case_folding(cleaned_text)
        tokenized_text = tokenizing(lowercased_text)
        normalized_text = normalisasi(tokenized_text)
        stopword_removed_text = stopword_removal(normalized_text)
        stemmed_text = stemming(stopword_removed_text)

        # save data to database
        data = PreprocessingModel(teks=item.teks, hasil=(" ").join(stemmed_text), sosmed=item.sosmed, label=item.label)
        session.add(data)
        session.commit()
    return redirect(url_for('preprocessing'))


@app.route('/preprocessing')
def preprocessing():
    data = session.query(PreprocessingModel).all()
    return render_template('preprocessing.html', data=data)


@app.route('/tfidf-proses')
def tfidf_proses():
    session.execute(text('TRUNCATE TABLE tfidf'))
    session.commit()
    data = session.query(PreprocessingModel).all()

    for item in data:
        teks = item.hasil

        if len(teks) > 0:
            tfidf_vectorizer = TfidfVectorizer()
            tfidf_matrix = tfidf_vectorizer.fit_transform([teks])

            tfidf_scores = tfidf_matrix.toarray()[0]

            for i, term in enumerate(tfidf_vectorizer.get_feature_names_out()):
                tfidf_score = tfidf_scores[i]

            # Hitung nilai IDF dari seluruh corpus (jika perlu)
            idf_values = tfidf_vectorizer.idf_
            tf_values = tfidf_matrix.toarray()[0]

            for j, term in enumerate(tfidf_vectorizer.get_feature_names_out()):
                tf_score = tf_values[j]
                idf_score = idf_values[j]

                tfidf_model = TFIDFModel(document_id=item.id, term=term, tf=tf_score, idf=idf_score,
                                         tfidf=tfidf_score)
                session.add(tfidf_model)

    session.commit()

    return redirect(url_for('tfidf'))

@app.route('/tfidf')
def tfidf():
    data = session.query(TFIDFModel).all()
    tfidf = []
    for item in data:
        tfidf.append(
            {
                'document_id': item.document_id,
                'term': item.term,
                'tf': item.tf,
                'idf': item.idf,
                'tfidf': item.tfidf
            }
        )
    return jsonify(tfidf)
    # return render_template('tfidf.html', data=data)

@app.errorhandler(jinja2.exceptions.TemplateNotFound)
def template_not_found(e):
    return not_found(e)


@app.errorhandler(404)
def not_found(e):
    return render_template('404.html')


if __name__ == '__main__':
    app.run(debug=True)
