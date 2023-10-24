#!/usr/bin/env python
import pickle

from flask import Flask, url_for, render_template, request, jsonify, redirect
import jinja2.exceptions
import pandas as pd
import mysql.connector
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from training_models_module import TrainingModel, PreprocessingModel, TFIDFModel
from preprocessing import cleaning_text, case_folding, tokenizing, stopword_removal, stemming, normalisasi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import re

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
    countTraining = session.query(TrainingModel).count()
    return render_template('index.html', countTraining=countTraining)


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


@app.route('/tfidf-proses1')
def tfidf_proses1():
    data = session.query(PreprocessingModel).all()

    # Ekstrak teks dari data
    corpus = [item.hasil for item in data]

    # Buat objek TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Hitung TF-IDF
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    # Hitung IDF dari seluruh corpus
    idf_values = tfidf_vectorizer.idf_

    # Persiapan untuk menyimpan hasil sebagai dictionary
    tfidf_data = []

    # Hitung nilai TF, IDF, dan skor TF-IDF
    for i, item in enumerate(data):
        tfidf_scores = tfidf_matrix[i].toarray()[0]

        tfidf_item = {"document_id": item.id, "tfidf_scores": {}, "tf": {}, "idf": {}}

        for j, term in enumerate(tfidf_vectorizer.get_feature_names_out()):
            tf_score = tfidf_scores[j]
            idf_score = idf_values[j]

            # Hitung skor TF-IDF
            tfidf_score = tf_score * idf_score

            tfidf_item["tf"][term] = tf_score
            tfidf_item["idf"][term] = idf_score
            tfidf_item["tfidf_scores"][term] = tfidf_score

        tfidf_data.append(tfidf_item)

    return jsonify(tfidf_data)

@app.route('/tfidf-proses2')
def tfidf_proses2():
    data = session.query(PreprocessingModel).all()

    tfidf_results = []

    # Ekstrak teks dari semua dokumen
    corpus = [item.hasil for item in data]

    # Buat objek TfidfVectorizer dengan semua term dalam korpus
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    # Dapatkan nama term (kata)
    terms = tfidf_vectorizer.get_feature_names_out()

    for doc_index, text in enumerate(corpus):
        tfidf_scores = tfidf_matrix[doc_index].toarray()[0]

        # Menggabungkan kata, indeks dokumen, dan bobot
        result_items = [{"term": term, "document_id": doc_index, "tfidf_score": tfidf_score}
                        for term, tfidf_score in zip(terms, tfidf_scores)]

        tfidf_results.extend(result_items)

    return jsonify(tfidf_results)

@app.route('/klasifikasisvm')
def klasifikasisvm():
    # Ambil data dari database
    preprocessingData = session.query(PreprocessingModel).all()

    id = [item.id for item in preprocessingData]
    teks = [item.teks for item in preprocessingData]
    corpus = [item.hasil for item in preprocessingData]
    labels = [item.label for item in preprocessingData]

    # Perbarui label 'Kebencian' menjadi -1 dan 'Non-Kebencian' menjadi 1
    labels = [-1 if label == 'Kebencian' else 1 for label in labels]

    # Hitung TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    # Proses pembuatan model Klasifikasi SVM LINEAR
    linear = SVC(kernel="linear")
    linear.fit(tfidf_matrix, labels)
    hasil_linear = linear.predict(tfidf_matrix)

    # Proses pembuatan model Klasifikasi SVM RBF
    rbf = SVC(kernel="rbf")
    rbf.fit(tfidf_matrix, labels)
    hasil_rbf = rbf.predict(tfidf_matrix)

    result = []

    for i in range(len(teks)):
        result.append({
            'id': id[i],
            'teks': teks[i],
            'label_asli': "Kebencian" if labels[i] == -1 else 'Non-Kebencian',
            'hasil_klasifikasi_linear': "Kebencian" if hasil_linear[i] == -1 else 'Non-Kebencian',
            'hasil_klasifikasi_rbf': "Kebencian" if hasil_rbf[i] == -1 else 'Non-Kebencian'
        })

    return jsonify(result)


@app.route('/klasifikasisvm1')
def klasifikasisvm1():
    # Ambil data dari database
    preprocessingData = session.query(PreprocessingModel).all()

    corpus = [item.hasil for item in preprocessingData]
    labels = [item.label for item in preprocessingData]

    # Perbarui label 'Kebencian' menjadi -1 dan 'Non-Kebencian' menjadi 1
    labels = [-1 if label == 'Kebencian' else 1 for label in labels]

    # Hitung TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    # Bagi data menjadi data pelatihan (80%) dan data pengujian (20%)
    X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, labels, test_size=0.2, random_state=0)

    with open('tfidf_vectorizer.pkl', 'wb') as tfidf_vectorizer_file:
        pickle.dump(tfidf_vectorizer, tfidf_vectorizer_file)

    # Proses pembuatan model Klasifikasi SVM LINEAR
    linear = SVC(kernel="linear")
    linear.fit(X_train, y_train)

    # Menyimpan model SVM linear ke dalam file
    with open('svm_linear_model.pkl', 'wb') as model_file:
        pickle.dump(linear, model_file)

    # Memuat kembali model yang telah disimpan
    with open('svm_linear_model.pkl', 'rb') as model_file:
        loaded_linear_model = pickle.load(model_file)

    # Mendapatkan label prediksi dari model yang telah dimuat
    hasil_linear = loaded_linear_model.predict(X_test)

    # Menghitung metrik kinerja
    accuracy = accuracy_score(y_test, hasil_linear)
    precision = precision_score(y_test, hasil_linear)
    recall = recall_score(y_test, hasil_linear)
    f1 = f1_score(y_test, hasil_linear)
    cm = confusion_matrix(y_test, hasil_linear)

    # Siapkan data teks dan label untuk respons JSON
    data = []
    for i in range(len(y_test)):
        data.append({
            'teks': corpus[i],
            'label_asli': 'Kebencian' if y_test[i] == -1 else 'Non-Kebencian',
            'hasil_linear': 'Kebencian' if hasil_linear[i] == -1 else 'Non-Kebencian'
        })

    # Siapkan respons JSON
    response = {
        'data': data,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm.tolist()
        }
    }

    return jsonify(response)

@app.route('/testing-data')
def testing_data():
    teks = "pemimpinnya bodoh ngapain milih orang bodoh"
    cleaned = cleaning_text(teks)
    lowercased = case_folding(cleaned)
    tokenized = tokenizing(lowercased)
    normalized = normalisasi(tokenized)
    stopword_removed = stopword_removal(normalized)
    stemmed = stemming(stopword_removed)

    preprocessed_text = (" ").join(stemmed)

    with open('tfidf_vectorizer.pkl', 'rb') as tfidf_vectorizer_file:
        loaded_tfidf_vectorizer = pickle.load(tfidf_vectorizer_file)

    with open('svm_linear_model.pkl', 'rb') as model_file:
        loaded_linear_model = pickle.load(model_file)

    hasil_linear = loaded_linear_model.predict(loaded_tfidf_vectorizer.transform([preprocessed_text]))

    return jsonify({
        'teks': teks,
        'preprocessed_text': preprocessed_text,
        'hasil_linear': 'Kebencian' if hasil_linear[0] == -1 else 'Non-Kebencian'
    })


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
