#!/usr/bin/env python
import pickle

from flask import Flask, url_for, render_template, request, jsonify, redirect, flash, g
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import jinja2.exceptions
import pandas as pd
import mysql.connector
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from models import db, TrainingModel, PreprocessingModel, TFIDFModel, UserModel, KlasifikasiTrainingModel, TestingModel, PrepocessingTestingModel, ScrapingModel
from preprocessing import cleaning_text, case_folding, tokenizing, stopword_removal, stemming, normalisasi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from werkzeug.security import generate_password_hash, check_password_hash
import os
import datetime
from svm import Svm
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/pemilu-sentimen-analysis'
db.init_app(app)
app.secret_key = 'your_secret_key'
login_manager = LoginManager(app)
login_manager.login_view = 'login'

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
def home():
    return render_template('index.html')
    # return redirect(url_for('login'))


@login_manager.user_loader
def load_user(user_id):
    return UserModel.query.get(int(user_id))

@app.before_request
def before_request():
    g.name = None
    g.role = None
    if current_user.is_authenticated:
        g.name = current_user.nama
        g.role = current_user.role



@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:  # Mengecek apakah pengguna sudah login
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = UserModel.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            if user.role == 'admin':
                return redirect(url_for('dashboard'))
            elif user.role == 'user':
                return redirect(url_for('dashboard'))
        else:
            flash('Login gagal. Periksa kembali username dan password Anda.', 'danger')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:  # Mengecek apakah pengguna sudah login
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        nama = request.form.get('nama')
        email = request.form.get('email')
        password = request.form.get('password')

        if not nama or not email or not password:
            flash('Semua kolom harus diisi.', 'danger')
        else:
            hashed_password = generate_password_hash(password)
            user = UserModel(nama=nama, password=hashed_password, email=email)
            session.add(user)
            session.commit()
            flash('Akun Anda telah dibuat! Silakan masuk.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    countTraining = TrainingModel.query.count()
    countTesting = TestingModel.query.count()
    return render_template('dashboard.html', countTraining=countTraining, countTesting=countTesting)


@app.route('/training-data')
@login_required
def training_data():
    if current_user.role == 'user':
        return redirect(url_for('dashboard'))
    data = TrainingModel.query.all()
    return render_template('trainingData.html', data=data)


@app.route('/upload-data-training', methods=['POST'])
@login_required
def upload_data_training():
    if current_user.role == 'user':
        return redirect(url_for('dashboard'))
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
                df.drop_duplicates(subset=['teks'], inplace=True)

                if 'teks' in df.columns and 'label' in df.columns:
                    for _, row in df.iterrows():
                        df.fillna('', inplace=True)
                        # Membuat objek model dan menyimpannya ke database
                        data = TrainingModel(teks=row['teks'], label=row['label'])
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
@login_required
def preprocessing_proses():
    data = TrainingModel.query.all()
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


@app.route('/preprocessing-training')
@login_required
def preprocessing():
    data = PreprocessingModel.query.all()
    return render_template('preprocessing.html', data=data)


@app.route('/tfidf-proses')
@login_required
def tfidf_proses():
    session.execute(text('TRUNCATE TABLE tfidf'))
    session.commit()
    data = PreprocessingModel.query.all()

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


@app.route('/klasifikasi-training')
@login_required
def klasifikasisvm():
    # Ambil data dari database
    preprocessingData = PreprocessingModel.query.all()
    session.execute(text('TRUNCATE TABLE klasifikasi_training'))

    teks = [item.teks for item in preprocessingData]
    corpus = [item.hasil for item in preprocessingData]
    labels = [item.label for item in preprocessingData]

    # Perbarui label 'Kebencian' menjadi -1 dan 'Non-Kebencian' menjadi 1
    labels = [1 if label == 'Non-Kebencian' else -1 for label in labels]

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    with open('tfidf_vectorizer.pkl', 'wb') as tfidf_vectorizer_file:
        pickle.dump(tfidf_vectorizer, tfidf_vectorizer_file)

    # Split the resampled data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, labels, test_size=0.2, random_state=0)

    # Train the SVM model on the resampled data

    linear = SVC(kernel="linear", C=1.0, random_state=0)
    model = linear.fit(X_train, y_train)

    # Menyimpan model SVM linear ke dalam file
    with open('svm_linear_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    # Memuat kembali model yang telah disimpan
    with open('svm_linear_model.pkl', 'rb') as model_file:
        loaded_linear_model = pickle.load(model_file)

    # buka file tfidf vectorizer
    with open('tfidf_vectorizer.pkl', 'rb') as tfidf_vectorizer_file:
        loaded_tfidf_vectorizer = pickle.load(tfidf_vectorizer_file)

    # Transform the training data using the loaded TF-IDF vectorizer
    X_train_transformed = loaded_tfidf_vectorizer.transform(corpus)

    # Predict on the training data using the loaded SVM model
    hasil_linear_train = loaded_linear_model.predict(X_train_transformed)

    # save data to database
    for i in range(len(labels)):
        data = KlasifikasiTrainingModel(teks=teks[i], label=labels[i], hasil_klasifikasi=hasil_linear_train[i])
        session.add(data)

    # count true positive, true negative, false positive, false negative
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    data = []
    for i in range(len(labels)):
        if labels[i] == -1 and hasil_linear_train[i] == -1:  # true negative
            tn += 1
        elif labels[i] == 1 and hasil_linear_train[i] == 1:  # true positive
            tp += 1
        elif labels[i] == -1 and hasil_linear_train[i] == 1:  # false positive
            fp += 1
        elif labels[i] == 1 and hasil_linear_train[i] == -1:  # false negative
            fn += 1
        data.append({
            'teks': teks[i],
            'label': "Kebencian" if labels[i] == -1 else "Non-Kebencian",
            'hasil': "Kebencian" if hasil_linear_train[i] == -1 else "Non-Kebencian",
        })
    print(f'True Positive (TP): {tp}')
    print(f'True Negative (TN): {tn}')
    print(f'False Positive (FP): {fp}')
    print(f'False Negative (FN): {fn}')
    print(len(labels))

    return render_template('klasifikasiTraining.html', data=data, true_positive=tp, true_negative=tn, false_positive=fp, false_negative=fn)

@app.route('/klasifikasi-training-buat')
@login_required
def klasifikasisvmBuat():
    # Ambil data dari database
    preprocessingData = PreprocessingModel.query.all()
    session.execute(text('TRUNCATE TABLE klasifikasi_training'))

    teks = [item.teks for item in preprocessingData]
    corpus = [item.hasil for item in preprocessingData]
    labels = [item.label for item in preprocessingData]

    # Perbarui label 'Kebencian' menjadi -1 dan 'Non-Kebencian' menjadi 1
    labels = [1 if label == 'Non-Kebencian' else -1 for label in labels]

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    with open('tfidf_vectorizer.pkl', 'wb') as tfidf_vectorizer_file:
        pickle.dump(tfidf_vectorizer, tfidf_vectorizer_file)

    # Split the resampled data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, labels, test_size=0.2, random_state=0)
    X_train_dense = X_train.toarray()

    # Ganti penggunaan SVC dengan Svm
    svm_model = Svm()
    model = svm_model.train(X_train_dense, y_train)

    # Menyimpan model SVM buatan ke dalam file
    with open('svm_model.pkl', 'wb') as model_file:
        pickle.dump(svm_model, model_file)

    # Memuat kembali model yang telah disimpan
    with open('svm_model.pkl', 'rb') as model_file:
        loaded_svm_model = pickle.load(model_file)

    # buka file tfidf vectorizer
    with open('tfidf_vectorizer.pkl', 'rb') as tfidf_vectorizer_file:
        loaded_tfidf_vectorizer = pickle.load(tfidf_vectorizer_file)

    # Transform the training data using the loaded TF-IDF vectorizer
    X_train_transformed = loaded_tfidf_vectorizer.transform(corpus)

    # Predict on the training data using the loaded SVM model
    hasil_svm_train = loaded_svm_model.predict(X_train_transformed)

    # save data to database
    for i in range(len(labels)):
        data = KlasifikasiTrainingModel(teks=teks[i], label=labels[i], hasil_klasifikasi=hasil_svm_train[i])
        session.add(data)

    # count true positive, true negative, false positive, false negative
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    data = []
    for i in range(len(labels)):
        if labels[i] == -1 and hasil_svm_train[i] == -1:  # true negative
            tn += 1
        elif labels[i] == 1 and hasil_svm_train[i] == 1:  # true positive
            tp += 1
        elif labels[i] == -1 and hasil_svm_train[i] == 1:  # false positive
            fp += 1
        elif labels[i] == 1 and hasil_svm_train[i] == -1:  # false negative
            fn += 1
        data.append({
            'teks': teks[i],
            'label': "Kebencian" if labels[i] == -1 else "Non-Kebencian",
            'hasil': "Kebencian" if hasil_svm_train[i] == -1 else "Non-Kebencian",
        })
    print(f'True Positive (TP): {tp}')
    print(f'True Negative (TN): {tn}')
    print(f'False Positive (FP): {fp}')
    print(f'False Negative (FN): {fn}')
    print(len(labels))

    return render_template('klasifikasiTraining.html', data=data, true_positive=tp, true_negative=tn, false_positive=fp, false_negative=fn)



@app.route('/cek-kalimat')
@login_required
def cek_kalimat():
    # get query string
    teks = request.args.get('kalimat')

    if teks is None:
        return render_template('cekKalimat.html')

    # teks = "pemimpinnya pintar ngapain milih orang pintar yang bisa memimpin negara ini dengan baik"
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

    hasil = "Kebencian" if hasil_linear[0] == -1 else 'Non-Kebencian'

    badge = "badge-danger" if hasil_linear[0] == -1 else 'badge-success'

    # return jsonify({
    #     'teks': teks,
    #     'preprocessed_text': preprocessed_text,
    #     'hasil_linear': 'Kebencian' if hasil_linear[0] == -1 else 'Non-Kebencian'
    # })

    return render_template('cekKalimat.html', teks=teks, preprocessed_text=preprocessed_text, hasil=hasil, badge=badge)


@app.route('/tfidf')
@login_required
def tfidf():
    data = TFIDFModel.query.all()
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


@app.route('/testing-data')
@login_required
def testing_data():
    data = TestingModel.query.all()
    countKebencian = TestingModel.query.filter_by(hasil_klasifikasi='Kebencian').count()
    countNonKebencian = TestingModel.query.filter_by(hasil_klasifikasi='Non-Kebencian').count()

    return render_template('testingData.html', data=data, countKebencian=countKebencian, countNonKebencian=countNonKebencian)

@app.route('/upload-data-testing', methods=['POST'])
@login_required
def upload_data_testing():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'File CSV tidak ditemukan'})

        file = request.files['file']

        if file and file.filename.endswith('.csv'):
            try:
                session.execute(text('TRUNCATE TABLE testing'))
                session.execute(text('TRUNCATE TABLE preprocessing_testing'))
                session.commit()
                df = pd.read_csv(file)

                if 'teks' in df.columns:
                    preprocessed_texts = []
                    for _, row in df.iterrows():
                        df.fillna('', inplace=True)
                        cleaned = cleaning_text(row['teks'])
                        lowercased = case_folding(cleaned)
                        tokenized = tokenizing(lowercased)
                        normalized = normalisasi(tokenized)
                        stopword_removed = stopword_removal(normalized)
                        stemmed = stemming(stopword_removed)
                        preprocessed_text = (" ").join(stemmed)
                        preprocessed_texts.append(preprocessed_text)

                    #add column
                    df['preprocessed_text'] = preprocessed_texts

                    corpus = [item for item in preprocessed_texts]

                    # load model
                    with open('tfidf_vectorizer.pkl', 'rb') as tfidf_vectorizer_file:
                        loaded_tfidf_vectorizer = pickle.load(tfidf_vectorizer_file)

                    # Transform the training data using the loaded TF-IDF vectorizer
                    X_train_transformed = loaded_tfidf_vectorizer.transform(corpus)

                    # Memuat kembali model yang telah disimpan
                    with open('svm_linear_model.pkl', 'rb') as model_file:
                        loaded_linear_model = pickle.load(model_file)

                    # Predict on the training data using the loaded SVM model
                    hasil_linear_train = loaded_linear_model.predict(X_train_transformed)
                    df['hasil'] = hasil_linear_train

                    # truncate table
                    session.execute(text('TRUNCATE TABLE testing'))
                    # save data to database
                    for _, row in df.iterrows():
                        hasil = "Kebencian" if row['hasil'] == -1 else 'Non-Kebencian'
                        data = TestingModel(teks=row['teks'], preprocessing=row['preprocessed_text'],hasil_klasifikasi=hasil)
                        session.add(data)

                    session.commit()
                    return redirect(url_for('testing_data'))
                else:
                    return jsonify({'error': 'Kolom "teks", diperlukan dalam file CSV'})
            except Exception as e:
                return jsonify({'error': f'Error: {str(e)}'})
        else:
            return jsonify({'error': 'File harus berformat CSV'})
    else:
        return jsonify({'error': 'Metode HTTP tidak valid, hanya mendukung POST'})


@app.route('/scrap-tweet', methods=['POST', 'GET'])
@login_required
def scrap_tweet():
    if request.method == 'POST':
        # Get user input
        twitter_auth_token = '151174fc0bd66d4866e87950d73346c50013d4b7'
        keyword = request.form['keyword']

        search_keyword = f'{keyword} lang:id -filter:links -filter:replies'

        if ' ' in keyword:
            keyword = keyword.replace(' ', '_')

        now = datetime.datetime.now()
        filename = f'{keyword}-{now.strftime("%Y-%m-%d-%H-%M-%S")}.csv'

        # Run the tweet harvesting code
        limit = 50
        command = f'npx --yes tweet-harvest@2.2.8 -o "{filename}" -s "{search_keyword}" -l {limit} --token {twitter_auth_token}'
        os.system(command)

        file_path = f'tweets-data/{filename}'

        # Read the CSV file into a Pandas DataFrame
        df = pd.read_csv(file_path, delimiter=';')

        selected_columns = df[['created_at', 'full_text']]

        preprocessed_texts = []

        # preprocessing
        for _, row in selected_columns.iterrows():
            df.fillna('', inplace=True)
            cleaned = cleaning_text(row['full_text'])
            lowercased = case_folding(cleaned)
            tokenized = tokenizing(lowercased)
            normalized = normalisasi(tokenized)
            stopword_removed = stopword_removal(normalized)
            stemmed = stemming(stopword_removed)
            preprocessed_text = (" ").join(stemmed)
            preprocessed_texts.append(preprocessed_text)

        selected_columns['preprocessed_text'] = preprocessed_texts


        # load model
        with open('tfidf_vectorizer.pkl', 'rb') as tfidf_vectorizer_file:
            loaded_tfidf_vectorizer = pickle.load(tfidf_vectorizer_file)

        with open('svm_linear_model.pkl', 'rb') as model_file:
            loaded_linear_model = pickle.load(model_file)

        corpus = [item for item in preprocessed_texts]
        # predict
        hasil_linear = loaded_linear_model.predict(loaded_tfidf_vectorizer.transform(corpus))

        # add column
        selected_columns['hasil'] = hasil_linear

        # truncate table
        session.execute(text('TRUNCATE TABLE scraping_tweet'))

        # save data to database
        for _, row in selected_columns.iterrows():
            hasil = "Kebencian" if row['hasil'] == -1 else 'Non-Kebencian'
            data = ScrapingModel(teks=row['full_text'], hasil_klasifikasi=hasil, created_at=row['created_at'], preprocessing=row['preprocessed_text'], keyword=keyword)
            session.add(data)

        session.commit()

        return redirect(url_for('scrap_tweet'))

    data = ScrapingModel.query.all()
    keyword = ScrapingModel.query.first()
    countKebencian = ScrapingModel.query.filter_by(hasil_klasifikasi='Kebencian').count()
    countNonKebencian = ScrapingModel.query.filter_by(hasil_klasifikasi='Non-Kebencian').count()
    return render_template('scrapTweet.html', data=data, countKebencian=countKebencian, countNonKebencian=countNonKebencian, keyword=keyword)


@app.errorhandler(jinja2.exceptions.TemplateNotFound)
def template_not_found(e):
    return not_found(e)


@app.errorhandler(404)
def not_found(e):
    return render_template('404.html')


if __name__ == '__main__':
    app.run(debug=True)
