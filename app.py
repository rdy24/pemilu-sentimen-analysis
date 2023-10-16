#!/usr/bin/env python
from flask import Flask, url_for, render_template, request, jsonify, redirect
import jinja2.exceptions
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from training_models_module import TrainingModel
from preprocessing import cleaning_text, case_folding, tokenizing, stopword_removal, stemming


app = Flask(__name__)

# Database connection
db_user = 'root'
db_password = ''
db_host = 'localhost'
db_name = 'pemilu-sentimen-analysis'

# Membuat koneksi ke database MySQL
db_url = f'mysql://{db_user}:{db_password}@{db_host}/{db_name}'
engine = create_engine(db_url)

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


@app.route('/preprocessing')
def preprocessing():
    data = session.query(TrainingModel).all()

    # Melakukan preprocessing pada data
    preprocessed_data = []
    for item in data:
        cleaned_text = cleaning_text(item.teks)
        lowercased_text = case_folding(cleaned_text)
        tokenized_text = tokenizing(lowercased_text)
        stopwords_removed_text = stopword_removal(tokenized_text)
        stemmed_text = stemming(stopwords_removed_text)

        preprocessed_data.append({
            'id': item.id,
            'teks': (" ").join(stemmed_text),  # Menggabungkan kembali kata-kata yang telah dipreproses
            'label': item.label,
            'sosmed': item.sosmed
        })

    return jsonify(preprocessed_data)

    # return render_template('preprocessing.html', data=preprocessed_data)


@app.errorhandler(jinja2.exceptions.TemplateNotFound)
def template_not_found(e):
    return not_found(e)

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html')

if __name__ == '__main__':
    app.run(debug=True)
