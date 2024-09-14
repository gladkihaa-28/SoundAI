from flask import Flask, request, jsonify
import numpy as np
import os
import librosa
import joblib

app = Flask(__name__)

# Загрузка модели при старте приложения
ensemble_model = joblib.load('ensemble_model.pkl')


# Функция для извлечения признаков MFCC из аудиофайла
def extract_mfcc(file_path, max_length=250):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)

    if len(mfccs_mean) > max_length:
        features = mfccs_mean[:max_length]
    else:
        features = np.pad(mfccs_mean, (0, max_length - len(mfccs_mean)), 'constant')

    return features


# API для классификации аудиофайла
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Сохраняем временный файл
    file_path = os.path.join('temp', file.filename)
    file.save(file_path)

    try:
        # Извлечение признаков
        features = extract_mfcc(file_path)
        features = np.expand_dims(features, axis=0)

        # Предсказание
        prediction = ensemble_model.predict(features)

        # Удаление временного файла
        os.remove(file_path)

        # Возвращаем предсказание в формате JSON
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        os.remove(file_path)
        return jsonify({'error': str(e)}), 500


# Запуск приложения
if __name__ == '__main__':
    if not os.path.exists('temp'):
        os.makedirs('temp')
    app.run(host='0.0.0.0', port=5000)
