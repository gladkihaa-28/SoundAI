import os
import numpy as np
import pandas as pd
import librosa
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, f1_score
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib


def extract_mfcc(file_path, max_length=250):
    y, sr = librosa.load(file_path, sr=None)

    # Извлечение MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Усреднение MFCC по временным окнам
    mfccs_mean = np.mean(mfccs, axis=1)

    # Обрезаем или дополняем нулями до фиксированной длины
    if len(mfccs_mean) > max_length:
        features = mfccs_mean[:max_length]
    else:
        features = np.pad(mfccs_mean, (0, max_length - len(mfccs_mean)), 'constant')

    return features


def load_data(csv_file, audio_dir, max_length=250):
    data = pd.read_csv(csv_file)
    features, labels = [], []

    for i, row in data.iterrows():
        print(i)
        file_path = os.path.join(audio_dir, row[0])
        feature = extract_mfcc(file_path, max_length=max_length)
        features.append(feature)
        labels.append(row[1])

    return np.array(features), np.array(labels)


X, y = load_data('train_gt.csv', 'train/', max_length=250)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid_rf = {
    'classifier__n_estimators': [100, 150, 200, 250, 300, 350],
    'classifier__max_depth': [5, 10, 15, 20]
}

param_grid_gb = {
    'classifier__n_estimators': [100, 150, 200, 250, 300, 350],
    'classifier__max_depth': [3, 5, 6]
}

param_grid_xgb = {
    'classifier__n_estimators': [100, 150, 200, 250, 300, 350],
    'classifier__learning_rate': [0.01, 0.1, 1],
    'classifier__max_depth': [3, 5, 6]
}

pipeline_rf = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

pipeline_gb = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

pipeline_xgb = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', XGBClassifier(random_state=42))
])

grid_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=3, scoring='f1', n_jobs=-1, verbose=2)
grid_gb = GridSearchCV(pipeline_gb, param_grid_gb, cv=3, scoring='f1', n_jobs=-1, verbose=2)
grid_xgb = GridSearchCV(pipeline_xgb, param_grid_xgb, cv=3, scoring='f1', n_jobs=-1, verbose=2)

grid_rf.fit(X_train, y_train)
grid_gb.fit(X_train, y_train)
grid_xgb.fit(X_train, y_train)

best_rf = grid_rf.best_estimator_
best_gb = grid_gb.best_estimator_
best_xgb = grid_xgb.best_estimator_

print(f'Best RF Parameters: {grid_rf.best_params_}')
print(f'Best GB Parameters: {grid_gb.best_params_}')
print(f'Best XGB Parameters: {grid_xgb.best_params_}')

ensemble = VotingClassifier(
    estimators=[
        ('rf', best_rf),
        ('gb', best_gb),
        ('xgb', best_xgb)
    ],
    voting='soft'
)

ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)
f1 = f1_score(y_test, y_pred)

print(f'Ensemble F1 Score: {f1}')
print(classification_report(y_test, y_pred))

joblib.dump(ensemble, 'ensemble_model.pkl')
print("Модель ансамбля успешно сохранена.")

for name, model in ensemble.named_estimators_.items():
    joblib.dump(model, f'{name}_model.pkl')
    print(f"Модель {name} успешно сохранена.")


def save_predictions(test_dir, output_csv, max_length=250):
    # Создаем список файлов из папки test
    test_files = [f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]
    predictions = []

    for i, file_name in enumerate(test_files):
        file_path = os.path.join(test_dir, file_name)
        feature = extract_mfcc(file_path, max_length=max_length)
        feature = np.expand_dims(feature, axis=0)
        prediction = ensemble.predict(feature)
        print(f"{i} --> {prediction}")
        predictions.append([file_name, int(prediction[0])])

    # Сохранение предсказаний в CSV
    pd.DataFrame(predictions, columns=['filename', 'prediction']).to_csv(output_csv, index=False)


save_predictions('final_test/', 'final_test_predictions6.csv')
