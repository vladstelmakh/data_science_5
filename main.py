import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def read_data(folder):
    data = pd.DataFrame()
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            file_path = os.path.join(folder, file)
            df = pd.read_csv(file_path, header=None, names=['accelerometer_X', 'accelerometer_Y'], skiprows=1)
            df['activity'] = folder.split('/')[-1]  # Додаємо мітку для класифікації
            data = pd.concat([data, df], ignore_index=True)
    return data


idle_data = read_data('data/idle')
running_data = read_data('data/running')
stairs_data = read_data('data/stairs')
walking_data = read_data('data/walking')

all_data = pd.concat([idle_data, running_data, stairs_data, walking_data], ignore_index=True)


X_train, X_test, y_train, y_test = train_test_split(all_data[['accelerometer_X', 'accelerometer_Y']], all_data['activity'], test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Навчання SVM моделі
svm_model = SVC(kernel='linear', C=1)
svm_model.fit(X_train_scaled, y_train)

# Навчання моделі випадкового лісу
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)


svm_predictions = svm_model.predict(X_test_scaled)
rf_predictions = rf_model.predict(X_test_scaled)

# Оцінка точності моделей
svm_accuracy = accuracy_score(y_test, svm_predictions)
rf_accuracy = accuracy_score(y_test, rf_predictions)


print(f'SVM Accuracy: {svm_accuracy}')
print(f'Random Forest Accuracy: {rf_accuracy}')

import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(confusion_matrix(y_test, svm_predictions), annot=True, fmt='d', cmap='Blues')
plt.title('SVM Confusion Matrix')
plt.show()


sns.heatmap(confusion_matrix(y_test, rf_predictions), annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.show()