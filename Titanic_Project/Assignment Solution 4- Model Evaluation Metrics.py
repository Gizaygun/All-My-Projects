from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Örnek veri seti (burada iris veri setini kullanıyoruz, sizin verinize uygun şekilde değiştirebilirsiniz)
from sklearn.datasets import load_iris
import numpy as np

# Veri setini yükleme
data = load_iris()
X = data.data
y = data.target

# Veriyi eğitim ve test setlerine ayırma (80% eğitim, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Karar ağacı modeli oluşturma ve eğitme
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Test seti ile tahmin yapma
y_pred_best = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_best)
print("Confusion Matrix:")
print(cm)

# Classification report
report = classification_report(y_test, y_pred_best)
print("\nClassification Report:")
print(report)

# Ekstra olarak precision, recall, f1-score hesaplamaları
from sklearn.metrics import precision_score, recall_score, f1_score

# Precision, recall, ve F1-score hesaplamaları
precision = precision_score(y_test, y_pred_best, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred_best, average='weighted', zero_division=1)
f1 = f1_score(y_test, y_pred_best, average='weighted', zero_division=1)

print("\nPrecision (weighted):", precision)
print("Recall (weighted):", recall)
print("F1-score (weighted):", f1)
