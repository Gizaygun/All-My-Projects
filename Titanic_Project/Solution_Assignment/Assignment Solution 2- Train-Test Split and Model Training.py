import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Titanic datasetini yükle
data = pd.read_csv(r"C:\Users\u23p51\Desktop\makine öğrenmesi\Solution_Assignment\titanic.csv")

# İlk birkaç satırı kontrol et
print(data.head())

# Verilerde eksik değerleri kontrol et
print(data.isnull().sum())

# 'Age' kolonundaki eksik değerleri doldur
data['Age'] = data['Age'].fillna(data['Age'].mean())

# Kategorik verileri sayısal verilere dönüştür
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# 'Embarked' gibi diğer kategorik veriler de sayısal hale getirilebilir (optional)
# data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Kullanılacak özellikler (features) ve hedef değişken (target)
# 'Name', 'Ticket', 'Cabin' gibi metin verilerini dışarıda bırak
X = data.drop(['Survived', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)  # 'Survived' dışındaki tüm kolonlar özellikler
y = data['Survived']  # 'Survived' hedef değişken

# Veriyi eğitim ve test setlerine ayırma (80% eğitim, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Karar ağacı sınıflandırıcısını eğitme
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Test setinde tahminler yapma
y_pred = clf.predict(X_test)

# Doğruluk değerlendirmesi
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
