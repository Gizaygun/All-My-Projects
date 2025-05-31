#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
# Veri setini yükleme
data_path = r'C:\Users\Gizem\Desktop\Veri_Seti_ChatGPT_İnsan.xlsx'
data = pd.read_excel(data_path)
# Veri setinin ilk beş satırını göster
print(data.head())
# Veri setinin sütun başlıklarını ve veri türlerini göster
print(data.info())


# In[31]:


import pandas as pd

# Veri setini yükleme
data_path = r'C:\Users\Gizem\Desktop\Veri_Seti_ChatGPT_İnsan.xlsx'
data = pd.read_excel(data_path)

# Veri setinin sütun başlıklarını ve veri türlerini göster
print(data.columns)


# In[12]:


import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# NLTK kütüphanesinin durak kelimeler ve WordNet lemmatizer'ını indir
nltk.download('stopwords')
nltk.download('wordnet')

# Veri setini yükleme
data_path = r'C:\Users\Gizem\Desktop\Veri_Seti_ChatGPT_İnsan.xlsx'
data = pd.read_excel(data_path)

# Durak kelimeler listesi
stop_words = set(stopwords.words('english'))

# Lemmatizer nesnesi
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Küçük harfe çevirme
    text = text.lower()
    # Noktalama işaretlerini ve sayıları kaldır
    text = re.sub(r'[\d\W]+', ' ', text)
    # Tokenize etme ve durak kelimeleri kaldırma
    words = text.split()
    words = [word for word in words if word not in stop_words]
    # Kelimeleri kökten ayırma
    words = [lemmatizer.lemmatize(word) for word in words]
    # Temizlenmiş metni geri döndür
    return ' '.join(words)

# Metin sütununu önişle
data['Processed_Text'] = data['Veri Seti'].apply(preprocess_text)

# İlk beş işlenmiş metni göster
print(data['Processed_Text'].head())

from sklearn.model_selection import train_test_split, KFold

# Önişleme adımı (metin önişlemeyi önceki adımda belirtilen gibi varsayıyorum)
data['Processed_Text'] = data['Veri Seti'].apply(preprocess_text) # Metin sütununun adını düzeltin

# Hedef sütun (ikinci sütun varsayılarak)
target = data.iloc[:, 1] # İkinci sütunu hedef olarak kullan

# Bağımsız değişkenler (özellikler)
features = data['Processed_Text']

# Veri setini eğitim ve test kümelerine ayırma
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# 10-katlı çapraz doğrulama için KFold nesnesi oluşturma
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# KFold ile çapraz doğrulama indexlerini yazdırma
for train_index, test_index in kf.split(X_train):
    print("TRAIN:", train_index, "TEST:", test_index)

# Eğitim ve test kümelerinin boyutlarını yazdır
print(f"Eğitim kümesi boyutu: {X_train.shape[0]}, Test kümesi boyutu: {X_test.shape[0]}")


# In[28]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Veri setini yükleme
data_path = r'C:\Users\Gizem\Desktop\Veri_Seti_ChatGPT_İnsan.xlsx'
data = pd.read_excel(data_path)

# Veri setinin ilk beş satırını göster
print(data.head())

# Veri setinin sütun başlıklarını ve veri türlerini göster
print(data.info())

# Bağımlı ve bağımsız değişkenleri ayırma
X = data['Gereksinimler']  # Bağımsız değişken
y = data['Yazar']  # Bağımlı değişken

# 'Gereksinim Tipi' sütununu da X'e ekleme
X = data['Gereksinimler'] + ' ' + data['Gereksinim Tipi']

# Metin verisini sayısal hale getirme (TF-IDF)
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# 'Yazar' sütununu sayısal değerlere dönüştürme
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Eğitim ve test veri setlerini ayırma
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_encoded, test_size=0.2, random_state=42)

# SVM modelini tanımlama ve eğitme
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Modelin performansını değerlendirme
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Modelin doğruluk değeri:", accuracy)


# In[32]:


import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np

# NLTK kütüphanesinin durak kelimeler ve WordNet lemmatizer'ını indir
nltk.download('stopwords')
nltk.download('wordnet')

# Veri setini yükleme
data_path = r'C:\Users\Gizem\Desktop\Veri_Seti_ChatGPT_İnsan.xlsx'
data = pd.read_excel(data_path)

# Veri setinin ilk beş satırını göster
print(data.head())

# Veri setinin sütun başlıklarını ve veri türlerini göster
print(data.info())

# Durak kelimeler listesi
stop_words = set(stopwords.words('english'))

# Lemmatizer nesnesi
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Küçük harfe çevirme
    text = text.lower()
    # Noktalama işaretlerini ve sayıları kaldır
    text = re.sub(r'[\d\W]+', ' ', text)
    # Tokenize etme ve durak kelimeleri kaldırma
    words = text.split()
    words = [word for word in words if word not in stop_words]
    # Kelimeleri kökten ayırma
    words = [lemmatizer.lemmatize(word) for word in words]
    # Temizlenmiş metni geri döndür
    return ' '.join(words)

# Metin sütununu önişle
data['Processed_Text'] = data['Gereksinimler'].apply(preprocess_text)

# Bağımlı ve bağımsız değişkenleri ayırma
X = data['Gereksinim Tipi']
y = data['Yazar']

# TF-IDF öznitelik çıkarımı
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Count Vectorizer öznitelik çıkarımı
count_vectorizer = CountVectorizer()
X_count = count_vectorizer.fit_transform(X)

# Word2Vec öznitelik çıkarımı
sentences = [text.split() for text in X]
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
word2vec_vectors = np.array([np.mean([word2vec_model.wv[word] for word in words if word in word2vec_model.wv] or [np.zeros(100)], axis=0) for words in sentences])

# Doc2Vec öznitelik çıkarımı
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(sentences)]
doc2vec_model = Doc2Vec(documents, vector_size=100, window=5, min_count=1, epochs=20)
doc2vec_vectors = np.array([doc2vec_model.dv[i] for i in range(len(documents))])

# One-Hot Encoding
onehot_vectorizer = CountVectorizer(binary=True)
X_onehot = onehot_vectorizer.fit_transform(X)

# 'Yazar' sütununu sayısal değerlere dönüştürme
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Öznitelik setlerini bir sözlükte toplama
feature_sets = {
    "TF-IDF": X_tfidf,
    "CountVector": X_count,
    "Word2Vec": word2vec_vectors,
    "Doc2Vec": doc2vec_vectors,
    "One-Hot": X_onehot
}

# Modelleri tanımlama
models = {
    "Lojistik Regresyon": LogisticRegression(max_iter=1000),
    "Karar Ağacı": DecisionTreeClassifier(),
    "Rastgele Orman": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()
}

# Her öznitelik seti ve model için eğitim, test ve performans değerlendirmesi yapma
results = {}
for feature_name, features in feature_sets.items():
    X_train, X_test, y_train, y_test = train_test_split(features, y_encoded, test_size=0.2, random_state=42)
    results[feature_name] = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        results[feature_name][model_name] = {
            "Precision": report['weighted avg']['precision'],
            "Recall": report['weighted avg']['recall'],
            "F1-Score": report['weighted avg']['f1-score'],
            "Confusion Matrix": cm
        }
        print(f"{feature_name} - {model_name} ile Performans:")
        print(f"Precision: {report['weighted avg']['precision']:.2f}")
        print(f"Recall: {report['weighted avg']['recall']:.2f}")
        print(f"F1-Score: {report['weighted avg']['f1-score']:.2f}")
        print(f"Confusion Matrix:\n{cm}\n")

# Sonuçları topluca yazdırma (isteğe bağlı)
print("Tüm Sonuçlar:")
print(results)


# In[ ]:




