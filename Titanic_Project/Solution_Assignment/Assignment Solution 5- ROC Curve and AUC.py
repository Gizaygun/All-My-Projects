import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer

# Örnek veri kümesi yükleyelim (burada Breast Cancer veri kümesi kullanılacak)
data = load_breast_cancer()
X = data.data
y = data.target

# Veriyi eğitim ve test olarak bölelim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# DecisionTreeClassifier modelini oluşturuyoruz ve eğitiyoruz
best_clf = DecisionTreeClassifier(random_state=42)
best_clf.fit(X_train, y_train)

# Modelin pozitif sınıfı için tahmin olasılıklarını alın
y_pred_prob = best_clf.predict_proba(X_test)[:, 1]

# ROC eğrisini hesaplayın
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

# AUC değerini hesaplayın
roc_auc = auc(fpr, tpr)

# ROC eğrisini çizin
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # rastgele tahmin çizgisi
plt.xlim([0.0, 1.0])  # X ekseninin sınırları
plt.ylim([0.0, 1.05])  # Y ekseninin sınırları
plt.xlabel('False Positive Rate')  # X eksenine etiket
plt.ylabel('True Positive Rate')  # Y eksenine etiket
plt.title('Receiver Operating Characteristic')  # Başlık
plt.legend(loc='lower right')  # Ağırlıklı etiket
plt.show()  # Grafiği göster
