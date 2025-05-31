from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Veri setini yükleyelim
data = load_iris()
X = data.data
y = data.target

# Eğitim ve test verisi olarak ayıralım
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Hiperparametreler için grid belirleyelim
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10]
}

# GridSearchCV nesnesini başlatalım
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)

# Modeli eğitelim
grid_search.fit(X_train, y_train)

# En iyi parametreleri yazdıralım
print(f"Best parameters: {grid_search.best_params_}")

# En iyi parametrelerle modeli eğitelim
best_clf = grid_search.best_estimator_

# Tahmin yapalım ve doğruluğu değerlendirelim
y_pred_best = best_clf.predict(X_test)
best_accuracy = accuracy_score(y_test, y_pred_best)
print(f"Best accuracy: {best_accuracy:.2f}")
