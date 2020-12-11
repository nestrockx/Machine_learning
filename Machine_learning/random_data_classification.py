#created by Paweł Kraszewski

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from time import time
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#metoda klasyfikująca
def classifyData(clf, X_train, X_test, y_train, y_test):

    t0 = time()
    #trening danych
    clf.fit(X_train, y_train)
    #wyznaczenie czasu treningu
    print ("czas trenowania:", round(time() - t0, 3), "s")

    t1 = time()
    #predykcja
    pred = clf.predict(X_test)
    #wyznaczenie czasu predykcji
    print ("czas predykcji:", round(time() - t1, 3), "s")

    #oczekiwane wartości
    expected = y_test

    #określenie precyzji predykcji
    print("dokładność: ", end="")
    print (round(accuracy_score(pred, expected), 3))

    #macierz pomyłek pokazująca ilość poprawnych i niepoprawnych klasyfikacji dla każdej z klas
    print(metrics.confusion_matrix(expected, pred))

#wczytanie losowych danych numerycznych do klasyfikacji z podziałem na 3 klasy
#X - atrybuty przewidujące
#y - atrybut przewidywany
X, y = datasets.make_classification(n_samples = 1000, n_classes = 3, n_informative = 2, n_clusters_per_class = 1)

#dzielenie danych na treningowe i testowane za pomocą kroswalidacji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

#4 rożne modele klasyfikujące wykonane na tych samych danych z tą samą próbką treningową
print("Naiwny klasyfikator Bayesa")
classifyData(GaussianNB(), X_train, X_test, y_train, y_test)
print("\nDrzewo decyzyjne")
classifyData(DecisionTreeClassifier(), X_train, X_test, y_train, y_test)
print("\nSVC")
classifyData(SVC(), X_train, X_test, y_train, y_test)
print("\nAlgorytm k-srodkow")
classifyData(KNeighborsClassifier(), X_train, X_test, y_train, y_test)
