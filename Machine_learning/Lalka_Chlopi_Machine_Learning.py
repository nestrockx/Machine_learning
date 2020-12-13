#created by Paweł Kraszewski

#ten projekt to model który dopasowuje zdanie do książki w której zostało ono napisane
#do stworzenia modelu wykorzystaliśmy wolnoźródłowe pliki dwóch książek: Lalka i Chłopi
#przy wykorzystaniu klasyfikatora LogisticRegression i próbce testowej 10%, predykcja osiagnęła precyzję na poziomie 90%

#||przykładowy output z wykonania programu||
#\/                           \/
'''
Start:

czas trenowania: 25.302 s
czas predykcji: 0.126 s
dokładność: 0.905

                    przewidziana:Lalka  przewidziana:Chlopi
oczekiwana:Lalka                 2442                   65
oczekiwana:Chlopi                 317                 1217
Press any key to continue . . .
'''

import re
import pandas as pd
from time import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.metrics import accuracy_score
from sklearn import metrics
import string

#przygotowanie tesktu, usuwanie liczb, znaków interpunkcyjnych
def preprocess(sentences):
    stringConcat = ""
    for x in sentences:
        stringConcat += x + "BREAK"
    stringConcat = stringConcat[:-1]
    f = open("stop_words.txt", "r", encoding='utf8')
    stop_words = f.read().split('\n')
    f.close()
    #print(stop_words)
    stringConcat = stringConcat.translate(str.maketrans('', '', string.punctuation))
    stringConcat = stringConcat.strip()
    stringConcat = re.sub(r'\d+', '', stringConcat)
    stringConcat = re.sub(r'—', '', stringConcat)
    stringConcat = re.sub(r'–', '', stringConcat)
    for x in stop_words:
        stringConcat = re.sub('\s' + x + '\s', ' ', stringConcat, flags = re.IGNORECASE)
    return stringConcat.split('BREAK')
    

#metoda klasyfikująca
def classifyData(clf, X_train, X_test, y_train, y_test):

    print("Start:\n")

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

    cmtx = pd.DataFrame(
    metrics.confusion_matrix(expected, pred, labels=['Lalka', 'Chlopi']), 
    index=['oczekiwana:Lalka', 'oczekiwana:Chlopi'], 
    columns=['przewidziana:Lalka', 'przewidziana:Chlopi']
    )

    #macierz pomyłek pokazująca ilość poprawnych i niepoprawnych klasyfikacji dla każdej z klas
    print("\n", cmtx)



dfSentences = pd.read_csv('LalkaChlopiZdania.csv')
dfBooks = pd.read_csv('KsiazkaLalkaChlopi.csv')

dfSentences = dfSentences['Zdanie'].astype(str).values.tolist()
dfBooks = dfBooks['Ksiazka'].astype(str).values.tolist()

dfSentences = preprocess(dfSentences)
#print(dfSentences)

features_train, features_test, labels_train, labels_test = train_test_split(dfSentences, dfBooks, test_size=0.1, random_state=24)


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
features_train_transformed = vectorizer.fit_transform(features_train)
features_test_transformed  = vectorizer.transform(features_test)


selector = SelectPercentile(f_classif, percentile=10)
selector.fit(features_train_transformed, labels_train)
features_train_transformed = selector.transform(features_train_transformed).toarray()
features_test_transformed  = selector.transform(features_test_transformed).toarray()

classifyData(LogisticRegression(), features_train_transformed, features_test_transformed, labels_train, labels_test)




