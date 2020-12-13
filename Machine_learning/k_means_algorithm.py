#created by Paweł Kraszewski

from sklearn.decomposition import PCA
import scipy.io as sio
import numpy as np
from random import randrange
import matplotlib.pyplot as plt

#funkcja obliczająca odległość taksówkową
def calcManhattanDistance(x, y):
    return np.sum(np.abs(x-y))

#funkcja obliczająca odległość euklidesowską
def calcEuclideanDistance(x, y):
    return np.linalg.norm(x-y)

#funkcja przedstawiająca pogrupowane dane w postaci wykresu na płasczyźnie
#z pomocą redukcji wymiarowości PCA
def show(indexedPoints):
    points = np.empty((0, len(indexedPoints[0, 1])))
    
    for x in indexedPoints:
        points = np.append(points, [x[1]], axis=0)

    pca = PCA(2)
    points = pca.fit_transform(points)

    indexedPoints2D = np.empty((0, 2))
    for i in range(len(points)):
        indexedPoints2D = np.append(indexedPoints2D, np.array([[indexedPoints[i, 0], points[i, :]]]), axis=0)

    colors = ["#0000FF", "#00FF00", "#FF0066", "#001166"]
    for x in indexedPoints2D:
        plt.scatter(x[1][0], x[1][1], c=colors[int(x[0])])

    plt.show()


#funkcja implementująca algorytm k-środków
#i zwrócenie pogrupowanych danych
def clusterize(Patterns, k):
    kCentroids = []
    kCentroidsTmp = []
    indexedPoints = np.empty((0, 2))

    for i in range(k):
        kCentroidsTmp.append(Patterns[randrange(len(Patterns))])

    while not np.array_equal(kCentroids, kCentroidsTmp):
        kCentroids = kCentroidsTmp.copy()
        indexedPoints = np.empty((0, 2))
        for point in Patterns:
            distances = np.empty((0, 2))
            i = 0
            for kCenter in kCentroidsTmp:
                currDistance = calcManhattanDistance(kCenter, point)
                distances = np.append(distances, np.array([[i, currDistance]]), axis = 0)
                i += 1
            distances = distances[np.argsort(distances[:, 1])]
            indexedPoints = np.append(indexedPoints, np.array([[distances[0, 0], point]]), axis = 0)
        for i in range(k):
            sumPoints = np.empty((0, len(point)))
            for j in range(len(indexedPoints)):
                if indexedPoints[j, 0] == i:
                    sumPoints = np.append(sumPoints, np.array([indexedPoints[j, 1]]), axis = 0)
            kCentroidsTmp[i] = np.mean(sumPoints, axis = 0)
        
    return indexedPoints


#załadowanie danych
dataMat = sio.loadmat("data")
Patterns = np.array(dataMat.get('Patterns'))
Patterns2 = np.array(dataMat.get('Patterns2'))


k = 4

indexedPoints = clusterize(Patterns, k)
show(indexedPoints)

indexedPoints = clusterize(Patterns2, k)
show(indexedPoints)

