import numpy as np
import cv2 as cv
import math


nClusters = 10
clusters = []
nFeatures = 3
centroids = centroids = np.zeros((nClusters, nFeatures))

np.random.seed(444)

def HSV(img):
  hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
  hsv[...,2] = 255
  cv.imshow("Imagen HSV", hsv)
  return hsv

def RGB(hsv):
  img2 = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)
  cv.imshow("Imagen RGB", img2)
  return img2

def Escgris(img):
  gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
  ret, thresh = cv.threshold(gray, 127, 255, 0)
  cv.imshow("Imagen umbralizada",thresh)
  return thresh

def euclidean_distance(p1, p2):
    n = p1.shape[0]#numero de filas 
    sum_squares = 0
    for i in range(n):
        sum_squares += (p1[i] - p2[i]) ** 2
    return math.sqrt(sum_squares)


def contornos(thresh):
  
  contours, _ = cv.findContours(thresh, cv.RETR_TREE,
                               cv.CHAIN_APPROX_SIMPLE)
  
  
  for i in range(0,len(contours)):
        print("Contorno;",i)
        print("Cantidad de contornos:", len(contours[i]))
        
  cnt1 = contours[13]
  x, y, w, h = cv.boundingRect(cnt1)

  cnt2 = contours[40]
  x2, y2, w2, h2 = cv.boundingRect(cnt2)

  cnt3 = contours[0]
  x3, y3, w3, h3 = cv.boundingRect(cnt3)

  cnt6 = contours[0]
  x4, y4, w4, h4 = cv.boundingRect(cnt6)
  
  return cnt1, cnt2, cnt3, cnt6



def Recta1(cnt1, no_objeto, img,img3):
  cv.drawContours(img3, [cnt1], -1, (0,255,0), 3)
  cv.imshow("Contorno del objeto 1",img3)
  
  x, y, w, h = cv.boundingRect(cnt1)
  img3 = cv.rectangle(img3, (x, y), (x+w, y+h), (0, 255, 0), 2)
  cv.imshow("Cuadro del objeto 1",img3)

  rect1 = cv.minAreaRect(cnt1)
  box1 = cv.boxPoints(rect1)
  box1 = np.int0(box1)

  img3 = cv.drawContours(img3, [box1], 0, (0, 0, 255), 2)
  cv.imshow("Cuadro ajustado al ancho del objeto",img3)

  color = (0, 255, 0)
  thickness = 2

  print("\n")

  dist_x_1 = (box1[0][0] + box1[3][0])/2
  dist_x_1 = int(dist_x_1)
  print("Coordenada X1 del objeto 1:",dist_x_1)

  dist_y_1 = (box1[0][1] + box1[3][1])/2
  dist_y_1 = int(dist_y_1)
  print("Coordenada Y1 del objeto 1:",dist_y_1)

  dist_x_2 = (box1[1][0] + box1[2][0])/2
  dist_x_2 = int(dist_x_2)
  print("Coordenada X2 del objeto 1:",dist_x_2)

  dist_y_2 = (box1[1][1] + box1[2][1])/2
  dist_y_2 = int(dist_y_2)
  print("Coordenada Y2 del objeto 1:",dist_y_2)

  start_point1 = (dist_x_1, dist_y_1)
  print(start_point1)
  end_point1 = (dist_x_2, dist_y_2)
  print(end_point1)
  image = cv.line(img, start_point1, end_point1, color, thickness)
  distance1 = math.sqrt((dist_x_2 - dist_x_1)**2+(dist_y_2 - dist_x_1)**2)
  print("Distancia 1: ",distance1)
  print("\n")
  return

def Recta2(cnt1, no_objeto, img,img3):
      
  cv.drawContours(img3, [cnt1], -1, (0,255,0), 3)
  cv.imshow("Contornos del objeto 2",img3)
  
  x, y, w, h = cv.boundingRect(cnt1)
  img3 = cv.rectangle(img3, (x, y), (x+w, y+h), (0, 255, 0), 2)
  cv.imshow("Cuadro del objeto 2",img3)

  rect1 = cv.minAreaRect(cnt1)
  box1 = cv.boxPoints(rect1)
  box1 = np.int0(box1)

  img3 = cv.drawContours(img3, [box1], 0, (0, 0, 255), 2)
  cv.imshow("Cuadro ajustado al ancho del objeto 2",img3)

  color = (0, 255, 0)
  thickness = 2
  print("\n")

  dist_x_1 = (box1[1][0] + box1[0][0])/2
  dist_x_1 = int(dist_x_1)
  print("Coordenada X1 del objeto 2: ",dist_x_1)

  dist_y_1 = (box1[1][1] + box1[0][1])/2
  dist_y_1 = int(dist_y_1)
  print("Coordenada Y1 del objeto 2: ",dist_y_1)

  dist_x_2 = (box1[3][0] + box1[2][0])/2
  dist_x_2 = int(dist_x_2)
  print("Coordenada X2 del objeto 2: ",dist_x_2)

  dist_y_2 = (box1[3][1] + box1[2][1])/2
  dist_y_2 = int(dist_y_2)
  print("Coordenada Y2 del objeto 2: ",dist_y_2)

  start_point1 = (dist_x_1, dist_y_1)
  print(start_point1)
  end_point1 = (dist_x_2, dist_y_2)
  print(end_point1)
  image = cv.line(img, start_point1, end_point1, color, thickness)
  distance1 = math.sqrt((dist_x_2 - dist_x_1)**2+(dist_y_2 - dist_x_1)**2)
  print("Distancia 2: ",distance1)
  print("\n")
  return

def euclidean_distance(p1, p2):
    n = p1.shape[0]#numero de filas 
    sum_squares = 0
    for i in range(n):
        sum_squares += (p1[i] - p2[i]) ** 2
    return math.sqrt(sum_squares)

def predict(sample):
    #predecir la distancia mas 
    # cercana entre un cluster y un sample
    # otorga un sample a un cluster y retorna el cluster que esta mas cerca del sample
    most_closer_distance = float('inf')
    most_closer_nCluster = 0
    for nCluster in range(nClusters):
        #centroid es tomar uno de los centroides disponibles()
        centroid = centroids[nCluster] # se toma un centoride a la vez
        tmp_distance = euclidean_distance(centroid, sample)
        #si la distancia tmp es menor a la distancia mas cercana
        if tmp_distance < most_closer_distance:
            most_closer_distance = tmp_distance
            most_closer_nCluster = nCluster
    return most_closer_nCluster

def cluster_samples(dataset:np.ndarray):
    #otorgar un cluster a los samples
    clusters = []
    for nCluster in range(nClusters):
        clusters.insert(nCluster, [])#se guardan los cluster en un arreglo
    for sample in dataset:
        #va a determinar el mejor cluster(cercano)
        most_closer_nCluster = predict(sample)
        clusters[most_closer_nCluster].append(sample)

def recalculate_centroids():
    #se recalculan los centroides a apartir de un promedio de cada feature por cada cluster
    for nCluster in range(nClusters):
        #clusters append de samples
        cluster = np.array(clusters[nCluster])
        print(cluster)
        tmp_centroid = []
        if len(cluster) <= 0: #el cluster estaa vacio =0
            continue
        for nFeature in range(nFeatures):
            feature_array = cluster[:, nFeature]
            #promedio de feature_array
            tmp_centroid.append(np.average(feature_array))
        #se hacen las comparaciones hasta que los puntos sean origiales
        #np.floor truncar valores
        centroids[nCluster] = np.floor(tmp_centroid)

def assign_random_centroids(dataset:np.ndarray):
    #Otorga un centroide escogiendo uno de los n-cluster con samples random y tomandolo como nuevo centroide 
    nSamples, _ = dataset.shape
    global centroids
    centroids = np.zeros((nClusters, nFeatures))#inicializar en ceros
    for nCluster in range(nClusters):
        #generar el centroide en una de las cordenadas
        rnd = np.random.randint(0, nSamples)
        #se van a asignar diferentes centroides a partir de un numero random
        centroids[nCluster] = dataset[rnd]

def fit(dataset:np.ndarray):
    #Entrenar al modelo de k-means a partir de un dataset
    nSamples, nFeatures = dataset.shape #detectar numero de filas y columnas del dataset
    nFeatures = nFeatures # atributo
    i = 0
    assign_random_centroids(dataset)
    tmpCentroids = centroids.copy()
    #comparar los centroides y que sean pocas iteraciones
    while (not np.array_equal(tmpCentroids, centroids)) and i < 20:           
        tmpCentroids = centroids.copy()
        cluster_samples(dataset)
        recalculate_centroids()
        i += 1

img = cv.imread(r"C:\Users\georo\OneDrive\Escritorio\Practica 7\Jit1.jpg",cv.IMREAD_COLOR) 
#cv.imshow("Original",img)
w,h,c = img.shape
img_preprocessed = np.reshape(img,(w*h,c))#transformar una lista
img_processed = np.zeros((w,h,c), dtype=np.uint8)
nClusters = 10

fit(img_preprocessed)
for x in range(w):
    for y in range(h):
        sample = img[x][y]
        cluster_predicted = predict(sample)
        img_processed[x][y] = np.floor(centroids[cluster_predicted])
print(f"Centroides: {centroids}")

hsv = HSV(img_processed)
img2 = RGB(hsv)
gris = Escgris(img2)
cnt1, cnt2, cnt3, cnt4 = contornos(gris)
img3 = img.copy()

Recta1(cnt1, 4, img, img3)
Recta2(cnt2, 2, img, img3)
cv.imshow("Imagen Final",img)
cv.imshow("Clusters",img_processed)
cv.waitKey(0)
cv.destroyAllWindows()
  