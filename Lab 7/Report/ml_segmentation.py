from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
import cv2
from collections import Counter
from sklearn.cluster import KMeans
plt.rcParams['image.interpolation'] = 'nearest'
plt.style.use('ggplot')

# read the image
img = cv2.imread('Lab6/city.jpg')
img_hab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
# Getting the values and plotting it
channel_a = img_hab[:, :, 1]
channel_b = img_hab[:, :, 2]
f1 = channel_a.flatten()
f2 = channel_b.flatten()
X_test = np.array(list(zip(f1, f2)))

# get train set X
X = list(zip(f1, f2))
X_tuple = []
for i in range(len(X)):
    X_tuple.append(tuple(X[i]))
X_count = Counter(X_tuple)
print(X_count)
X = list(set(X_tuple))
f1 = []
f2 = []
for i in range(len(X)):
    f1.append(X[i][0])
    f2.append(X[i][1])
plt.scatter(f1, f2, c='black', s=7)


# Euclidean Distance Caculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


# Number of clusters
k = 3
# X coordinates of random centroids
C_x = np.random.randint(90, 200, size=k)
# Y coordinates of random centroids
C_y = np.random.randint(60, 210, size=k)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
print("Initial Centroids")
print(C)

# Plotting along with the Centroids
plt.scatter(f1, f2, c='#050505', s=7)
plt.scatter(C_x, C_y, marker='*', s=200, c='g')
plt.show()

X = np.array(list(zip(f1, f2)))
# To store the value of centroids when it updates
C_old = np.zeros(C.shape)
# Cluster Lables(0, 1, 2, 3, 4, 5)
clusters = np.zeros(len(X))
# Error func. - Distance between new centroids and old centroids
error = dist(C, C_old, None)
# Loop will run till the error becomes zero
while error < 0.01:
    # Assigning each value to its closest cluster
    for i in range(len(X)):
        distances = dist(X[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    # Storing the old centroid values
    C_old = deepcopy(C)
    # Finding the new centroids by taking the average value
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    error = dist(C, C_old, None)


clusters = np.zeros(len(X_test))
for i in range(len(X_test)):
        distances = dist(X_test[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster

kmeans = KMeans(n_clusters=6)
kmeans.fit(X)
y_kmeans = kmeans.predict(X_test)

dst = np.copy(img)
clusters = y_kmeans
clusters = clusters.reshape(img.shape[:2])
colors = [(189, 107, 42), (91, 243, 208), (87, 154, 193), (195, 128, 67), (246, 118, 6), (32, 143, 251)]
for r in range(img.shape[0]):
    for c in range(img.shape[1]):
        dst[r, c] = colors[int(clusters[r, c])]
plt.figure(3)
plt.title("segmented image")
plt.imshow(dst)
