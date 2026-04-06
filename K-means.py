import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from copy import deepcopy

iris = datasets.load_iris()        #Imports data

samples = iris.data                #Saves data in local variable 

x = samples[:,0]                   #Seperates the first column
y = samples[:,1]                   #Seperates the second column

sepal_length_width = np.array(list(zip(x, y)))                 #Combines two columns into an array of coordinates (x, y)

centroids_x = np.random.uniform(min(x), max(x), size=3)
centroids_y = np.random.uniform(min(y), max(y), size=3)        #defines a random x and y coordinate for the first centriod

centroids = np.array(list(zip(centroids_x, centroids_y)))      #Combines two x and y coordinates into one(x, y) coordinate

def distance(a, b):                                            
  one = (a[0] - b[0]) ** 2
  two = (a[1] - b[1]) ** 2
  return (one + two) ** 0.5                             #Euclidian Distance equation

centroids_old = np.zeros(centroids.shape)               #Creates an empty array of centriods

labels = np.zeros(len(samples))                         #Creates an empty array of group labels for each point

distances = np.zeros(3)                                 #Creates an empty array of a points distance to each centroid

error = np.zeros(3)                                     #Creates an empty array of errors (distance between old and new centroid)

error[0] = distance(centroids[0], centroids_old[0])     #Defines the initial error for each centroid
error[1] = distance(centroids[1], centroids_old[1])
error[2] = distance(centroids[2], centroids_old[2])     

while error.all() != 0:                                             #Loops until the old and new centroids are equal for all three centroids

  for i in range(len(samples)):                                     #Loops through each coordinate in the iris dataset
    distances[0] = distance(sepal_length_width[i], centroids[0])
    distances[1] = distance(sepal_length_width[i], centroids[1])
    distances[2] = distance(sepal_length_width[i], centroids[2])    #Calculates the distance between the coordinate and all three centroids
    cluster = np.argmin(distances)                                  #Finds the smallest of the three distances
    labels[i] = cluster                                             #Saves the label (closest centroid) of the point

  centroids_old = deepcopy(centroids)                               #Saves centroids before recalculation

  for i in range(3):                                                                                #Loops through each centroid
    points = [sepal_length_width[j] for j in range(len(sepal_length_width)) if labels[j] == i]      #Creates an array of all the points in a group
    centroids[i] = np.mean(points, axis=0)                                                          #Identifies the new centroid of the group as the average of all the points in the group

  error[0] = distance(centroids[0], centroids_old[0])                                               #Defines the initial error for each centroid
  error[1] = distance(centroids[1], centroids_old[1])
  error[2] = distance(centroids[2], centroids_old[2]) 

colors = ['r', 'g', 'b']                                                                        #Colors for each group on the graph
for i in range(3):                                                                              #Loops through each loop
  points = np.array([sepal_length_width[j] for j in range(len(samples)) if labels[j] == i])     #Creates an array of all the points in a group
  plt.scatter(points[:, 0], points[:, 1], c=colors[i], alpha=0.5)                               #Plots each point with it's group's color
plt.scatter(centroids[:, 0], centroids[:, 1], marker='D', s=150)                                #Plots the centroids

plt.xlabel('sepal length (cm)')          #X-axis label
plt.ylabel('sepal width (cm)')           #Y-axis label

plt.show()                               #Returns the graph
