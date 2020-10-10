# SINGLE VARIABLE LINEAR REGRESSION

# Imports
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Getting the data and storing it in lists
import os
path = os.getcwd() + '/data.txt'

x = []
y = []
file = open(path,"r")
while True:
    line = file.readline()
    if not line:
        break
    else:
        values = line.split(",")
        x.append(float(values[0]))
        y.append(float(values[1].rstrip("\n")))
file.close()

# Plotting the data
plt.plot(x, y, 'o')
plt.show()

# Converting the data from a list to a column matrix
x = np.matrix(x).T
y = np.matrix(y).T

# Starting the linear regression
lm = LinearRegression()

# Find the line of best fit and theta values
lm.fit(x, y)
theta0 = lm.intercept_[0]
theta1 = lm.coef_[0][0]
print("The line of best fit is:")
print("y = " + str(theta0) + " + " + str(theta1) + "x")

# Plotting the line of best fit
lbfY = []
lbfX = []
xMin = int(np.min(x))
xMax = int(np.max(x))
for i in range(xMin, xMax+1):
    lbfY.append(theta0 + theta1 * i)
    lbfX.append(i)
plt.plot(x, y, 'o')
plt.plot(lbfX, lbfY)
plt.show()

# Making predictions using the model

# Find the y value when x is 35
p1 = lm.predict([[35]])
print("Prediction 1:", p1)

#Find the y value when x is 90
p2 = lm.predict([[90]])
print("Prediction 2:", p2)