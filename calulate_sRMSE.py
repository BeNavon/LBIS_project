import math
import numpy as np

def calculate_delta_phi(y_pred,y_actual):
    y1=y_actual[0]
    y2=y_actual[1]
    y1_hat = y_pred[0]
    y2_hat = y_pred[1]
    delta_phi = math.atan2( y2_hat*y1 - y1_hat*y2 , y1_hat*y1 + y2_hat*y2 )
    delta_phi = delta_phi / (2 * np.pi)
    return delta_phi

def calculate_delta_phi_vector(y1,y2,y1_hat,y2_hat):
    # Ensure inputs are NumPy arrays (in case they are lists)
    y1, y2, y1_hat, y2_hat = map(np.asarray, [y1, y2, y1_hat, y2_hat])
    # Compute angles using element-wise operations
    delta_phi = np.arctan2(y2_hat * y1 - y1_hat * y2, y1_hat * y1 + y2_hat * y2)
    delta_phi = delta_phi / (2 * np.pi)
    return delta_phi  # Returns a NumPy array

def calculate_sRMSE(y_pred,y_actual):
    # this code uses the assumption that y_pred and y_actual are in the dimensions: [num_examples,2]
    y1 = y_actual[:,0]
    y2 = y_actual[:,1]
    y1_hat = y_pred[:,0]
    y2_hat = y_pred[:,1]
    # Ensure inputs are NumPy arrays (in case they are lists)
    y1, y2, y1_hat, y2_hat = map(np.asarray, [y1, y2, y1_hat, y2_hat])
    # compute delta_phi:
    delta_phi_values = calculate_delta_phi_vector(y1, y2, y1_hat, y2_hat)
    print(delta_phi_values)
    # calculate the sRMSE :
    sRMSE = np.sqrt(np.power(delta_phi_values, 2).mean())
    return sRMSE

# -----------------------------------------
# check the function calculate_delta_phi()
# -----------------------------------------
# check if this function works with scalar inputs:
# y_pred=[1,0]
# y_actual=[0.99,0]
# delta_phi = calculate_delta_phi(y_pred,y_actual)
# print(delta_phi)
# plot the results
# import matplotlib.pyplot as plt
# x=[y_pred[0],y_actual[0]]
# y=[y_pred[1],y_actual[1]]
# plt.scatter(0, 0, color='black', label='Points')
# plt.scatter(x, y, color='red', label='Points')
# angle = delta_phi*np.pi - np.pi/2
# plt.scatter(math.cos(angle), math.sin(angle), color='black', label='Points')
# plt.axis("equal")
# plt.show()

# check if this function works with vector inputs:
# y1 = np.array([1, 0, -1, 0])  # Unit vectors
# y2 = np.array([0, 1, 0, -1])
# y1_hat = -y2 # Rotate (y1, y2) by 90° counterclockwise
# y2_hat = y1
# delta_phi_values = calculate_delta_phi_vector(y1, y2, y1_hat, y2_hat) # Compute delta_phi:
# print("Delta Phi Values:", delta_phi_values) # Print Results: expected output = [ 0.25 , 0.25 , 0.25 , 0.25 ]

# check if vectors can be raised by a power of two elementwise:
# a = np.array([1,2,3,4])
# print(a**2)

# check the function calculate_sRMSE:
y_pred = np.array([[1, 0, -1, 0],[0, 1, 0, -1]])
print(y_pred)
y_actual = np.array([[0, -1, 0, 1],[1, 0, -1, 0]])
sRMSE = calculate_sRMSE(y_pred,y_actual)
print(sRMSE)
# -----------------------------------------
