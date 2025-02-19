import math
import numpy as np

def calculate_delta_phi(y_pred,y_actual):
    y1=y_actual[0]
    y2=y_actual[1]
    y1_hat = y_pred[0]
    y2_hat = y_pred[1]
    angles = math.atan2( y2_hat*y1 - y1_hat*y2 , y1_hat*y1 + y2_hat*y2 )
    delta_phi = (angles + np.pi/2)/np.pi
    return delta_phi

def calculate_delta_phi_vector(y1,y2,y1_hat,y2_hat):
    # Ensure inputs are NumPy arrays (in case they are lists)
    y1, y2, y1_hat, y2_hat = map(np.asarray, [y1, y2, y1_hat, y2_hat])
    # Compute angles using element-wise operations
    angles = np.arctan2(y2_hat * y1 - y1_hat * y2, y1_hat * y1 + y2_hat * y2)
    # Normalize to range [0,1]
    delta_phi = (angles + np.pi / 2) / np.pi
    return delta_phi  # Returns a NumPy array

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
# -----------------------------------------
