import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
import numpy as np
import pickle
import math

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')

inline = open('../data/train.pkl','rb')
data = pickle.load(inline,encoding='bytes')
inline2 = open('../data/test.pkl','rb')
data2 = pickle.load(inline2,encoding='bytes')
temp_x = []
temp_y = []
temp_test_x = []
temp_test_y = []
for i in data:
    temp_x.append(i[0])
    temp_y.append(i[1])

for i in data2 : 
    temp_test_x.append(i[0])
    temp_test_y.append(i[1])

# print(temp_test_x)
# print(temp_test_y)

x = np.array(temp_x).reshape(-1,1)
y = np.array(temp_y)

test_x = np.array(temp_test_x).reshape(-1,1)

reg = LinearRegression().fit(x, y)
print('coefficient of determination : ' ,reg.score(x, y))
print("slope : " ,reg.coef_)
print("intercept : " ,reg.intercept_)

predicted_data = reg.predict(test_x)
bias = []
avg_prediction = 0
avg_true = 0
variance = 0
avg_bias = 0

# print(len(predicted_data))
# print(len(temp_test_y))

for i in range(0,len(temp_test_y)) : 
    bias.append(predicted_data[i] - temp_test_y[i])
    avg_prediction += predicted_data[i]
    avg_true += temp_test_y[i]

avg_prediction = avg_prediction/len(temp_test_y)
avg_true = avg_true/len(temp_test_y)
print("Avg_true : ",avg_true)

for i in range(0,len(temp_test_y)):
    # bias.append(avg_prediction - temp_test_y[i])
    avg_bias += bias[i]
    variance += math.pow((predicted_data[i] - avg_prediction),2)

avg_bias = avg_bias/len(temp_test_y)

variance = variance/len(temp_test_y)
print("Avg Bias : ",avg_bias)
# print("Bias : \n", bias)
print("Avg Prediction : ",avg_prediction)
print("Variance : ", variance)

plt.plot(temp_x,temp_y,'o')
plt.plot(temp_test_x,temp_test_y,'o')
abline(reg.coef_,reg.intercept_)
plt.show()