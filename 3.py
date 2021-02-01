import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
import numpy as np
import pickle
import random

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
var = list(range(len(data)))
random.shuffle(var)
packets = len(data)/10
temp_x = []
temp_y = []
temp_test_x = []
temp_test_y = []
resampled = []
temp = []

for i in range(0,len(data)):
    if((i+1)%packets == 0):
        temp.append(data[i])
        resampled.append(temp)
        temp = []
    else:
        temp.append(data[i])

for i in data:
    temp_x.append(i[0])
    temp_y.append(i[1])

for i in data2 : 
    temp_test_x.append(i[0])
    temp_test_y.append(i[1])

x= np.array(temp_x).reshape(-1,1)
y = np.array(temp_y)

reg = LinearRegression().fit(x, y)
print('coefficient of determination : ' ,reg.score(x, y))
print("slope : " ,reg.coef_)
print("intercept : " ,reg.intercept_)

plt.plot(temp_x,temp_y,'o')
abline(reg.coef_,reg.intercept_)
# plt.show()