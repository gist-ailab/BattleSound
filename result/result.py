import numpy as np

def acc(x):
    correct = x[0,0] + x[1,1]
    total = np.sum(x)
    return correct * 100 / total

def recall(x):
    x_0 = x[0,0] / (x[0,0] + x[0,1])
    x_1 = x[1,1] / (x[1,0] + x[1,1])
    return (x_0 + x_1) / 2

def precision(x):
    x_0 = x[0,0] / (x[0,0] + x[1,0])
    x_1 = x[1,1] / (x[0,1] + x[1,1])
    return (x_0 + x_1) / 2


def description(a,b,c,d):
    print('1', acc(a), recall(a), precision(a))
    print('2', acc(b), recall(b), precision(b))
    print('3', acc(c), recall(c), precision(c))
    print('4', acc(d), recall(d), precision(d))

    
# VCAD
print('Raw Signal (VCAD)')
a1 = np.array([[3531., 796.], [ 479., 4932.]])
a2 = np.array([[3197., 1130.], [1095., 4316.]])
a3 = np.array([[2757., 1570.], [1730., 3681.]])
a4 = np.array([[2337., 1990.], [2227., 3184.]])
description(a1,a2,a3,a4)

print('Mel-Spectrogram (VCAD)')
a1 = np.array([[3835., 492.], [ 507., 4904.]])
a2 = np.array([[3624., 703.], [1517., 3894.]])
a3 = np.array([[3324., 1003.], [1960., 3451.]])
a4 = np.array([[2827., 1500.], [2377., 3034.]])
description(a1,a2,a3,a4)


# GSED
print('Raw Signal (GSED)')
a1 = np.array([[4532., 520.], [ 740., 5576.]])
a2 = np.array([[4117., 935.], [ 639., 5677.]])
a3 = np.array([[3391., 1661.], [ 930., 5386.]])
a4 = np.array([[1997., 3055.], [2514., 3802.]])
description(a1,a2,a3,a4)

print('Mel-Spectrogram (GSED)')
a1 = np.array([[4660., 392.], [ 447., 5869.]])
a2 = np.array([[4569., 483.], [ 466., 5850.]])
a3 = np.array([[4121., 931.], [ 526., 5790.]])
a4 = np.array([[3200., 1852.], [1849., 4467.]])
description(a1,a2,a3,a4)
