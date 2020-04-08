# Jane Watts and Tim Player
# E205 lab 2 part 2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb

data = pd.read_excel('E205_Lab2_NuScenesData.xlsx')
x_4 = data['X_4']
y_4 = data['Y_4']
sp = get_speed(x_4, y_4, delta_t)
pdb.set_trace()


def get_speed(x, y, delta_t, data):
    # x and y are Series, delta_t is timestep (e.g. 0.5)
    speeds = []
    for i in range((len(x) - 1))
    this_speed = np.sqrt((x[i+1] - x[i]) ** 2 + (y[i+1] - y[i]) ** 2)
    speeds.append(this_speed)
    return pd.Series(speeds)
