import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression

data = [
    {"x":0,"y":0},
    {"x":1,"y":2},
    {"x":2,"y":4},
    {"x":3,"y":6},
    {"x":4,"y":8},
    {"x":5,"y":10},
    {"x":6,"y":12},
]

data = pd.DataFrame(data)
# data
targets = data.y
given = data.x
given = np.array(given)
given = given[:, None]
given
reg = LinearRegression()
reg.fit(given, targets)
print(reg.predict([[256]]))

# data
scatter_matrix(data)
plt.show()
