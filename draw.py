import numpy
import pandas as pd
df = pd.read_csv("output.csv")
print(df["Block"])
t = 0.01 * numpy.array(df["Block"].values)
energy = df["ETotal"].values

import matplotlib.pyplot as plt

plt.plot(t,energy)
plt.show()
