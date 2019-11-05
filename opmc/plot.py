import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('scores.csv')
ax = plt.gca()

df.plot(kind='line', x='frame_count', y='avg_score', ax=ax)

plt.show()