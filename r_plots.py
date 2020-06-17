import matplotlib.pyplot as plt
import numpy as np


t1 = [1.0874, 1.0799, 1.0713, 1.1559, 1.3387, 1.7548, 2.8392, 3.0896, 3.2763, 3.797, 6.0081]
t2 = [0.885, 0.895, 0.895, 0.89, 0.8925, 0.885, 0.895, 0.8975, 0.9025, 0.89, 0.88]
t3 = [0.8715, 0.8636, 0.8619, 0.8651, 0.8712, 0.8686, 0.8644, 0.8533, 0.8505, 0.8461, 0.8459]

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('iterations')
ax1.set_ylabel('kl_similarity', color=color)
ax1.plot(t1, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('bce loss', color=color)  # we already handled the x-label with ax1
ax2.plot(t3, 'bs')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

