import sys

import os
sys.path.append(os.path.abspath(r'C:\Users\hansul\projects\mempy_packages\memypfit\mempyfit\dev_reports'))

from logistic_growth.fitting import fit

sim = fit.simulator(fit.parameters)

import matplotlib.pyplot as plt
fig, ax = fit.data.plot('t-OD')
ax.plot(sim[:,0], sim[:,1])
plt.show()


