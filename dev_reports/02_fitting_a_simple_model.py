import sys
import os
sys.path.append(os.path.abspath(r'C:\Users\hansul\projects\mempy_packages\memypfit\mempyfit\dev_reports'))
from dev_reports.logistic_growth.fitting_problem import prob

sim = prob.simulator(prob.parameters)

import matplotlib.pyplot as plt
fig, ax = prob.data.plot('t-OD')
ax.plot(sim[:,0], sim[:,1])
plt.show()


