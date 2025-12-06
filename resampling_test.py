from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

n = 1000
dist1 = stats.norm(10, 0.25)
dist2 = stats.norm(13, 2.3)


samples1 = dist1.rvs(n)
samples2 = dist2.rvs(n)

plt.hist(samples1)

def diff_sd(samples1, samples2):
    return np.sqrt(np.std(samples1)**2/len(samples1) + np.std(samples2)**2/n)

diff_sd_analytical = diff_sd(samples1, samples2)

def diff_sd_bootstrap(samples1, samples2, num_bootstrap_trials = 1000):

    diffs_resampled = []

    for _ in range(num_bootstrap_trials):
        a = np.random.choice(samples1, size = n, replace = True)
        b = np.random.choice(samples2, size = n, replace = True)
        diff = np.mean(b) - np.mean(a)
        diffs_resampled.append(diff)

    return np.std(diffs_resampled)


def plot_pcterrs():
    bootstrap_efforts = np.linspace(100, 1e6, 100)
    errvals = []

    for nb in bootstrap_efforts:
        diff_sd_approx = diff_sd_bootstrap(samples1, samples2, num_bootstrap_trials=int(nb))
        errvals.append(np.abs(diff_sd_approx - diff_sd_analytical)/diff_sd_analytical)

    plt.plot(bootstrap_effort, errvals, marker = 'o')
    plt.show()

    
plot_pcterrs()