from fractional_brownian_simulation import *
import matplotlib.pyplot as plt

q = 10
n = 2 ** q + 1
t = 1
h = [0.1, 0.3, 0.5, 0.7, 0.9]
override = {
    'fontsize': 'x-large',
    'verticalalignment': 'center',
    'horizontalalignment': 'center'
}
fig = plt.figure(figsize=(8, 8))

plot1 = fig.add_subplot(511)
plot1.grid(False)
plot1.set_xticks([])
plot1.set_yticks([])
motion = fbm(h[0], n, t)
times = np.arange(len(motion))
plot1.plot(times, motion)
plot1.set_ylabel("H = {}".format(h[0]), override)

plot2 = fig.add_subplot(512)
plot2.grid(False)
plot2.set_xticks([])
plot2.set_yticks([])
motion = fbm(h[1], n, t)
plot2.plot(times, motion)
plot2.set_ylabel("H = {}".format(h[1]), override)

plot3 = fig.add_subplot(513)
plot3.grid(False)
plot3.set_xticks([])
plot3.set_yticks([])
seed = 10^2
rd.seed(seed)
x = rd.normal(size=n)
motion = np.sqrt(t / n) * np.cumsum(x)
plot3.plot(times, motion)
plot3.set_ylabel("H = {}".format(h[2]), override)

plot4 = fig.add_subplot(514)
plot4.grid(False)
plot4.set_xticks([])
plot4.set_yticks([])
motion = fbm(h[3], n, t)
plot4.plot(times, motion)
plot4.set_ylabel("H = {}".format(h[3]), override)

plot5 = fig.add_subplot(515)
plot5.grid(False)
plot5.set_xticks([])
plot5.set_yticks([])
motion = fbm(h[4], n, t)
plot5.plot(times, motion)
plot5.set_ylabel("H = {}".format(h[4]), override)
plt.show()
