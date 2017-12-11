import numpy as np
import matplotlib.pyplot as plt


# we compute the empirical q-variation of the log volatility (proxied with realized var)
def m(motion, t, q, delta):
    n = int(np.floor(t / delta))
    nb_starting_points = int(t - n * delta) + 1
    index = np.arange(n) * int(delta)
    # if one starting possible starting point
    if nb_starting_points == 1:
        log_fbm = np.log(motion[index])
        # directly compute value
        return np.mean(np.power(np.abs(log_fbm[1:] - log_fbm[:-1]), q))
    # if multiple possible starting points
    else:
        # compute for all starting points
        starts = np.arange(nb_starting_points)
        copies_motion = np.stack([motion[start:][index] for start in starts])
        log_fbm = np.log(copies_motion)
        # take the mean
        return np.mean(np.power(np.abs(log_fbm[:, 1:] - log_fbm[:, :-1]), q))


# we plot the the values of log(m(q, delta)) against log(delta)
def plotVar(motion, t):
    plt.figure()
    plt.title("Plot of log(m(q, delta)) against log(delta) \n for different values of q")
    arr_delta = np.arange(1, 50)
    logDelta = np.log(arr_delta)
    arr_q = [0.5, 1, 1.5, 2, 3]
    colors = ["r", "g", "b", "y", "k"]
    # compute for each value of q/delta
    for i in range(len(arr_q)):
        arr_res = []
        for j in range(arr_delta.size):
            logM = np.log(m(motion, t, arr_q[i], arr_delta[j]))
            arr_res.append(logM)
        # plot the results
        plt.plot(logDelta, arr_res,color=colors[i], label=" q = {} ".format(arr_q[i]))
    plt.legend(loc="best")
    plt.show()

