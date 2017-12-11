import numpy as np
import matplotlib.pyplot as plt


# we compute the empirical q-variation of the log volatility (proxied with realized var)
def m(motion, t, q, delta):
    if delta == 1:
        return np.mean(np.power(np.abs(np.log(motion[1:]) - np.log(motion[:-1])), q))
    n = int(np.floor(t / delta))
    index = np.arange(n - 1) * int(delta)
    starts = np.arange(delta - 1)
    # compute for all starting points
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
        plt.scatter(logDelta, arr_res, s=0.5, color=colors[i], label=" q = {} ".format(arr_q[i]))
    plt.legend(loc="best")
    plt.show()

