import numpy as np
import random
import math
import matplotlib.pyplot as plt

def logistic_eq(r,x):
    return r*x*(1-x)

# Create the bifurcation diagram
def bifurcation_diagram(seed, n_skip, n_iter, r_min=0,r_nsteps = 1128,x_nsteps= 191,noize_epsilon = 0.01):
    print("Starting with x0 seed {0}, skip plotting first {1} iterations, then plot next {2} iterations.".format(seed, n_skip, n_iter));
    # Array of r values, the x axis of the bifurcation plot
    R = []
    # Array of x_t values, the y axis of the bifurcation plot
    X = []
    RX = np.zeros((x_nsteps,r_nsteps))
    noize_epsilon = 0
    # Create the r values to loop. For each r value we will plot n_iter points
    r_range = np.linspace(r_min, 4, r_nsteps)

    for r_idx,r in enumerate(r_range):
        x = seed;
        # For each r, iterate the logistic function and collect datapoint if n_skip iterations have occurred
        for i in range(n_iter+n_skip+1):
            if i >= n_skip:
                x_idx = math.floor(x * x_nsteps)
                RX[x_idx % x_nsteps,r_idx] += 1
                R.append(r)
                X.append(x)

            x = logistic_eq(r,x)
            x += random.random()*noize_epsilon
            if x>1.0:
                x = 1.0
    # Plot the data
    RX = np.log10(RX+0.000001)
    dpi = 96
    #fig = plt.figure(figsize=(r_nsteps/my_dpi, x_nsteps/my_dpi), dpi=my_dpi)
    #plt.imshow(RX,interpolation='none')


    fig = plt.figure(frameon=False)
    fig.set_size_inches(RX.shape[1]/dpi, RX.shape[0]/dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(RX,interpolation='none')
    #plt.plot(R, X, ls='', marker=',')
    #plt.ylim(0, 1)
    #plt.xlim(r_min, 4)
    #plt.xlabel('r')
    #plt.ylabel('X')
    plt.savefig("logistic.png")

bifurcation_diagram(0.2, 100, 10000, r_min=2.8)
