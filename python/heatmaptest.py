'''
Created on 23 Jun 2014

@author: edwin
'''
import logging

from scipy.stats import multivariate_normal as mvn, norm

logging.basicConfig(level=logging.DEBUG)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import heatmapbcc
import gp_classifier_vb
# from matplotlib import cm


def genSimData(plot_3d_data=True):
    #SIMULATED DATA GENERATION    
    cx = scale * np.array([2, 5, 7])
    cy = scale * np.array([2, 5, 6])

    pop = np.array([1000, 500, 2000])
    
    ncentres = len(cx)
    
    gridx = np.tile( range(1,nx+1),(ny,1))
    gridy = np.tile(range(1,ny+1), (nx,1)).transpose()
    
    pop_grid = np.zeros( (nx,ny) )
    
    for ic in range(ncentres):
        pop_grid = np.round( pop_grid + pop[ic]*np.exp(\
                              -0.5*np.sqrt(np.square(gridx-cx[ic]) \
                              +np.square(gridy-cy[ic])) ))
        
    epi_centrex = scale*9
    epi_centrey = scale*8
    
    damage_grid = np.exp(-0.05*np.sqrt( np.square(gridx-epi_centrex) + np.square(gridy-epi_centrey)))
    
    fig = plt.figure(1)    
    if plot_3d_data:
        ax = fig.add_subplot(2,2,1,projection='3d')
        surf = ax.plot_surface(gridx, gridy, damage_grid, cstride=1, rstride=1, cmap=plt.get_cmap('spectral'))
        #     zlabel('Probability of damage','FontSize',15)
        plt.xlabel('Long')
        plt.ylabel('Lat')
        ax.set_zlim3d(0, 1)
    
        ax = fig.add_subplot(2, 2, 2, projection='3d')
        surf = ax.plot_surface(gridx, gridy, pop_grid, cstride=1, rstride=1, cmap=plt.get_cmap('spectral'))
        #     zlabel('Population density','FontSize',15)
        plt.xlabel('Long')
        plt.ylabel('Lat')
    
    #Generate some reports
    rec_x = scale * np.array([2, 5, 7, 9]);
    rec_y = scale * np.array([2, 5, 7, 8]);
    pop_response = np.array([0.1, 0.3, 0.7, 0.5]);
     
    nreceiver = len(rec_x);
    rec_range = scale*1;
    
    nresp = np.zeros(nreceiver,dtype=np.int64)
    npresp = np.zeros(nreceiver)
    C = None
    point_coords = None
        
    for i in range(nreceiver):
        detection_area = np.where( np.sqrt( np.square(gridx-rec_x[i]) \
                                               + np.square(gridy-rec_y[i])) \
                                               < rec_range)
    
        nresp[i] = np.sum( np.sum( pop_response[i]*pop_grid[detection_area]))
        nresp[i] = np.int64(nresp[i])             

        npresp[i] = np.sum( np.sum( np.random.binomial( \
                        np.int64(pop_response[i]*pop_grid[detection_area]), \
                        damage_grid[detection_area]) ))

        np.int64(npresp[i])

        for n in range(nresp[i]):
            repx = rec_x[i] + np.random.randint(-rec_range,rec_range)
            repy = rec_y[i] + np.random.randint(-rec_range,rec_range)
            if n<=npresp[i]:
                rep = 1
            else:
                rep = 0
            Crow = np.array([[i, n, rep]])
            coords_n = np.array([[n, repx, repy]])
            if C is None:
                C = Crow
                point_coords = coords_n
            else:
                C = np.concatenate((C, Crow), axis=0)
                point_coords = np.concatenate((point_coords, coords_n), axis=0)
                
    return nreceiver, C, point_coords, gridx, gridy, pop_grid, fig


def gen_synthetic_classifications(maxval=100, ndim=5):
    # f_prior_mean should contain the means for all the grid squares

    # Generate some data
    ls = np.ones(ndim) + 10
    sigma = 0.1
    N = 2000
    # C = 100 # number of labels for training
    s = 1  # inverse precision scale for the latent function.

    # Some random feature values
    xvals = []
    for dim in range(ndim):
        xvals.append(np.random.choice(maxval, N, replace=True))

    # remove repeated coordinates
    for coord in range(N):

        checks = np.array([xivals == xivals[coord] for xivals in xvals])
        occurrences = np.sum(checks, axis=0) == ndim

        while np.sum(occurrences) > 1:
            xvals[np.random.randint(0, ndim)][coord] = np.random.choice(maxval, 1)
            checks = np.array([xivals == xivals[coord] for xivals in xvals])
            occurrences = np.sum(checks, axis=0) == ndim

        print(coord)

    K = gp_classifier_vb.matern_3_2_from_raw_vals(np.array(xvals).T, ls)
    f = mvn.rvs(cov=K / s)  # zero mean

    # generate the noisy function values for the pairs
    fnoisy = norm.rvs(scale=sigma, size=N) + f

    # generate the discrete labels from the noisy function
    labels = np.round(gp_classifier_vb.sigmoid(fnoisy)).astype(int)

    xvals = np.array(xvals)
    return N, labels, xvals, f, K

def gen_5D_data():
    nsources = 4 # number of workers
    maxval = 10


    N, labels, xvals, f, K = gen_synthetic_classifications(maxval, 5)
    Ntest = int(N * 0.1)
    Ntrain = N - Ntest

    max_good = 20
    max_bad = 3

    C = None

    for s in range(nsources):
        alpha0 = np.array([[np.random.randint(1, max_good), np.random.randint(1, max_bad)],
                           [np.random.randint(1, max_bad),  np.random.randint(1, max_good)]])

        pi0 = alpha0 / np.sum(alpha0, axis=1)[:, None]

        for i in range(Ntrain):
            rep = np.random.rand() < pi0[labels[i], 1]
            Crow = np.array([[s, i, rep]])
            if C is not None:
                C = np.concatenate((C, Crow), axis=0)
            else:
                C = Crow

    coords = np.concatenate((np.arange(N)[:, None], xvals.T), axis=1)
    test_coords = coords[Ntrain:]
    return nsources, C, coords, test_coords


def runBCC(C, coords, nreceiver, nx, ny):
    z0 = 0.5
    alpha0 = np.array([[2, 1], [1, 2]])  
    heatmapcombiner = heatmapbcc.HeatMapBCC(2, 2, alpha0, nreceiver, z0)
    heatmapcombiner.verbose = False
    # heatmapcombiner.min_iterations = 200
    heatmapcombiner.uselowerbound = True#False
    heatmapcombiner.combine_classifications(C, coords)
    bcc_pred, _, _ = heatmapcombiner.predict_grid(nx, ny)
    return bcc_pred[1, :, :], heatmapcombiner

def runBCC_5D(C, coords, nsources, test_coords):
    z0 = 0.5
    alpha0 = np.array([[2, 1], [1, 2]])
    heatmapcombiner = heatmapbcc.HeatMapBCC(2, 2, alpha0, nsources, z0)
    heatmapcombiner.verbose = False
    # heatmapcombiner.min_iterations = 200
    heatmapcombiner.uselowerbound = True#False
    heatmapcombiner.combine_classifications(C, coords)
    bcc_pred, _, _ = heatmapcombiner.predict(test_coords, variance_method='sample')
    return bcc_pred[1, :], heatmapcombiner


def plotresults():
    
    ax = fig.add_subplot(2, 2, 3, projection='3d')    
    ax.plot_surface(gridx, gridy, bcc_pred, cstride=1, rstride=1, cmap=plt.get_cmap('spectral'))
#     zlabel('Pr(damage has occurred in this grid square)');
    plt.xlabel('Long');
    plt.ylabel('Lat');
    ax.set_zlim3d(0, 1)
       
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.plot_surface(gridx, gridy, np.multiply(bcc_pred,pop_grid), cstride=1, rstride=1, cmap=plt.get_cmap('spectral'))
#     zlabel('Expected number of individuals with damage');
    plt.xlabel('Long');
    plt.ylabel('Lat');
    
if __name__ == '__main__':

    scale = 3
    
    nx = scale*10
    ny = scale*10

    # nreceiver, C, coords, gridx, gridy, pop_grid, fig = genSimData(plot_3d_data=True)
    # bcc_pred, heatmapcombiner = runBCC(C, coords, nreceiver, nx, ny)
    # plotresults()

    nsources, C, coords, test_coords = gen_5D_data()
    bcc_pred, heatmapcombiner = runBCC_5D(C, coords, nsources, test_coords)