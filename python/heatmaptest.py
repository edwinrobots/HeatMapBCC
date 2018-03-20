'''
Created on 23 Jun 2014

@author: edwin
'''
import logging
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

def runBCC(C, coords, nreceiver, nx, ny):
    z0 = 0.5
    alpha0 = np.array([[2, 1], [1, 2]])  
    heatmapcombiner = heatmapbcc.HeatMapBCC(2, 2, alpha0, nreceiver, z0)
    heatmapcombiner.minNoIts = 20
    heatmapcombiner.combine_classifications(C, coords)
    bcc_pred, _, _ = heatmapcombiner.predict_grid(nx, ny)
    return bcc_pred[1, :, :], heatmapcombiner

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

    nreceiver, C, coords, gridx, gridy, pop_grid, fig = genSimData(plot_3d_data=True)
    bcc_pred, heatmapcombiner = runBCC(C, coords, nreceiver, nx, ny)
    plotresults()