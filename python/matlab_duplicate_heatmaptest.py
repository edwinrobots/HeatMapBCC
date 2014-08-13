'''
Created on 23 Jun 2014

@author: edwin
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import heatmapbcc
import gpgrid
# from matplotlib import cm


def genSimData():
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
    ax = fig.add_subplot(2,2,1,projection='3d')
    surf = ax.plot_surface(gridx, gridy, damage_grid, cstride=1, rstride=1, cmap=plt.get_cmap('spectral'))
#     zlabel('Probability of damage','FontSize',15)
    plt.xlabel('Long')
    plt.ylabel('Lat')
    
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    surf = ax.plot_surface(gridx, gridy, pop_grid, cstride=1, rstride=1, cmap=plt.get_cmap('spectral'))
#     zlabel('Population density','FontSize',15)
    plt.xlabel('Long')
    plt.ylabel('Lat')
    
    #Generate some reports
    snx = scale * np.array([2, 5, 7, 9]);
    sny = scale * np.array([2, 5, 7, 8]);
    pop_response = np.array([0.1, 0.3, 0.7, 0.5]);
     
    nsirran = len(snx);
    sn_range = scale*1;
    
    nresp = np.zeros(nsirran,dtype=np.int64)
    npresp = np.zeros(nsirran)
    C = []
    
    presp = [225,894,3735,547]
    
    for i in range(nsirran):
        detection_area = np.where( np.sqrt( np.square(gridx-snx[i]) \
                                               + np.square(gridy-sny[i])) \
                                               < sn_range)
    
        nresp[i] = np.sum( np.sum( pop_response[i]*pop_grid[detection_area]))
        nresp[i] = np.int64(nresp[i])
#         
#         npresp[i] = np.sum( np.sum( np.random.binomial( \
#                         np.int64(pop_response[i]*pop_grid[detection_area]), \
#                         damage_grid[detection_area]) ))        
        
#         npresp[i] = np.sum( np.sum( np.random.binomial( np.int64(pop_response[i]*pop_grid[detection_area]), \
#                           damage_grid[detection_area]>0.2) ))

        npresp[i] = presp[i]#np.int64(npresp[i])
            
        Crow = np.array([i, snx[i]-1, sny[i]-1, npresp[i], nresp[i]]).reshape((1,5))        
        if C==[]:
            C = Crow
        else:        
            C = np.concatenate((C,Crow), axis=0)
#         for n in range(nresp[i]):
#             repx = snx[i] + np.random.randint(-sn_range,sn_range)
#             repy = sny[i] + np.random.randint(-sn_range,sn_range)
#             if n<=npresp[i]:
#                 rep = 1
#             else:
#                 rep = 0
#             Crow = np.matrix([i, repx, repy, rep])
#             if C==[]:
#                 C = Crow
#             else:
#                 C = np.concatenate((C, Crow), axis=0)    
                
    return nsirran, C, gridx, gridy, pop_grid, fig

def runBCC(C,nx,ny,nsirran):
    nu0 = np.array([50.0, 50.0])
    alpha0 = np.array([[2, 1], [1, 2]])  
    combiner = heatmapbcc.Heatmapbcc(nx, ny, 2, 2, alpha0, nu0, nsirran)
    bcc_pred = combiner.combineClassifications(C)
    bcc_pred = bcc_pred[1,:,:].reshape((nx,ny))
    return bcc_pred    

def runGPDirectly(C,nx,ny):
    
    obs_x = C[:,1]
    obs_y = C[:,2]
    obs_points = C[:,3:C.shape[1]]
    
    f, Cov, mPr, stdPr, s = gpgrid.infer_gpgrid(obs_x, obs_y, obs_points, nx, ny)
    
    return mPr

def plotresults():
    
    ax = fig.add_subplot(2, 2, 3, projection='3d')    
    ax.plot_surface(gridx, gridy, bcc_pred, cstride=1, rstride=1, cmap=plt.get_cmap('spectral'))
#     zlabel('Pr(damage has occurred in this grid square)');
    plt.xlabel('Long');
    plt.ylabel('Lat');
       
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.plot_surface(gridx, gridy, np.multiply(bcc_pred,pop_grid), cstride=1, rstride=1, cmap=plt.get_cmap('spectral'))
#     zlabel('Expected number of individuals with damage');
    plt.xlabel('Long');
    plt.ylabel('Lat');    

if __name__ == '__main__':

    scale = 2
    
    nx = scale*10
    ny = scale*10

    nsirran, C, gridx, gridy, pop_grid, fig = genSimData()
    bcc_pred = runGPDirectly(C,nx,ny)#runBCC(C,nx,ny,nsirran)
    plotresults()