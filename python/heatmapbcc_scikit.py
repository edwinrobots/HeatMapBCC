from heatmapbcc import Heatmapbcc
from sklearn.gaussian_process import GaussianProcess
import logging
from gpgrid import sigmoid, logit
import numpy as np
from scipy.sparse import coo_matrix

class GP_scikit_wrapper:
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny
        
    def processobservations(self, obsx, obsy, obs_points):
        self.rawobsx = obsx
        self.rawobsy = obsy
        self.rawobs_points = obs_points  
            
        logging.debug("GP grid processing " + str(len(self.obsx)) + " observations.")
            
        self.obsx = np.array(obsx)
        self.obsy = np.array(obsy)
        if len(obs_points.shape)==1 or obs_points.shape[1]==1:
            self.obs_points = np.array(obs_points).reshape(-1)
            grid_all = coo_matrix((np.ones(len(self.obsx)), (self.obsx, self.obsy)), shape=(self.nx,self.ny)).toarray()
            self.obsx, self.obsy = grid_all.nonzero()            
            grid_p = coo_matrix((self.obs_points, (self.obsx, self.obsy)), shape=(self.nx,self.ny)).toarray()
            self.obs_points = grid_p[self.obsx,self.obsy]    
        
    def train(self, obsx, obsy, obs_points):
        print 'learn the posterior over function at the observed points'

        self.processobservations(obsx, obsy, obs_points)
        
        #instantiate the GP object
        self.gp = GaussianProcess(theta0=50, thetaL=1, thetaU=200) #what does theta0 do?
        
        #train it
        X = np.concatenate((self.obsx,self.obsy), axis=1)
        Y = logit(self.obs_points, 1)
        self.gp.fit(X, Y)
        
    def post_grid(self):
        print 'return a grid of posterior mean and var for prediction points'
        Xpredx = np.tile(np.arange(self.nx), (1,self.ny))
        Xpredy = np.tile(np.arange(self.ny), (1,self.nx))
        Xpred = np.concatenate((Xpredx.reshape((self.nx*self.ny,1)), Xpredy.T.reshape((self.nx*self.ny,1)) ),axis=1)
        Ypred = self.gp.predict(Xpred.T)
        kappa = sigmoid(Ypred) #this is a poor approximation
        return kappa.reshape((self.nx,self.ny)) # no variance?

class Heatmapbcc_scikit(Heatmapbcc):
    def createGP(self):
        return GP_scikit_wrapper(self.nx,self.ny)