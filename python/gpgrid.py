import numpy as np
from scipy.linalg import inv
from scipy.sparse import coo_matrix

import logging

def sigmoid(f,s):
    g = 1/(1+np.exp(-s*f))
    return g

def logit(g, s):
    f = -np.log(1/g - 1)/s
    return f

def target_var(f,s,v):
    mean = sigmoid(f,s)
    u = mean*(1-mean)
    v = v*s*s
    return u/(1/(u*v) + 1)

class GPGrid(object):

    rawobsx = []
    rawobsy = []
    rawobs_points = []

    obsx = []
    obsy = []
    obs_points = []
    grid_all = []
    
    obs_f = []
    obs_C = []
    obs_W = []
    
    nx = 0
    ny = 0
        
    G = []
    partialK = []
    z = []
      
    gp = []
    
    K = []
    Q = []
    
    f = []
    C = []
    
    m_post = []
    std_post = []
            
    def __init__(self, nx, ny, s=4, ls=100, calc_full_grid=False):
        self.nx = nx
        self.ny = ny  
        self.s = s #sigma scaling
        self.ls = ls #length-scale
        self.calc_full_grid = calc_full_grid
        
        logging.basicConfig(level=logging.DEBUG)

    
    def process_observations(self, obsx, obsy, obs_points): 
        
        #this can be passed in from the priors in future?
        nu0 = np.array([1, 1],dtype=np.float)
        
        if obs_points==[]:
            return [],[]        
        
        if self.z!=[] and obsx==self.rawobsx and obsy==self.rawobsy and obs_points==self.rawobs_points:
            return
        
        self.rawobsx = obsx
        self.rawobsy = obsy
        self.rawobs_points = obs_points  
            
        logging.debug("GP grid processing " + str(len(self.obsx)) + " observations.")
            
        self.obsx = np.array(obsx)
        self.obsy = np.array(obsy)
        self.obs_flat_idxs = np.ravel_multi_index((self.obsx, self.obsy), (self.nx,self.ny))
        
        if len(obs_points.shape)==1 or obs_points.shape[1]==1:
            self.obs_points = np.array(obs_points).reshape(-1)

            self.grid_all = coo_matrix((np.ones(len(self.obsx)), (self.obsx, self.obsy)), shape=(self.nx,self.ny)).toarray()            
            grid_p = coo_matrix((self.obs_points, (self.obsx, self.obsy)), shape=(self.nx,self.ny)).toarray()
            
            self.obsx, self.obsy = self.grid_all.nonzero()
            presp = grid_p[self.obsx,self.obsy]
            allresp = self.grid_all[self.obsx,self.obsy]
            
        elif obs_points.shape[1]==2:
            presp = np.array(obs_points[:,0]).reshape(obs_points.shape[0],1)
            self.obs_points = presp
            allresp = np.array(obs_points[:,1]).reshape(obs_points.shape[0],1)
            
        self.z = presp/allresp
        self.z = self.z.reshape((self.z.size,1)) - 0.5 

        #Update to produce training matrices only over known points
        obsx_tiled = np.tile(self.obsx, (len(self.obsx),1))
        obsy_tiled = np.tile(self.obsy, (len(self.obsy),1))
        ddx = np.array(obsx_tiled.T - obsx_tiled, dtype=np.float64)
        ddy = np.array(obsy_tiled.T - obsy_tiled, dtype=np.float64)
    
        Kx = np.exp( -ddx**2/self.ls )
        Ky = np.exp( -ddy**2/self.ls )
        K = Kx*Ky
        self.K = K + 1e-6 * np.eye(len(K)) # jitter 
    
        Pr_est = (presp+nu0[1])/(allresp+np.sum(nu0))
        self.Q = np.diagflat(Pr_est*(1-Pr_est)/(allresp+np.sum(nu0)))
    
    def train( self, obsx, obsy, obs_points, new_points=False ):
        self.process_observations(obsx, obsy, obs_points)
        if self.obsx==[]:
            #What is correct way to apply priors? 
            #Adding pseudo-counts will not apply to points that are  not included in training.
            mPr = 0.5
            stdPr = 0.25       
            return mPr, stdPr      
        logging.debug("gp grid starting training...")    
        f = np.zeros(len(self.obsx))
#         start = time.clock()        
        converged = False    
        nIt = 0
        while not converged and nIt<1000:
            old_f = f
        
            mean_X = sigmoid(f,self.s)
            self.G = np.diagflat( self.s*mean_X*(1-mean_X) )
        
            W = self.K.dot(self.G).dot( inv(self.G.dot(self.K).dot(self.G) + self.Q) )
        
            f = W.dot(self.G).dot(self.z) 
            C = self.K - W.dot(self.G).dot(self.K)
        
            diff = np.max(np.abs(f-old_f))
            converged = diff<1e-3
            logging.debug('GPGRID diff = ' + str(diff))
            nIt += 1
#         fin = time.clock()
#         print "train time: " + str(fin-start)            
           
        self.partialK = self.G.dot(np.linalg.inv(self.G.dot(self.K).dot(self.G) + self.Q) );    
        self.obs_f = f.reshape(-1)
        self.obs_C = C
        v = np.diag(C)
        if np.argwhere(v<0).size != 0:
            logging.warning("Variance was negative in GPGrid train()")
            v[v<0] = 0
        mPr_tr = sigmoid(self.obs_f, self.s)
        sdPr_tr = np.sqrt(target_var(self.obs_f, self.s, v))
        
        self.obs_W = W
        
        logging.debug("gp grid trained")
        
        return mPr_tr, sdPr_tr
    
    def post_grid(self): 
        #coordinates we want to predict as a single row
        nout = self.nx*self.ny
        outputx = np.tile(np.arange(self.nx, dtype=np.float).reshape(self.nx, 1), (1, self.ny)).reshape(nout, 1)
        outputy = np.tile(np.arange(self.ny, dtype=np.float).reshape(1, self.ny), (self.nx, 1)).reshape(nout, 1)
                       
        nobs = len(self.obsx)
        maxsize = 2.0 * 10**7
        nsplits = np.ceil(nout*nobs / maxsize)
        splitsize = int(np.ceil(nout/nsplits)) # make sure we don't kill memory

        calc_full_grid = self.calc_full_grid
        if self.f==[]:
            self.f = np.zeros(nout)
            calc_full_grid = True
        if self.C==[]:
            self.C = np.zeros(nout) # diagonal values only
        
        if not calc_full_grid:
            changed_obs = np.abs(self.obs_f - self.f[self.obs_flat_idxs]) > 0.05
         
        Gz = self.G.dot(self.z).reshape(-1)
         
        for s in np.arange(nsplits):
            
            logging.debug("Computing posterior for split %i out of %i" % (s,nsplits))
            
            start = int(s*splitsize)            
            end = int(start+splitsize) if start+splitsize<=nout else -1
            outputx_s = outputx[start:end,:]
            outputy_s = outputy[start:end,:]
            
            ddx = outputx_s - self.obsx
            ddy = outputy_s - self.obsy
            Ky = np.exp( -ddy**2/self.ls )
            Kx = np.exp( -ddx**2/self.ls )
            Kpred = Kx*Ky
            
            #update all idxs?
            if not calc_full_grid:
                changed = np.argwhere(np.sum(Kpred[:,changed_obs],axis=1)>0.1)
                Kpred = Kpred[changed,:]
                changed_s = changed + start
            elif end>=0:
                changed_s = np.arange(start,end)
            else:
                changed_s = np.arange(start,nout)  
                
            W = Kpred.dot(self.partialK)
            
            f_s = W.dot(Gz)
            C_s = 1 - np.sum(W.dot(self.G)*Kpred,axis=1)
            
            self.f[changed_s] = f_s
            self.C[changed_s] = C_s
            
        # Approximate the expected value of the variable transformed through the sigmoid.
        k = (1+(np.pi*self.C/8.0))**(-0.5)
        self.m_post = sigmoid(k*self.f,self.s)
        self.std_post = np.sqrt(target_var(self.f, self.s, self.C))
            
        #sampling is too slow.
        #             Number of samples we'll need
        #             nsamples = 1000
        #             sample from gaussian with mean self.s and variance self.C diagonal
        #             samples = np.random.multivariate_normal(f_s, np.diag(C_s), nsamples)
        #             samples = sigmoid(samples, self.s)
        #             self.m_post[changed] = np.mean(samples)
        #             self.std_post[changed] = np.std(samples)

        #turn them back into a grid
        m_post = self.m_post.reshape((self.nx, self.ny))
        std_post = self.std_post.reshape((self.nx, self.ny))
                
        return m_post, std_post