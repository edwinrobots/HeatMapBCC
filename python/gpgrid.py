import numpy as np
from scipy.linalg import inv
from scipy.sparse import coo_matrix

import time, logging

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

def latentvariance(std, mean,s):
    prec = 1/np.square(std)
    topvariance = mean*(1-mean)
    prec = prec - 1/topvariance
    var = 1/prec
    var = var/topvariance**2
    var = var/s**2
    return var

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
        
    s = 4 #sigma scaling
    ls = 100 #length-scale
    
    G = []
    partialK = []
    z = []
      
    gp = []
    
    K = []
    Q = []
    
    f = []
    C = []
        
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny  
    
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

    def post_peaks(self, f, C):     
        v = np.diag(C)
        stdPr = target_var(f,self.s,v)
        mPr = sigmoid(f,self.s)
          
        return mPr, stdPr
    
    def post_grid_fast(self):
        #A fast approximation that updates only the areas that have 
        #changed more than the specified amount
        f_end = self.G.dot(self.z) 
        
        update_indicator = np.zeros((self.nx,self.ny), dtype=np.bool)
        #f = np.zeros((self.nx,self.ny), dtype=np.float64)
        #C = np.ones((self.nx,self.ny), dtype=np.float64)

        update_indicator[self.obsx, self.obsy] = True
        self.f[self.obsx,self.obsy] = self.obs_f
        self.C[self.obsx, self.obsy] = np.diag(self.obs_C)
        
        nobs = len(self.obsx)
        logging.info("GP grid predicting posterior given " + str(nobs) + " observations.")

        for o in np.arange(nobs):
            logging.debug("Post grid fast update observation " + str(o) + " out of " + str(nobs))
            neighbour_x = 1
            neighbour_y = 1
            done = False
            while not done:
                nupdated = 0
                diff = 0
                for x in np.arange(neighbour_x+1):
                    for y in np.arange(neighbour_y+1):
                        #evaluate these points
                        i = self.obsx[o] + x
                        j = self.obsy[o] + y
                           
                        if i<self.nx and j<self.ny and not update_indicator[i,j]:
                            f, C = self.post_grid_square(i, j, f_end)
                            nupdated+=1
                            diff += np.abs(self.f[i,j]-f)
                            self.f[i,j] = f
                            self.C[i,j] = C
                            update_indicator[i,j] = True
                                                
                        i = self.obsx[o] - x
                        j = self.obsy[o] - y
                        if i<self.nx and j<self.ny and not update_indicator[i,j]:
                            f, C = self.post_grid_square(i, j, f_end)
                            nupdated+=1
                            diff += np.abs(self.f[i,j]-f)
                            self.f[i,j] = f
                            self.C[i,j] = C
                            update_indicator[i,j] = True                        
                        
                        i = self.obsx[o] + x
                        j = self.obsy[o] - y
                        if i<self.nx and j<self.ny and not update_indicator[i,j]:
                            f, C = self.post_grid_square(i, j, f_end)
                            nupdated+=1
                            diff += np.abs(self.f[i,j]-f)
                            self.f[i,j] = f
                            self.C[i,j] = C
                            update_indicator[i,j] = True                            
                        
                        i = self.obsx[o] - x
                        j = self.obsy[o] + y
                        if i<self.nx and j<self.ny and not update_indicator[i,j]:
                            f, C = self.post_grid_square(i, j, f_end)
                            nupdated+=1
                            diff += np.abs(self.f[i,j]-f)
                            self.f[i,j] = f
                            self.C[i,j] = C
                            update_indicator[i,j] = True                            
                        
                #calculate total difference
                if nupdated>0:          
                    diff = diff/nupdated
                else:
                    diff = 0
                logging.debug("fast update " + str(diff))
                
                if (diff<0.05 or nupdated==0) and neighbour_x>2:
                    done = True
                    
                neighbour_x += 1
                neighbour_y += 1
                    
        mPr = sigmoid(self.f,self.s)
        stdPr = np.sqrt(target_var(self.f, self.s, self.C))
        
        return mPr, stdPr 
    
    def post_grid_square(self, i, j, f_end):
        
        ddx = np.float64(i - self.obsx)
        ddy = np.float64(j - self.obsy)
        
        
        Kx = np.exp( -ddx**2/self.ls )
        Ky = np.exp( -ddy**2/self.ls )
        Kpred = Kx*Ky
        
        W = Kpred.dot(self.partialK)
        
        f = W.dot(f_end)
        C = 1 - Kpred.dot(W.dot(self.G))
        return f, C
    
    def post_grid(self):
#         self.f = np.zeros((self.nx,self.ny), dtype=np.float64)
#         self.C = np.ones((self.nx,self.ny), dtype=np.float64)
#         f_end = self.G.dot(self.z) 
#         for i in np.arange(self.nx):
#             for j in np.arange(self.ny):
#                 self.f[i,j], self.C[i,j] = self.post_grid_square(i, j, f_end)
#         
#         mPr = sigmoid(self.f,self.s)
#         stdPr = np.sqrt(target_var(self.f, self.s, self.C))
#         
#         return mPr, stdPr         
        
        if self.f==[]:
            return self.post_grid_oneloop()
        else:
            return self.post_grid_fast()
    
    def post_grid_oneloop(self):
        
        ddx = np.array(-self.obsx, dtype=np.float64).reshape((1,len(self.obsx)))
        ddy = np.tile(np.array(-self.obsy, dtype=np.float64), (self.ny,1))
        
        f_end = self.G.dot(self.z) 
        
        f = np.zeros((self.nx,self.ny), dtype=np.float64)
        C = np.ones((self.nx,self.ny), dtype=np.float64)

        j = np.arange(self.ny).reshape(self.ny,1)
        ddy = j + ddy
        Ky = np.exp( -ddy**2/self.ls )
        
#         start = time.clock()
        for i in np.arange(self.nx):
            Kx = np.exp( -ddx**2/self.ls )
            Kpred = Kx*Ky
        
            W = Kpred.dot(self.partialK)
        
            f[i,:] += W.dot(f_end).reshape(-1)
            C[i,:] -= np.sum(W.dot(self.G)*Kpred, axis=1).reshape(-1)
            ddx += 1

        print 'This is an incorrect way of estimating the mean and variance -- look up better approximations'
        mPr = sigmoid(f,self.s)
        stdPr = np.sqrt(target_var(f, self.s, C))
#         fin = time.clock()
#         print "pred time: " + str(fin-start)
        
        self.f = f
        self.C = C
        
        return mPr, stdPr   
    
    def post_grid_loops(self):
        
        ddx = np.array(-self.obsx, dtype=np.float64)
        ddy = np.array(-self.obsy, dtype=np.float64)
        
        f_end = self.G.dot(self.z) 
        
        f = np.zeros((self.nx,self.ny), dtype=np.float64)
        C = np.ones((self.nx,self.ny), dtype=np.float64)

        for i in np.arange(self.nx):
            Kx = np.exp( -ddx**2/self.ls )
            for j in np.arange(self.ny):
                Ky = np.exp( -ddy**2/self.ls )
                Kpred = Kx*Ky
        
                W = Kpred.dot(self.partialK)
        
                f[i,j] += W.dot(f_end)
                C[i,j] -= W.dot(self.G).dot(Kpred.T) #!!! Is G correct here or should it be transposed? Also, is Kpred correct?
                
                ddy += 1
            ddx += 1
            ddy -= self.ny

        mPr = sigmoid(f,self.s)
        stdPr = np.sqrt(target_var(f, self.s, C))
                
        self.f = f
        self.C = C
        
        return mPr, stdPr
         
    def post_grid_noloops(self):           
        ddy = np.arange(self.ny,dtype=np.float64).reshape((self.ny,1)) - np.tile(self.obsy, (self.ny,1))
        Ky = np.exp( -ddy**2/self.ls )
          
        fending = self.G.dot(self.z)
        ddx = np.arange(self.nx,dtype=np.float64).reshape((self.nx,1)) - np.tile(self.obsx, (self.nx,1))
        Kx = np.exp( -ddx**2/self.ls )      
        Kx = np.tile( Kx, (1,self.ny))
        Kx = Kx.reshape((self.nx*self.ny, len(self.obsx)))
        Ky = np.tile(Ky, (self.nx,1))
        Kpred = Kx*Ky
        W_i = Kpred.dot(self.partialK)  
        f = W_i.dot(fending).reshape((self.nx,self.ny))  
        C = np.ones(self.nx*self.ny)
        #Cwg = W_i.dot(self.G)
        #for i in np.arange(self.nx*self.ny):
        #    C[i] -= Cwg[i,:].dot(Kpred[i,:].T)
        C -= np.sum(W_i.dot(self.G)*Kpred, axis=1)
        C = C.reshape((self.nx,self.ny))

        mPr = sigmoid(f,self.s)
        stdPr = np.sqrt(target_var(f, self.s, C))
        
        self.f = f
        self.C = C        
        
        return mPr, stdPr
