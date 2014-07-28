import time
import numpy as np
from scipy.linalg import inv

def sigmoid(f,s):
    g = np.divide(1, 1+np.exp(-s*f))
    return g

def target_var(f,s,v):
    mean = sigmoid(f,s)
    u = mean*(1-mean)
    v = v*s*s
    return u/(1/(u*v) + 1)

def latentvariance(std, mean,s):
    prec = 1/np.square(std)
    topvariance = np.multiply(mean,(1-mean))
    prec = prec - np.divide(1,topvariance)
    var = np.divide(1,prec)
    var = np.divide(var, np.square(topvariance))
    var = np.divide(var, np.square(s))
    return var

class GPGrid(object):

    obsx = []
    obsy = []
    obs_points = []
    
    obs_f = []
    obs_C = []
    
    nx = 0
    ny = 0
    
    #nu0 = []
    
    s = 4 #sigma scaling
    ls = 100 #length-scale
    
    G = []
    partialK = []
    z = []
    
    #prior_mean = 0.5
    #prior_mean_latent = 0
        
    def __init__(self, nx, ny, nu0):
        self.nx = nx
        self.ny = ny  
        #self.nu0 = np.float64(nu0)        
    
    def process_observations( self ):
        if self.obs_points==[]:
            return [],[]
            
        self.obsx = np.array(self.obsx)
        self.obsy = np.array(self.obsy)
        if len(self.obs_points.shape)==1 or self.obs_points.shape[1]==1:
            grid_p = np.zeros((self.nx,self.ny))
            grid_all = np.zeros((self.nx,self.ny))
            self.obs_points = np.array(self.obs_points).reshape(-1)
            for i in range(len(self.obsx)):
                grid_p[self.obsx[i], self.obsy[i]] += self.obs_points[i]
                grid_all[self.obsx[i], self.obsy[i]] += 1
               
            deduped_idxs = np.argwhere(grid_all>0)
            self.obsx = deduped_idxs[:,0]
            self.obsy = deduped_idxs[:,1]   
                
            presp = grid_p[self.obsx,self.obsy]
            allresp = grid_all[self.obsx,self.obsy]
            
        elif self.obs_points.shape[1]==2:
            presp = np.array(self.obs_points[:,0]).reshape(self.obs_points.shape[0],1)
            allresp = np.array(self.obs_points[:,1]).reshape(self.obs_points.shape[0],1)
         
        #presp += self.nu0[1]
        #allresp += np.sum(self.nu0)
            
        self.z = np.divide(presp, allresp)
        self.z = self.z.reshape((self.z.size,1))
            
        return presp, allresp 
    
    def train( self, obsx, obsy, obs_points ):
        self.obsx = obsx
        self.obsy = obsy
        self.obs_points = obs_points
                
        presp, allresp = self.process_observations()
        
        #self.prior_mean = self.nu0[1] / np.sum(self.nu0)
        #self.prior_mean_latent = np.log(self.prior_mean/(1-self.prior_mean))
        #prior_var = self.prior_mean*(1-self.prior_mean)/(self.nu0[0]+self.nu0[1]+1)#1/self.nu0[0] + 1/self.nu0[1] #should be dealt with through Q?
        #prior_var_latent = latentvariance(prior_var, self.prior_mean, s) 
        
        if self.obsx==[]:
            print "What is correct way to apply priors? Adding pseudo-counts will not apply to points that" + \
            "are  not included in training."
            mPr = 0.5#self.prior_mean
            stdPr = 0.25#np.sqrt(prior_var)        
            f = 0#self.prior_mean_latent
            var = latentvariance(stdPr, mPr, self.s)
            return f, var, mPr, stdPr
        
        #Update to produce training matrices only over known points
        ddx = np.float64(np.tile(self.obsx, (len(self.obsx),1)).T - np.tile(self.obsx, (len(self.obsx),1)))
        ddy = np.float64(np.tile(self.obsy, (len(self.obsy),1)).T - np.tile(self.obsy, (len(self.obsy),1)))
        
        Kx = np.exp( -ddx**2/self.ls )
        Ky = np.exp( -ddy**2/self.ls )
        K = Kx*Ky
    
        f = np.zeros(len(self.obsx))
        K = K + 1e-6 * np.eye(len(K)) # jitter    
        
        Pr_est = (presp+1)/(allresp+2)
        Q = np.diagflat(Pr_est*(1-Pr_est)/(allresp+2))
        
        converged = False    
        nIt = 0
        while not converged and nIt<1000:
            old_f = f
        
            mean_X = sigmoid(f,self.s)
            self.G = np.diagflat( self.s*mean_X*(1-mean_X) )
        
            W = K.dot(self.G.T).dot( inv(self.G.dot(K).dot(self.G.T)+Q) )
        
            f = W.dot(self.G.T).dot(self.z-0.5)#self.prior_mean) # + self.prior_mean_latent 
            C = K - W.dot(self.G).dot(K) #possibly needs additional variance term sicne the observation is uncertain?
        
            diff = np.max(np.abs(f-old_f))
            converged = diff<1e-3
            #print 'GPGRID diff = ' + str(diff)
            nIt += 1
           
        self.partialK = self.G.T.dot(np.linalg.inv(self.G.dot(K).dot(self.G.T) + Q) );    
        self.obs_f = f.reshape(-1)
        self.obs_C = C
        v = np.diag(C)
        mPr_tr = sigmoid(self.obs_f, self.s)
        sdPr_tr = np.sqrt(target_var(self.obs_f, self.s, v))
        
        return self.obs_f, C, mPr_tr, sdPr_tr

    def post_peaks(self, f, C):     
        v = np.diag(C)
        stdPr = target_var(f,self.s,v)
        mPr = sigmoid(f,self.s)
          
        return mPr, stdPr
    
    def post_grid(self):
        
        ddx = np.float64(-self.obsx)
        ddy = np.float64(-self.obsy)
        
        f_end = self.G.T.dot(self.z-0.5)#self.prior_mean) 
        
        f = np.zeros((self.nx,self.ny), dtype=np.float64)# + self.prior_mean_latent
        C = np.ones((self.nx,self.ny), dtype=np.float64)
                
        start = time.clock()

        for i in np.arange(self.nx):
            for j in np.arange(self.ny):
                Kx = np.exp( -ddx**2/self.ls )
                Ky = np.exp( -ddy**2/self.ls )
        
                Kpred = Kx*Ky
        
                W = Kpred.dot(self.partialK)
        
                f[i,j] += W.dot(f_end)
                C[i,j] -= W.dot(self.G).dot(Kpred.T)
                
                ddy += 1
            ddx += 1
            ddy -= self.ny

        fin = time.clock()
        print "gpgrid prediction timer: " + str(fin-start)

        mPr = sigmoid(f,self.s)
        stdPr = np.sqrt(target_var(f, self.s, C))
        
        return mPr, stdPr, f, C
         
    def post_grid_noloops(self):           
        ddy = np.arange(self.ny,dtype=np.float64).reshape((self.ny,1)) - np.tile(self.obsy, (self.ny,1))
        Ky = np.exp( -ddy**2/self.ls )
          
        fending = self.G.T.dot(self.z-0.5)#self.prior_mean)
        ddx = np.arange(self.nx,dtype=np.float64).reshape((self.nx,1)) - np.tile(self.obsx, (self.nx,1))
        Kx = np.exp( -ddx**2/self.ls )  
        
        start = time.clock()
    
        Kx = np.tile( Kx, (1,self.ny))
        Kx = Kx.reshape((self.nx*self.ny, len(self.obsx)))
        Ky = np.tile(Ky, (self.nx,1))
        Kpred = Kx*Ky
        W_i = Kpred.dot(self.partialK)  
        f = W_i.dot(fending).reshape((self.nx,self.ny))#self.prior_mean_latent  +  
        C = np.ones(self.nx*self.ny)
        Cwg = W_i.dot(self.G)
        for i in np.arange(self.nx*self.ny):
            C[i] -= Cwg[i,:].dot(Kpred[i,:].T)
        #C -= np.sum(W_i.dot(self.G)*Kpred, axis=1)
        C = C.reshape((self.nx,self.ny))
        
        fin = time.clock()
        print "gpgrid prediction timer: " + str(fin-start)
                
        mPr = sigmoid(f,self.s)
        stdPr = np.sqrt(target_var(f, self.s, C))
        
        return mPr, stdPr, f, C
