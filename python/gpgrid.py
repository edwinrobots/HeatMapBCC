import numpy as np
from scipy.linalg import inv
from scipy.sparse import coo_matrix

import time

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
        
    s = 4 #sigma scaling
    ls = 100 #length-scale
    
    G = []
    partialK = []
    z = []
      
    gp = []
    
    K = []
    Q = []    
        
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny  
    
    def process_observations( self ):
        if self.obs_points==[]:
            return [],[]
            
        #print "GP grid processing " + str(len(self.obsx)) + " observations."
            
        self.obsx = np.array(self.obsx)
        self.obsy = np.array(self.obsy)
        if len(self.obs_points.shape)==1 or self.obs_points.shape[1]==1:
            
            self.obs_points = np.array(self.obs_points).reshape(-1)
            
            grid_p = coo_matrix((self.obs_points, (self.obsx, self.obsy)), shape=(self.nx,self.ny))
            grid_all = coo_matrix((np.ones(len(self.obsx)), (self.obsx, self.obsy)), shape=(self.nx,self.ny))
            
            self.obsx, self.obsy = grid_all.nonzero()
            presp = np.array(grid_p.todok().values())
            allresp = np.array(grid_all.todok().values())
            
        elif self.obs_points.shape[1]==2:
            presp = np.array(self.obs_points[:,0]).reshape(self.obs_points.shape[0],1)
            allresp = np.array(self.obs_points[:,1]).reshape(self.obs_points.shape[0],1)
            
        self.z = presp/allresp
        self.z = self.z.reshape((self.z.size,1)) - 0.5 

        #Update to produce training matrices only over known points
        ddx = np.float64(np.tile(self.obsx, (len(self.obsx),1)).T - np.tile(self.obsx, (len(self.obsx),1)))
        ddy = np.float64(np.tile(self.obsy, (len(self.obsy),1)).T - np.tile(self.obsy, (len(self.obsy),1)))
    
        Kx = np.exp( -ddx**2/self.ls )
        Ky = np.exp( -ddy**2/self.ls )
        K = Kx*Ky
        self.K = K + 1e-6 * np.eye(len(K)) # jitter 
            
    
        Pr_est = (presp+1)/(allresp+2)
        self.Q = np.diagflat(Pr_est*(1-Pr_est)/(allresp+2))
    
    def train( self, obsx, obsy, obs_points, new_points=False ):
        self.obsx = obsx
        self.obsy = obsy
        self.obs_points = obs_points
                
        if self.z==[] or new_points:
            self.process_observations()
        
        #print "gp grid starting training..."
        
        if self.obsx==[]:
            print "What is correct way to apply priors? Adding pseudo-counts will not apply to points that" + \
            "are  not included in training."
            mPr = 0.5
            stdPr = 0.25       
            f = 0
            var = latentvariance(stdPr, mPr, self.s)
            return f, var, mPr, stdPr      
    
        f = np.zeros(len(self.obsx))
        #print "gp grid starting loop"        
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
            #print 'GPGRID diff = ' + str(diff)
            nIt += 1
            
#         fin = time.clock()
#         print "train time: " + str(fin-start)            
           
        self.partialK = self.G.dot(np.linalg.inv(self.G.dot(self.K).dot(self.G) + self.Q) );    
        self.obs_f = f.reshape(-1)
        self.obs_C = C
        v = np.diag(C)
        mPr_tr = sigmoid(self.obs_f, self.s)
        sdPr_tr = np.sqrt(target_var(self.obs_f, self.s, v))
        
        #print "gp grid trained"
        
        return self.obs_f, C, mPr_tr, sdPr_tr

    def post_peaks(self, f, C):     
        v = np.diag(C)
        stdPr = target_var(f,self.s,v)
        mPr = sigmoid(f,self.s)
          
        return mPr, stdPr
    
    def post_grid(self):
#         return self.post_grid_noloops()
        #return self.post_grid_loops()
        return self.post_grid_oneloop()
    
    def post_grid_oneloop(self):
        
        ddx = np.float64(-self.obsx).reshape((1,len(self.obsx)))
        ddy = np.tile(np.float64(-self.obsy), (self.ny,1))
        
        f_end = self.G.dot(self.z) 
        
        f = np.zeros((self.nx,self.ny), dtype=np.float64)
        C = np.ones((self.nx,self.ny), dtype=np.float64)

        j = np.arange(self.ny).reshape(self.ny,1)
        Ky = np.exp( -(j-ddy)**2/self.ls )
        
#         start = time.clock()
        
        for i in np.arange(self.nx):
            Kx = np.exp( -ddx**2/self.ls )
            Kpred = Kx*Ky
        
            W = Kpred.dot(self.partialK)
        
            f[i,:] += W.dot(f_end).reshape(-1)
            C[i,:] -= np.sum(W.dot(self.G)*Kpred, axis=1).reshape(-1)
            ddx += 1

        mPr = sigmoid(f,self.s)
        stdPr = np.sqrt(target_var(f, self.s, C))
        
#         fin = time.clock()
#         print "pred time: " + str(fin-start)
        
        return mPr, stdPr, f, C    
    
    def post_grid_loops(self):
        
        ddx = np.float64(-self.obsx)
        ddy = np.float64(-self.obsy)
        
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
                C[i,j] -= W.dot(self.G).dot(Kpred.T)
                
                ddy += 1
            ddx += 1
            ddy -= self.ny

        mPr = sigmoid(f,self.s)
        stdPr = np.sqrt(target_var(f, self.s, C))
        
        return mPr, stdPr, f, C
         
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
        
        return mPr, stdPr, f, C
