import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.sparse import coo_matrix
from scipy.optimize import fmin_cobyla
from scipy.stats import gamma
from sklearn.gaussian_process import GaussianProcess
import scipy.linalg.fblas as fblas
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
    # hyperparameters
    nu0 = []# prior observations, used to determine the observation noise variance and the prior mean
    s = 4 # sigmoid scaling parameter
    ls = 100 # inner length scale of the GP
    
    # parameters for the hyperpriors if we want to optimise the hyperparameters
    gam_shape_ls = 100
    gam_shape_s = 4 
    gam_scale_ls = 0
    gam_scale_s = 0   
    gam_shape_nu = 200
    gam_scale_nu = []

    # save the training points
    obsx = []
    obsy = []
    obs_values = [] # value of the positive class at each observations. Any duplicate points will be summed.
    grid_all = []
    
    obs_f = []
    obs_C = []
    
    nx = 0
    ny = 0
        
    G = []
    partialK = []
    z = []
      
    gp = []
    
    K = []
    Q = []
    
    f = []
    v = []
    
    implementation="sklearn" # use the sklearn implementation or "native" to use the version implemented here 
            
    def __init__(self, nx, ny, s=4, ls=100, nu0=[1,1], force_update_all_points=False, implementation="native"):
        self.nx = nx
        self.ny = ny  
        self.s = s #sigma scaling
        self.ls = ls #length-scale
        self.update_all_points = force_update_all_points
        self.implementation = implementation
        self.nu0 = np.array(nu0, dtype=float)
        self.latentpriormean = logit(self.nu0[1] / np.sum(self.nu0), self.s)
    
    def process_observations(self, obsx, obsy, obs_values): 
        if obs_values==[]:
            return [],[]        
        
        if self.z!=[] and obsx==self.rawobsx and obsy==self.rawobsy and obs_values==self.rawobs_points:
            return
        
        logging.debug("GP grid processing " + str(len(self.obsx)) + " observations.")
            
        self.obsx = np.array(obsx)
        self.obsy = np.array(obsy)
        self.obs_flat_idxs = np.ravel_multi_index((self.obsx, self.obsy), (self.nx,self.ny))
        
        obs_values = np.array(obs_values)
        if obs_values.ndim==1 or obs_values.shape[1]==1:
            self.obs_values = np.array(obs_values).reshape(-1)

            self.grid_all = coo_matrix((np.ones(len(self.obsx)), (self.obsx, self.obsy)), shape=(self.nx,self.ny)).toarray()            
            grid_p = coo_matrix((self.obs_values, (self.obsx, self.obsy)), shape=(self.nx,self.ny)).toarray()
            
            self.obsx, self.obsy = self.grid_all.nonzero()
            presp = grid_p[self.obsx,self.obsy]
            allresp = self.grid_all[self.obsx,self.obsy]
            
        elif obs_values.shape[1]==2:
            presp = np.array(obs_values[:,0]).reshape(obs_values.shape[0],1)
            self.obs_values = presp
            allresp = np.array(obs_values[:,1]).reshape(obs_values.shape[0],1)
        
        #add on the prior counts
        self.z = presp/allresp - self.nu0[1] / np.sum(self.nu0) # subtract the prior mean
        self.z = self.z.reshape((self.z.size,1)) 
        
        #Update to produce training matrices only over known points
        obsx_tiled = np.tile(self.obsx, (len(self.obsx),1))
        obsy_tiled = np.tile(self.obsy, (len(self.obsy),1))
        ddx = np.array(obsx_tiled.T - obsx_tiled, dtype=np.float64)
        ddy = np.array(obsy_tiled.T - obsy_tiled, dtype=np.float64)
    
        Kx = np.exp( -ddx**2/self.ls )
        Ky = np.exp( -ddy**2/self.ls )
        K = Kx*Ky
        self.K = K + 1e-6 * np.eye(len(K)) # jitter 
    
        Pr_est = (presp+self.nu0[1]) / (allresp+np.sum(self.nu0))
        self.Q = np.diagflat(Pr_est*(1-Pr_est)/(allresp+np.sum(self.nu0)+1.0))
    
    def ln_modelprior(self):
        #Check and initialise the hyper-hyper-parameters if necessary
        if self.gam_scale_ls==0:
            self.gam_scale_ls = self.ls / float(self.gam_shape_ls)
        if self.gam_scale_s==0:
            self.gam_scale_s = self.s / float(self.gam_shape_s)
        if self.gam_scale_nu == []:
            self.gam_scale_nu = self.nu0 / float(self.gam_shape_nu)
        #Gamma distribution over each value. Set the parameters of the gammas.
        lnp_gp = gamma.logpdf(self.ls, self.gam_shape_ls, scale=self.gam_scale_ls) + \
                    gamma.logpdf(self.s, self.gam_shape_s, scale=self.gam_scale_s) + \
                    np.sum(gamma.logpdf(self.nu0, self.gam_shape_nu, scale=self.gam_scale_nu))
        return lnp_gp
    
    def neg_marginal_likelihood(self, hyperparams):
        '''
        Weight the marginal log data likelihood by the hyper-prior. Unnormalised posterior over the hyper-parameters.
        '''
        if np.any(np.isnan(hyperparams)) or np.any(hyperparams <= 0):
            return np.inf
        self.s = hyperparams[0]
        self.ls = hyperparams[1]
        self.nu0 = hyperparams[2:]
        mPr_tr = self.fit((self.obsx, self.obsy), self.obs_values, expectedlog=False)
        
        #calculate likelihood from the fitted model
        data_loglikelihood = np.sum(np.log(mPr_tr))
        log_model_prior = self.ln_modelprior()
        lml = data_loglikelihood + log_model_prior
        logging.debug("Log joint probability of the model & data: %f" % lml)
        return -lml #returns Negative!
    
    def optimize(self, obs_coords, obs_values):
        obsx = obs_coords[0]
        obsy = obs_coords[1]
        self.process_observations(obsx, obsy, obs_values)
        initialguess = np.concatenate(([self.s, self.ls], self.nu0))
        constraints = [lambda hp: np.all(hp)]
        opt_hyperparams = fmin_cobyla(self.neg_marginal_likelihood, initialguess, constraints)
        logging.debug("Optimal hyper-parameters: ")
        for param in opt_hyperparams:
            logging.debug(str(param))        
        return opt_hyperparams
    
    def fit( self, obs_coords, obs_values, expectedlog=False):
        obsx = obs_coords[0]
        obsy = obs_coords[1]
        self.process_observations(obsx, obsy, obs_values)
        if self.obsx==[]:
            mPr = 0.5
            stdPr = 0.25       
            return mPr, stdPr      
        logging.debug("gp grid starting training...")    
        
        if self.implementation=="native":
            if self.obs_f == []:
                f = np.zeros(len(self.obsx))
            else:
                f = self.obs_f
            converged = False    
            nIt = 0
            while not converged and nIt<100:
                old_f = f
            
                mean_X = sigmoid(f,self.s)
                self.G = np.diagflat( self.s*mean_X*(1-mean_X) )
            
                Cov = self.G.dot(self.K).dot(self.G) + self.Q
                L = cholesky(Cov,lower=True, check_finite=False, overwrite_a=True) # inv(Cov, overwrite_a=True, check_finite=False) 

                #W = self.K.dot(self.G).dot( inv(Cov) )
                #f = W.dot(self.G).dot(self.z)
                B = solve_triangular(L, self.G.dot(self.z), lower=True, overwrite_b=True)
                A = solve_triangular(L.T, B, lower=False, overwrite_b=True)

                f = self.K.dot(self.G).dot(A)             
                diff = np.max(np.abs(f-old_f))
                converged = diff<1e-3
                logging.debug('GPGRID diff = ' + str(diff))
                nIt += 1          
            V = solve_triangular(L, self.G.dot(self.K.T), lower=True)
            C = self.K - V.T.dot(V)       
            #C = self.K - W.dot(self.G).dot(self.K)     
            self.obs_C = C
            self.A = A
            v = np.diag(C)
            if np.argwhere(v<0).size != 0:
                logging.warning("Variance was negative in GPGrid fit()")
                v[v<0] = 0            
        elif self.implementation=="sklearn":
            obs_noise = np.diag(self.Q)/(self.z.reshape(-1)**2)
            self.gp = GaussianProcess(theta0=1.0/self.ls, nugget=obs_noise )
            #fit
            X = np.concatenate((self.obsx[:,np.newaxis], self.obsy[:,np.newaxis]), axis=1).astype(float)
            self.gp.fit(X, self.z)
            #predict
            f, v = self.gp.predict(X, eval_MSE=True)
        
        logging.debug("gp grid trained")
        self.obs_f = f.reshape(-1) + self.latentpriormean
        if expectedlog:
            mPr_tr = np.log(sigmoid(self.obs_f, self.s))
        else:
            k = (1+(np.pi*v/8.0))**(-0.5)    
            mPr_tr = sigmoid(k*self.obs_f, self.s)
        return mPr_tr
    
    def predict(self, output_coords, expectedlog=False):
        '''
        Evaluate the function posterior mean and variance at the given co-ordinates using the 2D squared exponential 
        kernel
        '''
        outputx = output_coords[0].astype(float)
        outputy = output_coords[1].astype(float)
        maxsize = 2.0 * 10**7
        nout = outputx.size

        if self.implementation=="native":
            nobs = len(self.obsx)
            nsplits = np.ceil(nout*nobs / maxsize)
            splitsize = int(np.ceil(nout/nsplits)) # make sure we don't kill memory
    
            # Determine whether to calculate all points from scratch, or just those close to new/changed observations
            update_all_points = self.update_all_points
            if self.f==[]:
                self.f = np.zeros(nout)
                update_all_points = True
                self.v = np.zeros(nout) # diagonal values only
            
            if not update_all_points:
                changed_obs = np.abs(self.obs_f - self.f[self.obs_flat_idxs]) > 0.05
             
            Cov = self.G.dot(self.K).dot(self.G) + self.Q
            L = cholesky(Cov,lower=True, check_finite=False, overwrite_a=True)
            Vpart = solve_triangular(L, self.G, lower=True, check_finite=False, overwrite_b=True)
            GA = self.G.dot(self.A)      
            
            for s in np.arange(nsplits):
                
                logging.debug("Computing posterior for split %i out of %i" % (s,nsplits))
                
                start = int(s*splitsize)            
                end = int(start+splitsize) if start+splitsize<=nout else -1
                outputx_s = outputx[start:end].reshape(end-start,1)
                outputy_s = outputy[start:end].reshape(end-start,1)
                
                ddx = outputx_s - self.obsx
                ddy = outputy_s - self.obsy
                Ky = np.exp( -ddy**2/self.ls )
                Kx = np.exp( -ddx**2/self.ls )
                Kpred = Kx*Ky
                
                #Kpred[Kpred<1e-10] = 0
                
                #update all idxs?
                if not update_all_points:
                    changed = np.argwhere(np.sum(Kpred[:,changed_obs],axis=1)>0.1)
                    Kpred = Kpred[changed,:]
                    changed_s = changed + start
                elif end>=0:
                    changed_s = np.arange(start,end)
                else:
                    changed_s = np.arange(start,nout)  
                    
                f_s = fblas.dgemm(alpha=1.0, a=Kpred.T, b=GA.T, trans_a=True, trans_b=True, overwrite_c=True)
                V = fblas.dgemm(alpha=1.0, a=Vpart.T, b=Kpred.T, trans_a=True, overwrite_c=True)
                v_s = 1.0 - np.sum(V**2,axis=0)
       
                self.f[changed_s] = f_s
                self.v[changed_s] = v_s
        elif self.implementation=="sklearn":
            #predict
            X = np.concatenate((outputx.reshape(nout,1), outputy.reshape(nout,1)), axis=1)
            self.f, self.v = self.gp.predict(X, eval_MSE=True, batch_size=maxsize)
                
        self.f += self.latentpriormean
                
        # Approximate the expected value of the variable transformed through the sigmoid.
        if expectedlog:
            m_post = np.log(sigmoid(self.f,self.s))
        else:
            k = (1+(np.pi*self.v/8.0))**(-0.5)
            m_post = sigmoid(k*self.f,self.s)
        logging.debug("gp grid predictions: %s" % str(m_post))
             
        return m_post
