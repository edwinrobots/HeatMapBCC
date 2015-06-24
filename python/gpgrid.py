import numpy as np
from scipy.linalg import cholesky, solve_triangular, inv, det
from scipy.sparse import coo_matrix
from scipy.optimize import fmin_cobyla
from scipy.stats import gamma
from scipy.special import gammaln
from sklearn.gaussian_process import GaussianProcess
import scipy.linalg.fblas as fblas
import logging

def sigmoid(f):
    g = 1/(1+np.exp(-f))
    return g

def logit(g):
    f = -np.log(1/g - 1)
    return f

def target_var(f,s,v):
    mean = sigmoid(f,s)
    u = mean*(1-mean)
    v = v*s*s
    return u/(1/(u*v) + 1)

class GPGrid(object):
    verbose = False
    
    # hyperparameters
    nu0 = []# prior observations, used to determine the observation noise variance and the prior mean
    s = 1 # inverse output scale
    ls = 100 # inner length scale of the GP
    
    # parameters for the hyperpriors if we want to optimise the hyperparameters
    gam_shape_ls = 1
    gam_scale_ls = 0
    gam_shape_s0 = 1 
    gam_scale_s0 = 1       
    gam_shape_nu = 200
    gam_scale_nu = []

    # save the training points
    obsx = []
    obsy = []
    obs_values = [] # value of the positive class at each observations. Any duplicate points will be summed.
    grid_obs_counts = []
    
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
            
    def __init__(self, nx, ny, ls=100, nu0=[1,1], gam_shape_ls=1, force_update_all_points=False, implementation="native"):
        self.nx = nx
        self.ny = ny  
        nu0 = np.array(nu0).astype(float)
        self.nu0 = nu0
        self.gam_shape_s0 = (np.sum(nu0) - 2)/2.0 + 0.001 # pseudo counts
        self.gam_scale_s0 = self.gam_shape_s0 * 1000000 # (nu0[1]/np.sum(nu0)) * (1-nu0[1]/np.sum(nu0)) *
        self.s = self.gam_shape_s0/self.gam_scale_s0 #sigma scaling
        self.ls = ls #length-scale
        self.gam_shape_ls = gam_shape_ls # the scale of the gamma prior of the lengthscale will be calculated from the initial value self.ls
        self.update_all_points = force_update_all_points
        self.implementation = implementation
        
        self.latentpriormean = logit(self.nu0[1] / np.sum(self.nu0))
    
    def process_observations(self, obsx, obsy, obs_values): 
        if obs_values==[]:
            return [],[]        
        
        if self.z!=[] and obsx==self.rawobsx and obsy==self.rawobsy and obs_values==self.rawobs_points:
            return
        
        if self.verbose:
            logging.debug("GP grid fitting " + str(len(self.obsx)) + " observations.")
            
        self.obsx = np.array(obsx)
        self.obsy = np.array(obsy)
#         self.obs_flat_idxs = np.ravel_multi_index((self.obsx, self.obsy), (self.nx,self.ny))
        
        obs_values = np.array(obs_values)
        if obs_values.ndim==1 or obs_values.shape[1]==1: # duplicate locations allowed, one count at each array entry
            self.obs_values = np.array(obs_values).reshape(-1)

            self.grid_obs_counts = coo_matrix((np.ones(len(self.obsx)), (self.obsx, self.obsy)), shape=(self.nx,self.ny)).toarray()            
            grid_obs_pos_counts = coo_matrix((self.obs_values, (self.obsx, self.obsy)), shape=(self.nx,self.ny)).toarray()
            
            self.obsx, self.obsy = self.grid_obs_counts.nonzero()
            obs_pos_counts = grid_obs_pos_counts[self.obsx,self.obsy]
            obs_total_counts = self.grid_obs_counts[self.obsx,self.obsy]
            
        elif obs_values.shape[1]==2: 
            # obs_values given as two columns: first is positive counts, second is total counts. No duplicate locations
            obs_pos_counts = np.array(obs_values[:,0]).reshape(obs_values.shape[0],1)
            self.obs_values = obs_pos_counts
            obs_total_counts = np.array(obs_values[:,1]).reshape(obs_values.shape[0],1)
        
        self.obs_total_counts = obs_total_counts
        
        #Difference between observed value and prior mean
        # Is this right at this point? Should it be moderated by prior pseudo-counts? Because we are treating this as a noisy observation of kappa.
        obs_probs = obs_pos_counts/obs_total_counts[:, np.newaxis].reshape((obs_pos_counts.size,1))
        self.z = obs_probs#- (self.nu0[1]/np.sum(self.nu0))
        
        #Update to produce training matrices only over known points
        obsx_tiled = np.tile(self.obsx, (len(self.obsx),1))
        obsy_tiled = np.tile(self.obsy, (len(self.obsy),1))
        self.obs_ddx = np.array(obsx_tiled.T - obsx_tiled, dtype=np.float64)
        self.obs_ddy = np.array(obsy_tiled.T - obsy_tiled, dtype=np.float64)
    
        # Mean probability at observed points given local observations
        self.obs_mean_prob = (obs_pos_counts+1) / (obs_total_counts+2)
        # Noise in observations
        self.Q = np.diagflat(self.obs_mean_prob * (1 - self.obs_mean_prob) / obs_total_counts)
    
    def ln_modelprior(self):
        #Check and initialise the hyper-hyper-parameters if necessary
        if self.gam_scale_ls==0:
            self.gam_scale_ls = self.ls / float(self.gam_shape_ls)
        #Gamma distribution over each value. Set the parameters of the gammas.
        lnp_gp = - gammaln(self.gam_shape_ls) - self.gam_shape_ls*np.log(self.gam_scale_ls) \
                   + (self.gam_shape_ls-1)*np.log(self.ls) - self.ls/self.gam_scale_ls
        return lnp_gp
    
    def lowerbound(self):
        #calculate likelihood from the fitted model 
        v = np.diag(self.obs_C)[:, np.newaxis]
        k = (1+(np.pi*v/8.0))**(-0.5)
        mean_prob_obs = sigmoid(self.obs_f) + k
        if not hasattr(self, 'logdetQ'):
            self.logdetQ = np.sum(np.log(np.diag(cholesky(self.Q, lower=True, overwrite_a=False, check_finite=False))))*2.0
            self.Qinv = np.linalg.pinv(self.Q) # Q is diagonal so inverse is trivial
        data_loglikelihood = np.sum( self.obs_values * np.log(sigmoid(self.obs_f)) + \
                                (self.obs_total_counts-self.obs_values)*np.log(sigmoid(-self.obs_f)) )
        #-0.5*(len(self.obs_f)*np.log(2*np.pi) + self.logdetQ + \
        #   (self.z - mean_prob_obs).T.dot(self.Qinv).dot(self.z - mean_prob_obs) ) #+ \
        #   np.trace(self.G.dot(self.Qinv).dot(self.G).dot(self.obs_C)) ) #this trace term cancels with a term in pf_minus_qf

        cholKs = self.cholK / np.sqrt(self.old_s)
                
        logdetC = np.sum(np.log(np.diag(cholesky(self.obs_C, lower=True, overwrite_a=False, check_finite=False))))*2.0
        logdetK = np.sum(np.log(np.diag(cholKs)))*2.0
        invK_f_fT = solve_triangular(cholKs, solve_triangular(cholKs.T, self.obs_f.dot(self.obs_f.T) + self.obs_C, \
                                                               lower=True), lower=False, overwrite_b=True)
        # cancels with term in data loglikelihood
                                                               
        #invK_f_fT_original = inv(self.Ks).dot(self.obs_f.dot(self.obs_f.T) + self.obs_C) 
        pf_minus_qf = 0.5*(logdetC - logdetK - np.trace(invK_f_fT) + len(self.obs_f) )# cancels with term in data loglikelihood
        
        ps_minus_qs = gammaln(self.gam_shape_s)-gammaln(self.gam_shape_s0) \
                        + self.gam_shape_s0*np.log(self.gam_scale_s0)\
                        - self.gam_shape_s*np.log(self.gam_scale_s) + (self.gam_shape_s0-self.gam_shape_s)*np.log(self.s)\
                        + (self.gam_scale_s-self.gam_scale_s0)*(self.s)
        
#         print "data " + str(data_loglikelihood)
#         print "ps_minus_qs " + str(ps_minus_qs)
#         print "pf " + str(pf_minus_qf)
                
        lb = data_loglikelihood + pf_minus_qf + ps_minus_qs
        # if using steinberg/bonilla this value usually goes up even if lb goes down due to changing C estimate
        print "least squares minimisation: " + str(lb - 0.5*logdetC)
        
        return lb
    
    def neg_marginal_likelihood(self, hyperparams, expectedlog=False):
        '''
        Weight the marginal log data likelihood by the hyper-prior. Unnormalised posterior over the hyper-parameters.
        '''
        if np.any(np.isnan(hyperparams)) or np.any(hyperparams <= 0):
            return np.inf
        self.ls = hyperparams[0]
        logging.debug("Length-scale: %f" % self.ls)
        self.obs_f = [] # make sure we start again
        self.fit((self.obsx, self.obsy), self.obs_values, expectedlog=expectedlog, process_obs=False)
        
        marginal_log_likelihood = self.lowerbound()
        
        log_model_prior = self.ln_modelprior()
        lml = marginal_log_likelihood + log_model_prior
        logging.debug("Log joint probability of the model & data: %f" % lml)
        return -lml #returns Negative!
    
    def optimize(self, obs_coords, obs_values, expectedlog=True):
        obsx = obs_coords[0]
        obsy = obs_coords[1]
        self.process_observations(obsx, obsy, obs_values) # process the data here so we don't repeat each call
        initialguess = [self.ls]
        constraints = [lambda hp,_: np.all(hp>0)]
        opt_hyperparams = fmin_cobyla(self.neg_marginal_likelihood, initialguess, constraints, args=(expectedlog,), 
                                    rhobeg=100, rhoend=10)
        logging.debug("Optimal hyper-parameters: ")
        for param in opt_hyperparams:
            logging.debug(str(param))        
        return self.mean_prob_obs, opt_hyperparams
    
    def fit( self, obs_coords, obs_values, expectedlog=False, process_obs=True):
        obsx = obs_coords[0]
        obsy = obs_coords[1]
        if process_obs:
            self.process_observations(obsx, obsy, obs_values)
        # get the correct covariance matrix
        Kx = np.exp( -self.obs_ddx**2/self.ls )
        Ky = np.exp( -self.obs_ddy**2/self.ls )
        K = Kx*Ky
        self.K = K + 1e-6 * np.eye(len(K)) # jitter                
        self.cholK = cholesky(self.K, overwrite_a=False, check_finite=False)
        
        if self.obsx==[]:
            mPr = 0.5
            stdPr = 0.25       
            return mPr, stdPr     
        if self.verbose: 
            logging.debug("gp grid starting training...")    
        
        if self.implementation=="native":
            self.cholK = cholesky(self.K, overwrite_a=False, check_finite=False)
            self.gam_shape_s = self.gam_shape_s0 + len(self.obsx)/2.0
            
            if self.obs_f == [] or len(self.obs_f)<len(self.obsx):
                self.obs_f = logit(self.obs_mean_prob)#np.zeros((len(self.obsx), 1))
                self.gam_scale_s = self.gam_shape_s / self.s

            mean_X = self.obs_mean_prob

            stepsize = 0.36

            converged = False    
            nIt = 0
            lb = -np.inf
            oldlb = -np.inf   
            diff = 0    
            while not converged and nIt<100:
                
                if self.verbose:
                    logging.debug("Updated inverse output scale: " + str(self.s))
                
                old_f = self.obs_f
                                    
                self.Ks = self.K/self.s
                converged_inner = False
                nIt_inner = 0
                while not converged_inner and nIt_inner < 1:
                    self.G = np.diagflat( mean_X*(1-mean_X) ) 
            
                    Cov = self.G.dot(self.Ks).dot(self.G) + self.Q
                    L = cholesky(Cov,lower=True, check_finite=False, overwrite_a=True)  
            
                    zhat = 0.5
                    #zhat = mean_X - self.G.dot(self.obs_f)  # updates a la steinberg. Can find suboptimal values because
                    # it optimises mean in isolation, ignoring the fact that lower bound depends also, on covariance,
                    # which is approximated using a function of the mean (self.G). The alternative used above fits to 
                    # the data using an iterative EM-like process where the goal is consistency between self.G and the 
                    # mean, not minimising the squared differences.
                    #zhat = mean_X # use this if we want to do a newtonian method with the hessian.
                    B = solve_triangular(L, (self.z - zhat), lower=True, overwrite_b=True)
                    A = solve_triangular(L.T, B, overwrite_b=True)

                    #H = np.diagflat(1-2*mean_X) + 1e-6 * np.eye(len(K)) # jitter 
                    #self.obs_f = old_f*(1-stepsize) + stepsize*inv(H).dot(B) # uses second derivative -- better for least-squares minimisation but doesn't improve lower bound
                    stepsize = np.diag(self.G)[:, np.newaxis] # good heuristic to achieve convergence: gets lower as values more extreme
                    self.obs_f = old_f*(1-stepsize) + stepsize*self.Ks.dot(self.G).dot(A)                 
                    #self.obs_f = self.Ks.dot(self.G).dot(inv(self.Q + self.G.dot(self.Ks).dot(self.G))).dot(self.G).dot(self.z-0.5)
                    mean_X = sigmoid(self.obs_f)
                          
                    diff = np.max(np.abs(self.obs_f - old_f))
                    if self.verbose:
                        logging.debug('GPGRID diff = ' + str(diff) + ", iteration " + str(nIt))
                        
                    nIt_inner += 1
                    if diff < 1e-2:
                        converged_inner = True
                        
                nIt += 1
                V = solve_triangular(L, self.G.dot(self.Ks.T), lower=True, overwrite_b=True)
                self.obs_C = self.Ks - V.T.dot(V) 
                #self.obs_C = self.Ks - self.Ks.dot(self.G).dot(inv(self.Q+self.G.dot(self.Ks).dot(self.G))).dot(self.G).dot(self.Ks)
  
                #update the scale parameter of the output scale distribution (also called latent function scale/sigmoid steepness)
                invK_C_plus_ffT = solve_triangular(self.cholK.T, (self.obs_C + self.obs_f.dot(self.obs_f.T)), lower=True, overwrite_b=True)
                invK_C_plus_ffT = solve_triangular(self.cholK, invK_C_plus_ffT, overwrite_b=True)
                #D_old = inv(self.K).dot(self.obs_f.dot(self.obs_f.T) + self.obs_C)
                self.gam_scale_s = float(self.gam_scale_s0) + 0.5*np.trace(invK_C_plus_ffT) 
                #update s to its current expected value
                self.old_s = self.s # Approximations for Binary Gaussian Process Classification, Hannes Nickisch
                self.s = self.gam_shape_s/self.gam_scale_s
                
                #self.M = cholesky(self.Ks, lower=True, overwrite_a=False, check_finite=False)
                diff_s = np.abs(self.s-self.old_s)
                
                oldlb = lb
                lb = self.lowerbound()

                if self.verbose:
                    print lb
                difflb = np.abs(lb-oldlb)
                                
                converged = (diff_s<1e-3) & (diff<1e-3) & (difflb<1)
                if self.verbose:                  
                    print "gam scale s: " + str(self.gam_scale_s)
                            
        elif self.implementation=="sklearn":
            obs_noise = np.diag(self.Q)/((self.z-0.5).reshape(-1)**2)
            self.gp = GaussianProcess(theta0=1.0/self.ls, nugget=obs_noise )
            #fit
            X = np.concatenate((self.obsx[:,np.newaxis], self.obsy[:,np.newaxis]), axis=1).astype(float)
            self.gp.fit(X, self.z-0.5)
            #predict
            self.obs_f, v = self.gp.predict(X, eval_MSE=True)
        if self.verbose:
            logging.debug("gp grid trained")
        self.obs_f = self.obs_f + self.latentpriormean
        if expectedlog:
            mean_prob_obs = np.log(sigmoid(self.obs_f))
            mean_prob_notobs = np.log(sigmoid(-self.obs_f))
        else:
            k = (1+(np.pi*v/8.0))**(-0.5)    
            mean_prob_obs = sigmoid(k*self.obs_f)
            mean_prob_notobs = sigmoid(k*-self.obs_f)
            
        self.mean_prob_obs = mean_prob_obs
        self.mean_prob_notobs = mean_prob_notobs
        return mean_prob_obs
    
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
            
            #if not update_all_points: # how can we do this if we don't always have grid outputs and inputs?
            #    changed_obs = np.abs(self.obs_f - self.f[self.obs_flat_idxs]) > 0.05
             
            Cov = self.G.dot(self.Ks).dot(self.G) + self.Q
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
#                 if not update_all_points:
#                     changed = np.argwhere(np.sum(Kpred[:,changed_obs],axis=1)>0.1)
#                     Kpred = Kpred[changed,:]
#                     changed_s = changed + start
#                el...
                if end>=0:
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
                
        f = self.f + self.latentpriormean
                
        # Approximate the expected value of the variable transformed through the sigmoid.
        if expectedlog:
            m_post = np.log(sigmoid(f))
        else:
            k = (1+(np.pi*self.v/8.0))**(-0.5)
            m_post = sigmoid(k*f)
        if self.verbose:
            logging.debug("gp grid predictions: %s" % str(m_post))
             
        return m_post
