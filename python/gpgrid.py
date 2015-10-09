import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.sparse import coo_matrix
from scipy.optimize import fmin#_cobyla
from scipy.special import gammaln, gamma
import scipy.linalg.blas as fblas
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
    
    # hyper--parameters
    s = 1 # inverse output scale
    ls = 100 # inner length scale of the GP
    mu0 = 0
    z0 = 0.5
    # parameters for the hyper-priors if we want to optimise the hyper-parameters
    shape_ls = 1
    rate_ls = 0
    shape_s0 = 1 
    rate_s0 = 1       

    # save the training points
    obsx = []
    obsy = []
    obs_values = [] # value of the positive class at each observations. Any duplicate points will be summed.
    
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
    
    max_iter_VB = 500
    max_iter_G = 2
    conv_threshold = 0.1

    cov_func = "sqexp"# "matern" #

    outputx = []
    outputy = []
    
    def __init__(self, nx, ny, z0=0.5, shape_s0=1, rate_s0=10.0, shape_ls=10, rate_ls=0.1, ls_initial=None, force_update_all_points=False):
        #Grid size for prediction
        self.nx = nx
        self.ny = ny
        
        # Prior mean
        self.z0 = z0
        self.mu0 = logit(z0)
        
        # Output scale (sigmoid scaling)  
        self.shape_s0 = shape_s0 # prior pseudo counts * 0.5
        self.rate_s0 = rate_s0 # prior sum of squared deviations
        self.s = self.shape_s0/self.rate_s0 
        
        #Length-scale
        self.shape_ls = shape_ls # prior pseudo counts * 0.5
        self.rate_ls = rate_ls # analogous to a sum of changes between points at a distance of 1
        if np.any(ls_initial):
            self.ls = ls_initial
        else:
            self.ls = self.shape_ls / self.rate_ls
            self.ls = np.zeros(2) + self.ls

        #Algorithm configuration        
        self.update_all_points = force_update_all_points
    
    def process_observations(self, obsx, obsy, obs_values): 
        if obs_values==[]:
            return [],[]        
        
        if self.z!=[] and obsx==self.rawobsx and obsy==self.rawobsy and obs_values==self.rawobs_points:
            return
        
        if self.verbose:
            logging.debug("GP grid fitting " + str(len(self.obsx)) + " observations.")
            
        self.obsx = np.array(obsx)
        self.obsy = np.array(obsy)
        
        obs_values = np.array(obs_values)
        if self.obsx.dtype=='int' and (obs_values.ndim==1 or obs_values.shape[1]==1): # duplicate locations allowed, one count at each array entry
            grid_obs_counts = coo_matrix((np.ones(len(self.obsx)), (self.obsx, self.obsy)), shape=(self.nx,self.ny)).toarray()            
            grid_obs_pos_counts = coo_matrix((obs_values, (self.obsx, self.obsy)), shape=(self.nx,self.ny)).toarray()
            
            self.obsx, self.obsy = grid_obs_counts.nonzero()
            self.obs_values = grid_obs_pos_counts[self.obsx,self.obsy]
            obs_pos_counts = self.obs_values[:, np.newaxis]
            obs_total_counts = grid_obs_counts[self.obsx,self.obsy][:, np.newaxis]
            
        elif self.obsx.dtype=='float' and (obs_values.ndim==1 or obs_values.shape[1]==1):
            # assume no clashes
#             counters = {}
#             pos_counters = {}
#             
#             self.obsx = []
#             self.obsy = []
#             
#             for i in range(len(obsx)):
#                 coord = (obsx[i], obsy[i])
#                 if coord in counters:
#                     counters[coord] += 1.0
#                     pos_counters[coord] += obs_values[i]
#                 else:
#                     counters[coord] = 1.0
#                     pos_counters[coord] = obs_values[i]
#                     
#                     self.obsx.append(obsx[i])
#                     self.obsy.append(obsy[i])
#             
#             self.obs_values = np.array(pos_counters.values())
            self.obs_values = obs_values
            obs_pos_counts = self.obs_values[:, np.newaxis]
            #obs_total_counts = np.array(counters.values())[:, np.newaxis]
            obs_total_counts = np.ones(obs_pos_counts.shape)
             
        elif obs_values.shape[1]==2: 
            # obs_values given as two columns: first is positive counts, second is total counts. No duplicate locations
            obs_pos_counts = np.array(obs_values[:,0]).reshape(obs_values.shape[0],1)
            self.obs_values = obs_pos_counts
            obs_total_counts = np.array(obs_values[:,1]).reshape(obs_values.shape[0],1)
        
        self.obs_total_counts = obs_total_counts
        
        #Difference between observed value and prior mean
        # Is this right at this point? Should it be moderated by prior pseudo-counts? Because we are treating this as a noisy observation of kappa.
        obs_probs = obs_pos_counts/obs_total_counts
        self.z = obs_probs
        
        #Update to produce training matrices only over known points
        obsx_tiled = np.tile(self.obsx, (len(self.obsx),1))
        obsy_tiled = np.tile(self.obsy, (len(self.obsy),1))
        self.obs_ddx = np.array(obsx_tiled.T - obsx_tiled, dtype=np.float64)
        self.obs_ddy = np.array(obsy_tiled.T - obsy_tiled, dtype=np.float64)
    
        # Mean probability at observed points given local observations
        self.obs_mean_prob = (obs_pos_counts+1) / (obs_total_counts+2)
        # Noise in observations
        self.Q = np.diagflat(self.obs_mean_prob * (1 - self.obs_mean_prob) / (obs_total_counts + 1))
    
    def ln_modelprior(self):
        #Gamma distribution over each value. Set the parameters of the gammas.
        lnp_gp = - gammaln(self.shape_ls) + self.shape_ls*np.log(self.rate_ls) \
                   + (self.shape_ls-1)*np.log(self.ls) - self.ls*self.rate_ls
        return np.sum(lnp_gp)
    
    def lowerbound(self):
        #calculate likelihood from the fitted model 
        if not hasattr(self, 'logdetQ'):
            self.logdetQ = np.sum(np.log(np.diag(cholesky(self.Q, lower=True, overwrite_a=False, check_finite=False))))*2.0
            self.Qinv = np.linalg.pinv(self.Q) # Q is diagonal so inverse is trivial
        data_loglikelihood = np.sum( self.obs_values * np.log(sigmoid(self.obs_f)) + \
                                (self.obs_total_counts-self.obs_values)*np.log(sigmoid(-self.obs_f)) )
        
        logp_minus_logq = self.logp_minus_logq()
        
        lb = data_loglikelihood + logp_minus_logq
        return lb

    def logp(self):
        cholKs = self.cholK * np.sqrt(self.old_s) # is this correct? Previously had a / sign instead of a *
        logdetK = np.sum(np.log(np.diag(cholKs)))*2.0
        invK_f_fT = solve_triangular(cholKs, solve_triangular(cholKs.T, self.obs_f.dot(self.obs_f.T) + self.obs_C, \
                                                               lower=True), lower=False, overwrite_b=True)

        prob_s = -gammaln(self.shape_s0) + self.shape_s0*np.log(self.rate_s0) + self.shape_s0 * np.log(self.s) - self.rate_s0 * self.s     
        return prob_s + 0.5 * -(np.trace(invK_f_fT) + logdetK)
        
    def logq(self):
        _, logdetC = np.linalg.slogdet(self.obs_C)#np.sum(np.log(np.diag(cholesky(self.obs_C, lower=True, overwrite_a=False, check_finite=False))))*2.0
        
        q_s = -gammaln(self.shape_s) + self.shape_s*np.log(self.rate_s) + self.shape_s * np.log(self.s) - self.rate_s * self.s
        
        return q_s + 0.5 * -(len(self.obs_f) + logdetC) 
        
    def logp_minus_logq(self):     
        return self.logp() - self.logq()        
    
    def neg_marginal_likelihood(self, hyperparams, expectedlog=False):
        '''
        Weight the marginal log data likelihood by the hyper-prior. Unnormalised posterior over the hyper-parameters.
        '''
        if np.any(np.isnan(hyperparams)):
            return np.inf
        self.ls = np.exp(hyperparams[0:2])
        if np.any(np.isinf(self.ls)):
            return np.inf
        logging.debug("Length-scale: %f, %f" % (self.ls[0], self.ls[1]))
        self.obs_f = [] # make sure we start again
        #original_s = self.s
        self.fit((self.obsx, self.obsy), self.obs_values, expectedlog=expectedlog, process_obs=False, update_s=False)
        self.fit((self.obsx, self.obsy), self.obs_values, expectedlog=expectedlog, process_obs=False, update_s=True)
        logging.debug("Inverse output scale: %f" % self.s)        
        #self.opt_s = self.s # saved the optimized value
        #self.s = original_s
        
        marginal_log_likelihood = self.lowerbound()
        
        log_model_prior = self.ln_modelprior()
        lml = marginal_log_likelihood + log_model_prior
        logging.debug("Log marginal likelihood: %f, joint probability of the model & data: %f" % (marginal_log_likelihood, lml))
        return -lml #returns Negative!
    
    def optimize(self, obs_coords, obs_values, expectedlog=True):
        obsx = obs_coords[0]
        obsy = obs_coords[1]
        self.process_observations(obsx, obsy, obs_values) # process the data here so we don't repeat each call
        initialguess = np.log(self.ls)
        #constraints = []#[lambda hp,_: 1 if np.all(np.asarray(hp)>0) else -1]
        opt_hyperparams = fmin(self.neg_marginal_likelihood, initialguess, maxfun=20, full_output=False, ftol=1, xtol=100)
        #fmin_cobyla(self.neg_marginal_likelihood, initialguess, constraints, args=(expectedlog,),
        #                            rhobeg=0.5, rhoend=0.2)
        opt_hyperparams[0] = np.exp(opt_hyperparams[0])
        #self.s = self.opt_s # use the optimal value since s would be reset to the original value otherwise
        logging.debug("Optimal hyper-parameters: ")
        for param in opt_hyperparams:
            logging.debug(str(param))        
        return self.mean_prob_obs, opt_hyperparams

    def sq_exp_cov(self, xvals, yvals):
        Kx = np.exp( -xvals**2 / self.ls[0] )
        Ky = np.exp( -yvals**2 / self.ls[1] )
        K = Kx * Ky
        return K

#     def matern(self, xvals, yvals):
#         nu = 1
#         rho = 1
# 
#         #1.0/(gamma(nu) * 2**(nu-1)) *
#         return K

    def fit( self, obs_coords, obs_values, expectedlog=False, process_obs=True, update_s=True):
        obsx = obs_coords[0]
        obsy = obs_coords[1]
        if process_obs:
            self.process_observations(obsx, obsy, obs_values)
        # get the correct covariance matrix
        if self.cov_func == "sqexp":
            self.K = self.sq_exp_cov(self.obs_ddx, self.obs_ddy)
        elif self.cov_func == "matern":
            self.K = self.matern(self.obs_ddx, self.obs_ddy)
        self.K = self.K + 1e-6 * np.eye(len(self.K)) # jitter only when fitting
        self.cholK = cholesky(self.K, overwrite_a=False, check_finite=False)
        
        if self.obsx==[]:
            mPr = 0.5
            stdPr = 0.25       
            return mPr, stdPr     
        if self.verbose: 
            logging.debug("gp grid starting training with length-scales %f, %f..." % (self.ls[0], self.ls[1]) )
        
        self.shape_s = self.shape_s0 + len(self.obsx)/2.0
        
        if not np.any(self.obs_f) or len(self.obs_f)<len(self.obsx):
            prev_obs_f = logit(self.obs_mean_prob)
            self.obs_C = self.K / self.s
            mean_X = self.obs_mean_prob         
            self.G = np.diagflat( mean_X*(1-mean_X) )
            prev_mean_X = mean_X    
        else:
            prev_obs_f = self.obs_f - self.mu0
            prev_mean_X = (self.mean_prob_obs - self.z0)
            
        old_obs_f = prev_obs_f
        conv_count = 0    
        nIt = 0
        diff = 0
        while not conv_count>3 and nIt<self.max_iter_VB:
            converged_inner = False
            nIt_inner = 0
            
            #update the scale parameter of the output scale distribution (also called latent function scale/sigmoid steepness)
            if update_s and np.any(self.obs_f): 
                # only do this if the observations have been properly initialised, otherwise use first guess for self.s
                invK_C_plus_ffT = solve_triangular(self.cholK.T, self.obs_C + prev_obs_f.dot(prev_obs_f.T), lower=True, overwrite_b=True)
                invK_C_plus_ffT = solve_triangular(self.cholK, invK_C_plus_ffT, overwrite_b=True)
                #D_old = inv(self.K).dot(self.obs_f.dot(self.obs_f.T) + self.obs_C)
                self.rate_s = float(self.rate_s0) + 0.5 * np.trace(invK_C_plus_ffT) 
                #update s to its current expected value
                self.old_s = self.s # Approximations for Binary Gaussian Process Classification, Hannes Nickisch
                self.s = self.shape_s/self.rate_s            
                if self.verbose:
                    logging.debug("Updated inverse output scale: " + str(self.s))
            
            self.Ks = self.K/self.s
            
            while not converged_inner and nIt_inner < self.max_iter_G:
                Cov = self.G.dot(self.Ks).dot(self.G) + self.Q
                self.L = cholesky(Cov,lower=True, check_finite=False, overwrite_a=True)  
        
                B = solve_triangular(self.L, (self.z - self.z0), lower=True, overwrite_b=True)
                self.A = solve_triangular(self.L.T, B, overwrite_b=True)
        
                #H = np.diagflat(1-2*mean_X) + 1e-6 * np.eye(len(K)) # jitter 
                #self.obs_f = old_f*(1-Nlabel_increment) + Nlabel_increment*inv(H).dot(B) # uses second derivative -- better for least-squares minimisation but doesn't improve lower bound
                Nlabel_increment = 0.5#(np.diag(self.G)[:, np.newaxis])  * np.random.rand()# good heuristic to achieve convergence: gets lower as values more extreme
                self.obs_f = prev_obs_f*(1-Nlabel_increment) + Nlabel_increment*self.Ks.dot(self.G).dot(self.A)
                #self.obs_f = self.Ks.dot(self.G).dot(inv(self.Q + self.G.dot(self.Ks).dot(self.G))).dot(self.G).dot(self.z-0.5)
                mean_X = sigmoid(self.obs_f)
                self.G = np.diagflat( mean_X*(1-mean_X) )
                      
                diff = np.max(np.abs(mean_X - prev_mean_X))
                if self.verbose:
                    logging.debug('GPGRID diff = ' + str(diff) + ", iteration " + str(nIt))
                    
                nIt_inner += 1
                if diff < self.conv_threshold:
                    converged_inner = True

                prev_mean_X = mean_X
                prev_obs_f = self.obs_f

            nIt += 1
            V = solve_triangular(self.L, self.G.dot(self.Ks.T), lower=True, overwrite_b=True)
            self.obs_C = self.Ks - V.T.dot(V) 

            converged = (diff<self.conv_threshold) #& ((diff_s / old_diff_s <1.05) | (diff_s / old_diff_s < 0.95) | (diff_s<0.0001))
            if converged:
                conv_count += 1
            else:
                conv_count = 0
                           
        if self.verbose:
            logging.debug("gp grid trained with inverse output scale updates=%i" % update_s)

        self.obs_f = self.obs_f + self.mu0
        self.changed_obs = (np.abs(self.obs_f - old_obs_f) > 0.05).reshape(-1)
        if expectedlog:
            mean_prob_obs = np.log(sigmoid(self.obs_f))
            mean_prob_notobs = np.log(sigmoid(-self.obs_f))
        else:
            k = (1+(np.pi*np.diag(self.obs_C)/8.0))**(-0.5)    
            mean_prob_obs = sigmoid(k*self.obs_f)
            mean_prob_notobs = sigmoid(k*-self.obs_f)
            
        self.mean_prob_obs = mean_prob_obs
        self.mean_prob_notobs = mean_prob_notobs
        return mean_prob_obs.reshape(-1)
    
    def predict(self, output_coords, expectedlog=False, variance_method='rough'):
        '''
        Evaluate the function posterior mean and variance at the given co-ordinates using the 2D squared exponential 
        kernel
        '''
        outputx = output_coords[0].astype(float)
        outputy = output_coords[1].astype(float)

        maxsize = 2.0 * 10**7
        nout = outputx.size
        
        nobs = len(self.obsx)
        nsplits = np.ceil(nout*nobs / maxsize)
        splitsize = int(np.ceil(nout/nsplits)) # make sure we don't kill memory

        # Determine whether to calculate all points from scratch, or just those close to new/changed observations
        update_all_points = self.update_all_points
        if self.f==[] or update_all_points or len(self.outputx) != len(outputx) or np.any(self.outputx != outputx) or \
                np.any(self.outputy != outputy):
            self.f = np.zeros(nout)
            update_all_points = True
            self.v = np.zeros(nout) # diagonal values only        

        self.outputx = outputx # save these coordinates so we can check if they are the same when called again
        self.outputy = outputy

        Cov = self.G.dot(self.Ks).dot(self.G) + self.Q
        self.L = cholesky(Cov,lower=True, check_finite=False, overwrite_a=True)  

        #Vpart = solve_triangular(self.L, self.G, lower=True, check_finite=False, overwrite_b=True)
        
        for s in np.arange(nsplits):
            
            logging.debug("Computing posterior for split %i out of %i" % (s,nsplits))
            
            start = int(s*splitsize)            
            end = int(start+splitsize) if start+splitsize<=nout else -1
            outputx_s = outputx[start:end].reshape(end-start,1)
            outputy_s = outputy[start:end].reshape(end-start,1)
            if end>=0:
                output_flat = np.arange(start, end)
            else:
                output_flat = np.arange(start, nout)  
            
            ddx = outputx_s - self.obsx
            ddy = outputy_s - self.obsy
            if self.cov_func == "sqexp":
                Kpred = self.sq_exp_cov(ddx, ddy)
                Vprior = self.sq_exp_cov(0, 0)
            elif self.cov_func == "matern":
                Kpred = self.matern(ddx, ddy)
                Vprior = self.sq_exp_cov(0, 0)
            Kpred = Kpred / self.s
            Vprior = Vprior / self.s
            
            #update all idxs?
            if not update_all_points:
                changed = np.argwhere(np.sum(Kpred[:,self.changed_obs],axis=1)>0.1).flatten()
                Kpred = Kpred[changed,:]
                output_flat = output_flat[changed]
                
            f_s = fblas.dgemm(alpha=1.0, a=Kpred.T, b=self.G.dot(self.A).T, trans_a=True, trans_b=True, overwrite_c=True)
            V = solve_triangular(self.L, self.G.dot(Kpred.T), lower=True, overwrite_b=True)
            v_s = Vprior - np.sum(V**2, axis=0)
   
            self.f[output_flat] = f_s
            self.v[output_flat] = v_s
                
        f = self.f + self.mu0
                
        # Approximate the expected value of the variable transformed through the sigmoid.
        if expectedlog:
            m_post = np.log(sigmoid(f))
            v_post = self.v
        else:
            k = 1.0 / np.sqrt(1 + (np.pi * self.v / 8.0))
            m_post = sigmoid(k*f)
            if variance_method == 'rough':
                v_post = (sigmoid(f + np.sqrt(self.v)) - m_post)**2 + (sigmoid(f - np.sqrt(self.v)) - m_post)**2 / 2.0
        if self.verbose:
            logging.debug("gp grid predictions: %s" % str(m_post))
             
        return m_post, v_post

    def get_mean_density(self):
        '''
        Return an approximation to the mean density having calculated the latent mean using predict().
        :return:
        '''
        k = 1.0 / np.sqrt(1 + (np.pi * self.v / 8.0))
        m_post = sigmoid(k * (self.f * self.mu0))
        return m_post