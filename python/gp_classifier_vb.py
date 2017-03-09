import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.sparse import coo_matrix
from scipy.optimize import fmin
from scipy.special import gammaln, psi, binom
from scipy.stats import norm, gamma
import logging

def coord_arr_to_1d(arr):
    arr = np.ascontiguousarray(arr)
    return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))

def coord_arr_from_1d(arr, dtype, dims):
    arr = arr.view(dtype)
    return arr.reshape(dims)

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

def temper_extreme_probs(probs, zero_only=False):
    if not zero_only:
        probs[probs > 1-1e-7] = 1-1e-7
        
    probs[probs < 1e-7] = 1e-7
    
    return probs

def diagonal(distances, ls):
    same_locs = np.sum(distances, axis=2) == 0
    K = np.zeros((distances.shape[0], distances.shape[1]), dtype=float)
    K[same_locs] = 1.0        
    return K

def sq_exp_cov(distances, ls):
    K = np.zeros(distances.shape)
    for d in range(distances.shape[2]):
        K[:, :, d] = np.exp( -distances[:, :, d]**2 / ls[d] )
    K = np.prod(K, axis=2)
    return K

def matern_3_2(distances, ls):
    K = np.zeros(distances.shape)
    for d in range(distances.shape[2]):
        K[:, :, d] = np.abs(distances[:, :, d]) * 3**0.5 / ls[d]
        K[:, :, d] = (1 + K[:, :, d]) * np.exp(-K[:, :, d])
    K = np.prod(K, axis=2)
    return K

def matern_3_2_from_raw_vals(vals, ls):
    distances = np.zeros((vals[0].size, vals[0].size, vals.shape[0]))
    for i, xvals in enumerate(vals):
        if xvals.ndim == 1:
            xvals = xvals[:, np.newaxis]
        elif xvals.shape[0] == 1 and xvals.shape[1] > 1:
            xvals = xvals.T
        xdists = xvals - xvals.T
        distances[:, :, i] = xdists
    K = matern_3_2(distances, ls)
    return K

class GPClassifierVB(object):
    verbose = False
    
    # hyper-parameters
    s = 1 # inverse output scale
    ls = 100 # inner length scale of the GP
    n_lengthscales = 1 # separate or single length scale?
    mu0 = 0

    # parameters for the hyper-priors if we want to optimise the hyper-parameters
    shape_ls = 1
    rate_ls = 0
    shape_s0 = 1 
    rate_s0 = 1       

    # save the training points
    obs_values = [] # value of the positive class at each observations. Any duplicate points will be summed.
    obs_f = []
    obs_C = []
    
    G = []
    z = []
    K = []
    Q = []
    f = []
    v = []
    
    n_converged = 1 # number of iterations while the algorithm appears to be converged -- in case of local maxima    
    max_iter_VB = 200#1000
    min_iter_VB = 5
    max_iter_G = 10
    conv_threshold = 1e-5
    conv_threshold_G = 1e-5
    conv_check_freq = 2
    
    uselowerbound = True
    
    p_rep = 1.0 # weight report values by a constant probability to indicate uncertainty in the reports
    
    def __init__(self, ninput_features, z0=0.5, shape_s0=2, rate_s0=2, shape_ls=10, rate_ls=1, ls_initial=None, 
                 force_update_all_points=False, kernel_func='matern_3_2'):
        #Grid size for prediction
        self.ninput_features = ninput_features
        
        if ls_initial is not None:
            self.n_lengthscales = len(ls_initial) # can pass in a single length scale to be used for all dimensions
        else:
            self.n_lengthscales = self.ninput_features
            
        # Output scale (sigmoid scaling)  
        self.shape_s0 = float(shape_s0) # prior pseudo counts * 0.5
        self.rate_s0 = float(rate_s0) # prior sum of squared deviations
        self.shape_s = float(shape_s0)
        self.rate_s = float(rate_s0) # init to the priors until we receive data
        self.s = self.shape_s0 / self.rate_s0         
                
        # Prior mean
        self._init_prior_mean_f(z0)
                    
        #Length-scale
        self.shape_ls = shape_ls # prior pseudo counts * 0.5
        self.rate_ls = rate_ls # analogous to a sum of changes between points at a distance of 1
        if np.any(ls_initial):
            self.ls = np.array(ls_initial)
        else:
            self.ls = self.shape_ls / self.rate_ls
            self.ls = np.zeros(self.ninput_features) + self.ls

        self.ls = self.ls.astype(float)

        #Algorithm configuration        
        self.update_all_points = force_update_all_points

        self._select_covariance_function(kernel_func)
        
    # Initialisation --------------------------------------------------------------------------------------------------
    
    def _init_params(self, mu0):
        self._init_obs_mu0(mu0)     
        # Prior noise variance
        self._init_obs_prior()
            
        self._estimate_obs_noise()    
        
        self._init_obs_f()
            
        # Get the correct covariance matrix
        self.K = self.kernel_func(self.obs_distances, self.ls)
        self.K += 1e-6 * np.eye(len(self.K)) # jitter
        self.cholK = cholesky(self.K, overwrite_a=False, check_finite=False)
                    
        # Initialise here to speed up dot product -- assume we need to do this whenever there is new data
        self.Cov = np.zeros((self.Ntrain, self.Ntrain))
        self.KsG = np.zeros((self.n_locs, self.Ntrain))  
            
        # initialise s
        self._init_s()  
        
        #g_obs_f = self._update_jacobian(G_update_rate) # don't do this here otherwise the loop below will repeate the 
        # same calculation with the same values, meaning that the convergence check will think nothing changes in the 
        # first iteration. 
        if self.G is not 0 and not len(self.G):
            self.G = 0
                
    def _init_prior_mean_f(self, z0):
        self.mu0_default = logit(z0)
        
    def _init_obs_mu0(self, mu0):
        if mu0 is None:
            mu0 = self.mu0_default
        self.mu0 = np.zeros((self.n_locs, 1)) + mu0
    
    def _init_obs_f(self):
        # Mean probability at observed points given local observations
        self.obs_f = logit(self.obs_mean)
        self.Ntrain = self.obs_f.size 
    
    def _estimate_obs_noise(self):
        # Noise in observations
        nu0_total = np.sum(self.nu0, axis=0)
        self.obs_mean = (self.obs_values + self.nu0[1]) / (self.obs_total_counts + nu0_total)
        var_obs_mean = self.obs_mean * (1-self.obs_mean) / (self.obs_total_counts + nu0_total + 1) # uncertainty in obs_mean
        self.Q = (self.obs_mean * (1 - self.obs_mean) - var_obs_mean) / self.obs_total_counts
        self.Q = self.Q.flatten()
                
    def _init_obs_prior(self):
        f_samples = norm.rvs(loc=self.mu0, scale=np.sqrt(self.rate_s0/self.shape_s0), size=(self.n_locs, 50000))
        rho_samples = self.forward_model(f_samples)
        rho_mean = np.mean(rho_samples)
        rho_var = np.var(rho_samples)
        # find the beta parameters
        a_plus_b = 1.0 / (rho_var / (rho_mean*(1 - rho_mean))) - 1
        a = a_plus_b * rho_mean
        b = a_plus_b * (1 - rho_mean)
        #b = 1.0
        #a = 1.0
        self.nu0 = np.array([b, a])
        if self.verbose:
            logging.debug("Prior parameters for the observation noise variance are: %s" % str(self.nu0))        

    def _init_s(self):
        self.shape_s = self.shape_s0 + self.n_locs/2.0 # reset!
        self.rate_s = (self.rate_s0 + 0.5 * np.sum((self.obs_f-self.mu0)**2)) + self.rate_s0*self.shape_s/self.shape_s0            
        self.s = self.shape_s / self.rate_s        
        self.Elns = psi(self.shape_s) - np.log(self.rate_s)
        
        self.Ks = self.K / self.s
        self.obs_C = self.K / self.s
        
        self.old_s = self.s
        if self.verbose:
            logging.debug("Setting the initial precision scale to s=%.3f" % self.s)

    def _select_covariance_function(self, cov_type):
        if cov_type == 'diagonal':
            self.kernel_func = diagonal
        elif cov_type == 'matern_3_2':
            self.kernel_func = matern_3_2
        elif cov_type == 'sq_exp':
            self.kernel_func = sq_exp_cov
        else:
            logging.error('GPClassifierVB: Invalid covariance type %s' % cov_type)        

    # Input data handling ---------------------------------------------------------------------------------------------

    def _count_observations(self, obs_coords, n_obs, poscounts, totals):
        obs_coords = np.array(obs_coords)
        if obs_coords.shape[0] == self.ninput_features and obs_coords.shape[1] != self.ninput_features:
            if obs_coords.ndim == 3 and obs_coords.shape[2] == 1:
                obs_coords = obs_coords.reshape((obs_coords.shape[0], obs_coords.shape[1]))            
            obs_coords = obs_coords.T
            
        if obs_coords.dtype=='int': # duplicate locations should be merged and the number of duplicates counted
            ravelled_coords = coord_arr_to_1d(obs_coords)
            uravelled_coords, idxs = np.unique(ravelled_coords, return_inverse=True)
            grid_obs_counts = coo_matrix((totals, (idxs, np.ones(n_obs))) ).toarray()            
            grid_obs_pos_counts = coo_matrix((poscounts, (idxs, np.ones(n_obs))) ).toarray()
        
            nonzero_idxs = grid_obs_counts.nonzero()[0] # ravelled coordinates with duplicates removed
            self.obs_coords = coord_arr_from_1d(uravelled_coords[nonzero_idxs], obs_coords.dtype, 
                                                [nonzero_idxs.size, self.ninput_features])
            return grid_obs_pos_counts[nonzero_idxs, 1], grid_obs_counts[nonzero_idxs, 1]
                    
        elif obs_coords.dtype=='float': # Duplicate locations are not merged
            self.obs_coords = obs_coords          
        
            return poscounts, totals # these remain unaltered as we have not de-duplicated
                       
    def _process_observations(self, obs_coords, obs_values, totals=None): 
        if obs_values==[]:
            return [],[]      
        
        obs_values = np.array(obs_values)
        n_obs = obs_values.shape[0]                 
        
        if self.verbose:
            logging.debug("GP grid for %i observations." % n_obs)
        
        if not np.any(totals >= 0):
            if (obs_values.ndim==1 or obs_values.shape[1]==1): # obs_value is one column with values of either 0 or 1
                totals = np.ones(n_obs)
            else: # obs_values given as two columns: first is positive counts, second is total counts.
                totals = obs_values[:, 1]
        elif (obs_values.shape[1]==2):
            logging.warning('GPClassifierVB received two sets of totals; ignoring the second column of the obs_values argument')            
          
        if (obs_values.ndim==1 or obs_values.shape[1]==1): # obs_value is one column with values of either 0 or 1
            poscounts = obs_values.flatten()
        elif (obs_values.shape[1]==2): # obs_values given as two columns: first is positive counts, second is total counts. 
            poscounts = obs_values[:, 0]
            
        if not np.any(self.obs_values >= 0):
            poscounts[poscounts == 1] = self.p_rep
            poscounts[poscounts == 0] = 1 - self.p_rep

        # remove duplicates etc.
        poscounts, totals = self._count_observations(obs_coords, n_obs, poscounts, totals)
        self.obs_values = poscounts[:, np.newaxis]
        self.obs_total_counts = totals[:, np.newaxis]
        n_locations = self.obs_coords.shape[0]
        self.n_locs = n_locations
            
        if self.verbose:
            logging.debug("Number of observed locations =" + str(self.obs_values.shape[0]))
        
        self._observations_to_z()
        
        #Update to produce training matrices only over known points
        self.obs_distances = np.zeros((n_locations, n_locations, self.ninput_features))
        obs_coords_3d = self.obs_coords.reshape((n_locations, 1, self.ninput_features))
        for d in range(self.ninput_features):
            self.obs_distances[:, :, d] = obs_coords_3d[:, :, d].T - obs_coords_3d[:, :, d]
        self.obs_distances = self.obs_distances.astype(float)
         
    def _observations_to_z(self):
        obs_probs = self.obs_values/self.obs_total_counts
        self.z = obs_probs         
         
    # Mapping between latent and observation spaces -------------------------------------------------------------------
    
    def forward_model(self, f, subset_idxs=[]):
        if len(subset_idxs):
            return sigmoid(f[subset_idxs])
        else:
            return sigmoid(f)
    
    def _update_jacobian(self, G_update_rate=1.0):
        g_obs_f = self.forward_model(self.obs_f.flatten()) # first order Taylor series approximation
        J = np.diag(g_obs_f * (1-g_obs_f))
        if G_update_rate==1 or not len(self.G) or self.G.shape != J.shape: # either G has not been initialised, or is from different observations
            self.G = J
        else:
            self.G = G_update_rate * J + (1 - G_update_rate) * self.G
        self.g_obs_f = g_obs_f         
         
    # Log Likelihood Computation ------------------------------------------------------------------------------------- 
    
    def lowerbound(self, return_terms=False):
        if self.verbose:
            logging.debug('Total f mean = %f' % np.sum(np.abs(self.obs_f)))
            logging.debug('Total f var = %f' % np.sum(np.diag(self.obs_C))) 

        logrho, lognotrho = self._logpt()
        
#         k = 1.0 / np.sqrt(1 + (np.pi * np.diag(self.obs_C)[:, np.newaxis] / 8.0))
#         rho_rough = sigmoid(k* self.obs_f)
#         notrho_rough = sigmoid(-k*self.obs_f)
#         logrho = np.log(rho_rough)
#         lognotrho = np.log(notrho_rough)
#          
#         rho = self.forward_model(self.obs_f)
#         logrho = np.log(rho)
#         lognotrho = np.log(1 - rho)
        
        data_ll = self._data_ll(logrho, lognotrho)
        
        logp_f = self._logpf()
        logq_f = self._logqf() 

        logp_s = self._logps()
        logq_s = self._logqs()
         
        if self.verbose:      
            logging.debug("DLL: %.5f, logp_f: %.5f, logq_f: %.5f, logp_s-logq_s: %.5f" % (data_ll, logp_f, logq_f, logp_s-logq_s) )
#             logging.debug("pobs : %.4f, pz: %.4f" % (pobs, pz) )
            logging.debug("logp_f - logq_f: %.5f. logp_s - logq_s: %.5f" % (logp_f - logq_f, logp_s - logq_s))
            logging.debug("LB terms without the output scale: %.3f" % (data_ll + logp_f - logq_f))

        lb = data_ll + logp_f - logq_f + logp_s - logq_s
        
        if return_terms:
            return lb, data_ll, logp_f, logq_f, logp_s, logq_s
        
        return lb
    
    def ln_modelprior(self):
        #Gamma distribution over each value. Set the parameters of the gammas.
        lnp_gp = - gammaln(self.shape_ls) + self.shape_ls*np.log(self.rate_ls) \
                   + (self.shape_ls-1)*np.log(self.ls) - self.ls*self.rate_ls
        return np.sum(lnp_gp)
    
    def _data_ll(self, logrho, lognotrho):
        bc = binom(self.obs_total_counts, self.z * self.obs_total_counts)
        logbc = np.log(bc)
        lpobs = np.sum(self.z*self.obs_total_counts * logrho + self.obs_total_counts*(1-self.z) * lognotrho)
        lpobs += np.sum(logbc)
        
        data_ll = lpobs    
        return data_ll      
    
    def _logpt(self):    
        f_samples = norm.rvs(loc=self.obs_f, scale=np.sqrt(np.diag(self.G.T.dot(np.diag(self.Q)).dot(self.G)))[:, np.newaxis], 
                             size=(self.n_locs, 5000))
        rho_samples = self.forward_model(f_samples)
        rho_samples = temper_extreme_probs(rho_samples)
        lognotrho_samples = np.log(1 - rho_samples)
        logrho_samples = np.log(rho_samples)
        logrho = np.mean(logrho_samples, axis=1)[:, np.newaxis]
        lognotrho = np.mean(lognotrho_samples, axis=1)[:, np.newaxis]

        return logrho, lognotrho        
    
    def _logpf(self):
        _, logdet_K = np.linalg.slogdet(self.K)
        logdet_Ks = - len(self.obs_f) * self.Elns + logdet_K
        
        mu0 = np.zeros((len(self.obs_f), 1)) + self.mu0
        
        invK_expecF = solve_triangular(self.cholK, self.obs_C + self.obs_f.dot(self.obs_f.T) - \
                   mu0.dot(self.obs_f.T) - self.obs_f.dot(mu0.T) + mu0.dot(mu0.T), trans=True, check_finite=False)
        invK_expecF = solve_triangular(self.cholK, invK_expecF, check_finite=False)
        invK_expecF *= self.s # because we're using self.cholK not cholesky(self.Ks)
        D = len(self.obs_f)
        logpf = 0.5 * (- np.log(2*np.pi)*D - logdet_Ks - np.trace(invK_expecF))
        return logpf
        
    def _logqf(self):
        # We want to do this, but we can simplify it, since the x and mean values cancel:
        _, logdet_C = np.linalg.slogdet(self.obs_C)
        D = len(self.obs_f)
        logqf = 0.5 * (- np.log(2*np.pi)*D - logdet_C - D)    
        return logqf 

    def _logps(self):
        logprob_s = - gammaln(self.shape_s0) + self.shape_s0 * np.log(self.rate_s0) + (self.shape_s0-1) * self.Elns \
                    - self.rate_s0 * self.s
        return logprob_s            
        
    def _logqs(self):
        lnq_s = - gammaln(self.shape_s) + self.shape_s * np.log(self.rate_s) + (self.shape_s-1) * self.Elns - \
                self.rate_s * self.s
                
        return lnq_s
    
    def neg_marginal_likelihood(self, hyperparams, dimension, use_MAP=False):
        '''
        Weight the marginal log data likelihood by the hyper-prior. Unnormalised posterior over the hyper-parameters.
        '''
        if np.any(np.isnan(hyperparams)):
            return np.inf
        self.ls[dimension] = np.exp(hyperparams)
        if np.any(np.isinf(self.ls)):
            return np.inf
        
        # make sure we start again
        #Sets the value of parameters back to the initial guess
        self._init_params(None)
        self.fit(process_obs=False, optimize=False)
        if self.verbose:
            logging.debug("Inverse output scale: %f" % self.s)        
        
        marginal_log_likelihood = self.lowerbound()
        
        if use_MAP:
            log_model_prior = self.ln_modelprior()        
            lml = marginal_log_likelihood + log_model_prior
        else:
            lml = marginal_log_likelihood
        logging.debug("LML: %f, with Length-scales: %s" % (lml, self.ls))
        return -lml
    
    # Training methods ------------------------------------------------------------------------------------------------

    def fit(self, obs_coords=None, obs_values=None, totals=None, process_obs=True, mu0=None, optimize=False, 
            maxfun=20, use_MAP=False):
        '''
        obs_coords -- coordinates of observations as an N x D array, where N is number of observations, 
        D is number of dimensions
        '''
        if optimize:
            return self.optimize(obs_coords, obs_values, totals, process_obs, mu0, maxfun, use_MAP)
        
        # Initialise the objects that store the observation data
        if process_obs:
            self._process_observations(obs_coords, obs_values, totals)
            self._init_params(mu0)     
         
        elif mu0 is not None: # updated mean but not updated observations
            self._init_obs_mu0(mu0)     
        if not len(self.obs_coords):
            return
             
        if self.verbose: 
            logging.debug("GP Classifier VB: training with length-scales %s" % (self.ls) )
        
        self.vb_iter = 0
        converged_count = 0
        prev_val = -np.inf
        while converged_count < self.n_converged and self.vb_iter<self.max_iter_VB:    
            self._expec_f()
            
            #update the output scale parameter (also called latent function scale/sigmoid steepness)
            self._expec_s()
                                   
            converged, prev_val = self._check_convergence(prev_val)
            converged_count += converged
            self.vb_iter += 1    
            
        self._update_f() # this is needed so that L and A match s
            
        if self.verbose:
            logging.debug("gp grid trained with inverse output scale %.5f" % self.s)

    def optimize(self, obs_coords, obs_values, totals=None, process_obs=True, mu0=None, maxfun=20, use_MAP=False, 
                 nrestarts=1):

        if process_obs:
            self._process_observations(obs_coords, obs_values, totals) # process the data here so we don't repeat each call
            self._init_params(mu0)
                    
        for d, ls in enumerate(self.ls):
            min_nlml = np.inf
            best_opt_hyperparams = None
            best_iter = -1            
            
            logging.debug("Optimising length-scale for %i dimension" % d)
            
            # optimise each length-scale sequentially in turn
            for r in range(nrestarts):
                if ls == 1:
                    logging.warning("Changing length-scale of 1 to 2 to avoid optimisation problems.")
                    ls = 2.0
            
                initialguess = np.log(ls)
                logging.debug("Initial length-scale guess for dimension %i in restart %i: %.3f" % (d, r, ls))
        
                ftol = self.conv_threshold * 1e2
                logging.debug("Ftol = %.5f" % ftol)
                opt_hyperparams, nlml, _, _, _ = fmin(self.neg_marginal_likelihood, initialguess, maxfun=maxfun, 
                                              ftol=ftol, xtol=ls * 1e100, full_output=True, args=(d, use_MAP,))

                if nlml < min_nlml:
                    min_nlml = nlml
                    best_opt_hyperparams = opt_hyperparams
                    best_iter = r
                    
                # choose a new lengthscale for the initial guess of the next attempt
                ls = gamma.rvs(self.shape_ls, scale=1.0/self.rate_ls)
    
            if best_iter < r:
                # need to go back to the best result
                self.neg_marginal_likelihood(best_opt_hyperparams, d, use_MAP=False)

        logging.debug("Optimal hyper-parameters: %s" % self.ls)   
        return self.ls, -min_nlml # return the log marginal likelihood

    def _expec_f(self):
        ''' 
        Compute the expected value of f given current q() distributions for other parameters
        '''
        diff_G = 0
        G_update_rate = 1.0 # start with full size updates
        # Iterate a few times to get G to stabilise
        for G_iter in range(self.max_iter_G):
            oldG = self.G                
            self._update_jacobian(G_update_rate)                
            self._update_f()
            prev_diff_G = diff_G # save last iteration's difference
            diff_G = np.max(np.abs(oldG - self.G))
            # Use a smaller update size if we get stuck oscillating about the solution
            if np.abs(diff_G) - np.abs(prev_diff_G) < 1e-6 and G_update_rate > 0.1:
                G_update_rate *= 0.9
            if self.verbose:
                logging.debug("Iterating over G: diff was %.5f in iteration %i" % (diff_G, G_iter))
            if diff_G < self.conv_threshold_G:
                break;
        if G_iter >= self.max_iter_G - 1:
            if self.verbose:
                logging.debug("G did not converge: diff was %.5f" % diff_G)        

    def _update_f(self):
        self.KsG = self.Ks.dot(self.G.T, out=self.KsG)
        self.Cov = self.KsG.T.dot(self.G.T, out=self.Cov)
        self.Cov[range(self.Cov.shape[0]), range(self.Cov.shape[0])] += self.Q
        
        # use the estimate given by the Taylor series expansion
        z0 = self.forward_model(self.obs_f) + self.G.dot(self.mu0 - self.obs_f) 
        
        self.L = cholesky(self.Cov, lower=True, check_finite=False, overwrite_a=True)
        B = solve_triangular(self.L, (self.z - z0), lower=True, overwrite_b=True, check_finite=False)
        self.A = solve_triangular(self.L, B, lower=True, trans=True, overwrite_b=False, check_finite=False)
        self.obs_f = self.KsG.dot(self.A, out=self.obs_f) + self.mu0 # need to add the prior mean here?
        V = solve_triangular(self.L, self.KsG.T, lower=True, overwrite_b=True, check_finite=False)
        self.obs_C = self.Ks - V.T.dot(V, out=self.obs_C) 

    def _expec_s(self):
        self.old_s = self.s         
        L_expecFF = solve_triangular(self.cholK, self.obs_C + self.obs_f.dot(self.obs_f.T) \
                                      - self.mu0.dot(self.obs_f.T) - self.obs_f.dot(self.mu0.T) + 
                                      self.mu0.dot(self.mu0.T), trans=True, 
                                     overwrite_b=True, check_finite=False)
        LT_L_expecFF = solve_triangular(self.cholK, L_expecFF, overwrite_b=True, check_finite=False)
        self.rate_s = self.rate_s0 + 0.5 * np.trace(LT_L_expecFF) 
        #Update expectation of s. See approximations for Binary Gaussian Process Classification, Hannes Nickisch
        self.s = self.shape_s / self.rate_s
        self.Elns = psi(self.shape_s) - np.log(self.rate_s)                      
        if self.verbose:
            logging.debug("Updated inverse output scale: " + str(self.s))
        self.Ks = self.K / self.s           

    def _check_convergence(self, prev_val):
        if self.uselowerbound and np.mod(self.vb_iter, self.conv_check_freq)==self.conv_check_freq-1:
            oldL = prev_val
            L = self.lowerbound()
            diff = (L - oldL) / np.abs(L)
            
            if self.verbose:
                logging.debug('GP Classifier VB lower bound = %.5f, diff = %.5f at iteration %i' % (L, diff, self.vb_iter))
                
            if diff < - self.conv_threshold: # ignore any error of less than ~1%, as we are using approximations here anyway
                logging.warning('GP Classifier VB Lower Bound = %.5f, changed by %.5f in iteration %i' % 
                                (L, diff, self.vb_iter))
                logging.warning('-- probable approximation error or bug. Output scale=%.3f.' % (self.s))
                
            current_value = L
            converged = diff < self.conv_threshold
        elif np.mod(self.vb_iter, self.conv_check_freq)==2:
            diff = np.max([np.max(np.abs(self.g_obs_f - prev_val)), 
                       np.max(np.abs(self.g_obs_f*(1-self.g_obs_f) - prev_val*(1-prev_val))**0.5)])
            if self.verbose:
                logging.debug('GP Classifier VB g_obs_f diff = %.5f at iteration %.i' % (diff, self.vb_iter) )
                
            sdiff = np.abs(self.old_s - self.s) / self.s
            if self.verbose:
                logging.debug('GP Classifier VB s diff = %.5f' % sdiff)

            diff = np.max([diff, sdiff])
            current_value = self.g_obs_f
            converged = (diff < self.conv_threshold) & (self.vb_iter > 2)
        else:
            return False, prev_val # not checking in this iteration, return the old value and don't converge

        return (converged & (self.vb_iter+1 >= self.min_iter_VB)), current_value 

    # Prediction methods ---------------------------------------------------------------------------------------------
    
    def predict(self, output_coords, variance_method='rough', max_block_size=1e5, expectedlog=False, return_not=False, 
                mu0_output=None):
        '''
        Evaluate the function posterior mean and variance at the given co-ordinates using the 2D squared exponential 
        kernel
        '''
        
        # if no output_coords provided, give predictions at the fitted locations
        if not output_coords == None and not len(output_coords):
            return self.predict_obs(variance_method, expectedlog, return_not)
        
        nblocks, noutputs = self._init_output_arrays(output_coords, max_block_size)
        
        if mu0_output is None:
            self.mu0_output = np.zeros((noutputs, 1)) + self.mu0_default
        else:
            self.mu0_output = mu0_output
        
        if variance_method=='sample':
            m_post = np.empty((noutputs, 1), dtype=float)
            not_m_post = np.empty((noutputs, 1), dtype=float)
            v_post = np.empty((noutputs, 1), dtype=float)
        
        for block in range(nblocks):
            if self.verbose:
                logging.debug("GPClassifierVB predicting block %i of %i" % (block, nblocks))            
            blockidxs = self._predict_block(block, max_block_size, noutputs)
                    
            # Approximate the expected value of the variable transformed through the sigmoid.
            if variance_method == 'sample' or expectedlog:
                m_post[blockidxs, :], not_m_post[blockidxs, :], v_post[blockidxs, :] = \
                    self._post_sample(self.f[blockidxs, :], self.v[blockidxs, :], expectedlog)
                
        if variance_method == 'rough' and not expectedlog:
            m_post, not_m_post, v_post = self._post_rough(self.f, self.v)
        elif variance_method == 'rough':
            logging.warning("Switched to using sample method as expected log requested. No quick method is available.")
            
        if return_not:
            return m_post, not_m_post, v_post
        else:
            return m_post, v_post   
       
    def predict_obs(self, variance_method='rough', expectedlog=False, return_not=False):
        f = self.obs_f
        v = np.diag(self.obs_C)[:, np.newaxis]
        if variance_method=='rough' and not expectedlog:
            m_post, not_m_post, v_post = self._post_rough(f, v)
        else:
            if variance_method == 'rough':
                logging.warning("Switched to using sample method as expected log requested. No quick method is available.")
            m_post, not_m_post, v_post = self._post_sample(f, v, variance_method)
        if return_not:
            return m_post, not_m_post, v_post
        else:
            return m_post, v_post       
       
    def predict_grid(self, nx, ny, variance_method='rough', max_block_size=1e5, return_not=False, mu0_output=None):    
        nout = nx * ny
        outputx = np.tile(np.arange(nx, dtype=np.float).reshape(nx, 1), (1, ny)).reshape(nout)
        outputy = np.tile(np.arange(ny, dtype=np.float).reshape(1, ny), (nx, 1)).reshape(nout)
        self.predict([outputx, outputy], variance_method, max_block_size, return_not=return_not, mu0_output=mu0_output)
    
    def predict_f(self, items_coords=[], max_block_size=1e5, mu0_output=None):
        nblocks, noutputs = self._init_output_arrays(items_coords, max_block_size)
                
        if mu0_output is not None and len(mu0_output):
            self.mu0_output = mu0_output
        else:
            self.mu0_output = np.zeros((noutputs, 1)) + self.mu0_default
                
        for block in range(nblocks):
            if self.verbose:
                logging.debug("GPClassifierVB predicting block %i of %i" % (block, nblocks))            
            self._predict_block(block, max_block_size, noutputs)
        
        return self.f, self.v       
        
    def _init_output_arrays(self, output_coords, max_block_size):
        self.output_coords = np.array(output_coords).astype(float)
        if self.output_coords.shape[0] == self.ninput_features and self.output_coords.shape[1] != self.ninput_features:
            if self.output_coords.ndim == 3 and self.output_coords.shape[2] == 1:
                self.output_coords = self.output_coords.reshape((self.output_coords.shape[0], self.output_coords.shape[1]))                  
            self.output_coords = self.output_coords.T
            
        noutputs = self.output_coords.shape[0]

        self.f = np.empty((noutputs, 1), dtype=float)
        self.v = np.empty((noutputs, 1), dtype=float)

        nblocks = int(np.ceil(float(noutputs) / max_block_size))

        return nblocks, noutputs          
        
    def _predict_block(self, block, max_block_size, noutputs):
        
        maxidx = (block + 1) * max_block_size
        if maxidx > noutputs:
            maxidx = noutputs
        blockidxs = np.arange(block * max_block_size, maxidx, dtype=int)
            
        self._expec_f_output(blockidxs)
        
        if np.any(self.v[blockidxs] < 0):
            self.v[(self.v[blockidxs] < 0) & (self.v[blockidxs] > -1e-6)] = 0
            if np.any(self.v[blockidxs] < 0): # anything still below zero?
                logging.error("Negative variance in GPClassifierVB.predict(), %f" % np.min(self.v[blockidxs]))
        
        self.f[blockidxs, :] = self.f[blockidxs, :] + self.mu0_output[blockidxs, :]
        
        return blockidxs        
        
    def _expec_f_output(self, blockidxs):
        block_coords = self.output_coords[blockidxs]        
        
        distances = np.zeros((block_coords.shape[0], self.obs_coords.shape[0], self.ninput_features))
        for d in range(self.ninput_features):
            distances[:, :, d] = block_coords[:, d:d+1] - self.obs_coords[:, d:d+1].T
        
        K_out = self.kernel_func(distances, self.ls)
        K_out /= self.s        
        
        self.f[blockidxs, :] = K_out.dot(self.G.T).dot(self.A)
        
        V = solve_triangular(self.L, self.G.dot(K_out.T), lower=True, overwrite_b=True, check_finite=False)
        self.v[blockidxs, 0] = 1.0 / self.s#self.kernel_func([0, 0], self.ls) / self.s
        self.v[blockidxs, 0] -= np.sum(V**2, axis=0) #np.diag(V.T.dot(V))[:, np.newaxis]
        
    def _post_rough(self, f, v):
        k = 1.0 / np.sqrt(1 + (np.pi * v / 8.0))
        m_post = sigmoid(k*f)
        not_m_post = sigmoid(-k*f)
        v_post = (sigmoid(f + np.sqrt(v)) - m_post)**2 + (sigmoid(f - np.sqrt(v)) - m_post)**2 / 2.0
        
        return m_post, not_m_post, v_post
    
    def _post_sample(self, f, v, expectedlog): 
        # draw samples from a Gaussian with mean f and variance v
        f_samples = norm.rvs(loc=f, scale=np.sqrt(v), size=(len(f), 10000))
        rho_samples = sigmoid(f_samples)
        rho_samples = temper_extreme_probs(rho_samples)
        rho_not_samples = 1 - rho_samples 
        if expectedlog:
            rho_samples = np.log(rho_samples)
            rho_not_samples = np.log(rho_not_samples)
        m_post = np.mean(rho_samples, axis=1)[:, np.newaxis]
        not_m_post = np.mean(rho_not_samples, axis=1)[:, np.newaxis]
        v_post = np.mean((rho_samples - m_post)**2, axis=1)[:, np.newaxis]
        
        return m_post, not_m_post, v_post
    
if __name__ == '__main__':
    logging.warning('Caution: this is not a proper test of the whole algorithm.')
    from scipy.stats import multivariate_normal as mvn
    # run some tests on the learning algorithm
    
    N = 100
    
    # generate test ground truth
    s = 10
    mean = np.zeros(N)
    
    gridsize = 1000.0
    ls = 10.0
    
    x_all = np.arange(N) / float(N) * gridsize
    y_all = np.arange(N) / float(N) * gridsize
    
    ddx = x_all[:, np.newaxis] - x_all[np.newaxis, :]
    ddy = y_all[:, np.newaxis] - y_all[np.newaxis, :]
            
    Kx = np.exp( -ddx**2 / ls )
    Ky = np.exp( -ddy**2 / ls )
    K = Kx * Ky
    
    K += np.eye(N) * 1e-6
    
    K_over_s = K / s
    invK = np.linalg.inv(K)
    L = cholesky(K, lower=True, check_finite=False)    
    
    nsamples = 500
    shape0 = 1.0
    rate0 = 1.0
    
    # now try to infer the output scale given the ground truth
    shape = shape0
    rate = rate0
    for i in range(nsamples):
        f_true = mvn.rvs(mean=mean, cov=K_over_s)[:, np.newaxis]
        
        shape += 0.5 * len(f_true)
        rate += 0.5 * np.trace( solve_triangular(L, solve_triangular(L, f_true.dot(f_true.T), 
                 lower=True, overwrite_b=True, check_finite=False), trans=True, overwrite_b=True, check_finite=False ))
    post_s = shape / rate
    print shape
    print rate
    print "Posterior estimate of s is %f" % post_s