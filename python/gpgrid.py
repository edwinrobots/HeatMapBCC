import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.sparse import coo_matrix
from scipy.optimize import fmin
from scipy.special import gammaln, psi
from scipy.stats import norm
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
    
    # hyper-parameters
    s = 1 # inverse output scale
    ls = 100 # inner length scale of the GP
    n_lengthscales = 1 # separate or single length scale?
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
    
    max_iter_VB = 200#1000
    max_iter_G = 200
    conv_threshold = 1e-5
    conv_threshold_G = 1e-3
    conv_check_freq = 2
    
    uselowerbound = True
    
    expec_sigmoid_method = 'k_approximation'

    outputx = []
    outputy = []
    
    p_rep = 1.0 # weight report values by a constant probability to indicate uncertainty in the reports
    
    def __init__(self, nx, ny, z0=0.5, shape_s0=None, rate_s0=None, s_initial=None, shape_ls=10, rate_ls=0.1, 
                 ls_initial=None, force_update_all_points=False, n_lengthscales=1):
        #Grid size for prediction
        self.nx = nx
        self.ny = ny
        
        self.n_lengthscales = n_lengthscales
        
        # Prior mean
        self.z0 = z0
        self.mu0 = logit(z0)
        
        # Output scale (sigmoid scaling)  
        
        if not shape_s0:
            shape_s0 = 1.0
        if not rate_s0:
            rate_s0 = 1.0#5.0#1/200.0#0.5 / logit(0.25**2 + 0.5) # SD of 0.25 in probability space  
        self.shape_s0 = shape_s0 # prior pseudo counts * 0.5
        self.rate_s0 = rate_s0 # prior sum of squared deviations
        self.shape_s = shape_s0
        self.rate_s = rate_s0 # init to the priors until we receive data
        if np.any(s_initial):
            self.s_initial = s_initial
            self.s = s_initial
        else:
            self.s_initial = self.shape_s0 / self.rate_s0
            self.s = self.shape_s0 / self.rate_s0 
            
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
    
    def reset_s(self):
        '''
        Sets the value of s back to the initial point
        '''
        self.s = self.s_initial
        if hasattr(self, 'obs_mean_prob'):
            self.obs_f = logit(self.obs_mean_prob) + self.mu0
        if hasattr(self, 'K'):
            self.obs_C = self.K / self.s
    
    def process_observations(self, obsx, obsy, obs_values): 
        if obs_values==[]:
            return [],[]      
        
        if self.z!=[] and obsx==self.rawobsx and obsy==self.rawobsy and obs_values==self.rawobs_points:
            return
        
        if self.verbose:
            logging.debug("GP grid fitting " + str(len(obsx)) + " observations.")
            
        self.obsx = np.array(obsx)
        self.obsy = np.array(obsy)
        
        obs_values = np.array(obs_values)        
        if self.obsx.dtype=='int' and (obs_values.ndim==1 or obs_values.shape[1]==1): # duplicate locations allowed, one count at each array entry
            if not np.any(self.obs_values):
                obs_values[obs_values == 1] = self.p_rep
                obs_values[obs_values == 0] = 1 - self.p_rep   
            
            grid_obs_counts = coo_matrix((np.ones(len(self.obsx)), (self.obsx, self.obsy)), shape=(self.nx,self.ny)).toarray()            
            grid_obs_pos_counts = coo_matrix((obs_values, (self.obsx, self.obsy)), shape=(self.nx,self.ny)).toarray()
            
            self.obsx, self.obsy = grid_obs_counts.nonzero()
            self.obs_values = grid_obs_pos_counts[self.obsx,self.obsy]
            obs_pos_counts = self.obs_values[:, np.newaxis]
            obs_total_counts = grid_obs_counts[self.obsx,self.obsy][:, np.newaxis]
            
        elif self.obsx.dtype=='float' and (obs_values.ndim==1 or obs_values.shape[1]==1):
            if not np.any(self.obs_values):
                obs_values[obs_values == 1] = self.p_rep
                obs_values[obs_values == 0] = 1 - self.p_rep
            self.obs_values = obs_values
            obs_pos_counts = self.obs_values[:, np.newaxis]
            obs_total_counts = np.ones(obs_pos_counts.shape)
             
        elif obs_values.shape[1]==2: 
            if not np.any(self.obs_values):
                obs_values[obs_values[:,0] == 1, 0] = self.p_rep
                obs_values[obs_values[:,0] == 0, 0] = 1 - self.p_rep            
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
        self.Q = np.diagflat(self.obs_mean_prob * (1 - self.obs_mean_prob) / obs_total_counts)
                    
    def ln_modelprior(self):
        #Gamma distribution over each value. Set the parameters of the gammas.
        lnp_gp = - gammaln(self.shape_ls) + self.shape_ls*np.log(self.rate_ls) \
                   + (self.shape_ls-1)*np.log(self.ls) - self.ls*self.rate_ls
        return np.sum(lnp_gp)
    
    def lowerbound(self):
        #calculate likelihood from the fitted model 
        y = self.obs_values.flatten().astype(float)
        
        if self.expec_sigmoid_method=='sample':
            f_samples = norm.rvs(loc=self.obs_f.flatten(), scale=np.sqrt(np.diag(self.obs_C)), size=1000) 
            rho_samples = sigmoid(f_samples)
            ln_rho_samples = np.log(rho_samples)
            ln_rho_neg_samples = np.log(1 - rho_samples)
            ElnRho = np.mean(ln_rho_samples, axis=0)
            ElnNotRho = np.mean(ln_rho_neg_samples, axis=0)
        else:
            k = 1.0 / np.sqrt(1 + (np.pi * np.diag(self.obs_C) / 8.0))
            k = k[:, np.newaxis]    
            E_Rho = sigmoid(k*self.obs_f)
            ElnRho = np.log(E_Rho)
            ElnNotRho = np.log(1-E_Rho)
        
        data_ll = np.sum(y * ElnRho + (self.obs_total_counts.flatten() - y) * ElnNotRho)
        
        logp_f = self.logpf()
        logq_f = self.logqf() 

        logp_s = self.logps()
        logq_s = self.logqs()
          
        if self.verbose:      
            logging.debug("DLL: %.5f, logp_f: %.5f, logq_f: %.5f, logp_s-logq_s: %.5f" % (data_ll, logp_f, logq_f, logp_s-logq_s) )
        
        lb = data_ll + logp_f - logq_f + logp_s - logq_s
        return lb

    def logpf(self):
        _, logdet_K = np.linalg.slogdet(self.K)
        logdet_Ks = - len(self.obs_f) * self.Elns + logdet_K
        #_, logdet_Ks = np.linalg.slogdet(self.Ks)
        invK_expecF = solve_triangular(self.cholK, self.obs_C + self.obs_f.dot(self.obs_f.T), trans=True, check_finite=False)
        invK_expecF = solve_triangular(self.cholK, invK_expecF, check_finite=False)
        invK_expecF *= self.s
        logpf = 0.5 * (- logdet_Ks - np.trace(invK_expecF))
        return logpf
        
    def logqf(self):
        # We want to do this, but we can simplify it, since the x and mean values cancel:
        _, logdet_C = np.linalg.slogdet(self.obs_C)
        logqf = 0.5 * (- logdet_C - len(self.obs_f))    
        return logqf 

    def logps(self):
        logprob_s = - gammaln(self.shape_s0) + self.shape_s0 * np.log(self.rate_s0) + (self.shape_s0-1) * self.Elns \
                    - self.rate_s0 * self.s
        return logprob_s            
        
    def logqs(self):
        lnq_s = - gammaln(self.shape_s) + self.shape_s * np.log(self.rate_s) + (self.shape_s-1) * self.Elns - \
                self.rate_s * self.s
                
        return lnq_s
    
    def neg_marginal_likelihood(self, hyperparams, expectedlog=False, use_MAP=False):
        '''
        Weight the marginal log data likelihood by the hyper-prior. Unnormalised posterior over the hyper-parameters.
        '''
        if np.any(np.isnan(hyperparams)):
            return np.inf
        self.ls = np.exp(hyperparams)
        self.ls = np.zeros(2) + self.ls
        if np.any(np.isinf(self.ls)):
            return np.inf
        
        # make sure we start again
        self.reset_s()
        self.obs_f = []
        self.fit((self.obsx, self.obsy), self.obs_values, expectedlog=expectedlog, process_obs=False, update_s=True)
        if self.verbose:
            logging.debug("Inverse output scale: %f" % self.s)        
        
        marginal_log_likelihood = self.lowerbound()
        
        log_model_prior = self.ln_modelprior()
        if use_MAP:
            lml = marginal_log_likelihood + log_model_prior
        else:
            lml = marginal_log_likelihood
        logging.debug("LML: %f, with Length-scale: %f, %f" % (lml, self.ls[0], self.ls[1]))
        return -lml
    
    def optimize(self, obs_coords, obs_values, expectedlog=True, maxfun=20, use_MAP=False):
        obsx = obs_coords[0]
        obsy = obs_coords[1]
        self.process_observations(obsx, obsy, obs_values) # process the data here so we don't repeat each call
        
        for d, ls in enumerate(self.ls):
            if ls == 1:
                logging.warning("Changing length-scale of 1 to 2 to avoid optimisation problems.")
                self.ls[d] = 2.0
        
        initialguess = np.log(self.ls)
        if self.n_lengthscales == 1:
            initialguess = initialguess[0]
        #initial_nlml = self.neg_marginal_likelihood(initialguess)
        ftol = np.abs(np.log(self.ls[0]) * 1e-4) #np.abs(initial_nlml / (1e3 * (int(self.ls[0] < 10.0) * (10.0 - self.ls[0]) + 1)))
        logging.debug("Ftol = %.5f" % ftol)
        opt_hyperparams, self.nlml, _, _, _ = fmin(self.neg_marginal_likelihood, initialguess, maxfun=maxfun, ftol=ftol, 
                               xtol=1e100, full_output=True, args=(use_MAP,))

        opt_hyperparams[0] = np.exp(opt_hyperparams[0])

        msg = "Optimal hyper-parameters: "
        for param in opt_hyperparams:
            msg += str(param)
        logging.debug(msg)   
        return self.mean_prob_obs, opt_hyperparams

    def sq_exp_cov(self, xvals, yvals):
        Kx = np.exp( -xvals**2 / self.ls[0] )
        Ky = np.exp( -yvals**2 / self.ls[1] )
        K = Kx * Ky
        return K

    def expec_fC(self, G_update_rate=1.0):
        KsG = self.Ks.dot(self.G.T, self.KsG)        
        self.Cov = KsG.T.dot(self.G.T, out=self.Cov) + self.Q 
        
        self.L = cholesky(self.Cov, lower=True, check_finite=False, overwrite_a=True)
        B = solve_triangular(self.L, (self.z - self.z0), lower=True, overwrite_b=True, check_finite=False)
        self.A = solve_triangular(self.L, B, lower=True, trans=True, overwrite_b=False, check_finite=False)
        self.obs_f = KsG.dot(self.A, out=self.obs_f)
        V = solve_triangular(self.L, KsG.T, lower=True, overwrite_b=True, check_finite=False)
        self.obs_C = self.Ks - V.T.dot(V, out=self.obs_C) 
        
        mean_X = sigmoid(self.obs_f.flatten())
        self.G = G_update_rate * np.diag(mean_X * (1-mean_X)) + (1 - G_update_rate) * self.G

    def fit(self, obs_coords, obs_values, expectedlog=False, process_obs=True, update_s=True):
        # Initialise the objects that store the observation data
        obsx = obs_coords[0]
        obsy = obs_coords[1]
        if process_obs:
            self.process_observations(obsx, obsy, obs_values)
            
        # Get the correct covariance matrix
        self.K = self.sq_exp_cov(self.obs_ddx, self.obs_ddy)
        self.K += 1e-6 * np.eye(len(self.K)) # jitter
        self.cholK = cholesky(self.K, overwrite_a=False, check_finite=False)
        
        if self.obsx==[]:
            mPr = 0.5
            stdPr = 0.25       
            return mPr, stdPr
             
        if self.verbose: 
            logging.debug("gp grid starting training with length-scales %f, %f..." % (self.ls[0], self.ls[1]) )
        
        self.shape_s = self.shape_s0 + len(self.obsx)/2.0
        
        # Initialise objects
        if not np.any(self.obs_f) or len(self.obs_f) != len(self.obsx):
            # Case where we have a completely new dataset or new observations
            self.obs_f = logit(self.obs_mean_prob)
            self.obs_C = self.K / self.s
            mean_X = self.obs_mean_prob.flatten() 
            self.G = np.diag(mean_X*(1-mean_X))
            prev_mean_X = mean_X    
            
            # Initialise here to speed up dot product            
            self.Cov = np.zeros(self.G.shape)
            self.KsG = np.zeros(self.G.shape)
            
            # initialise s
            self.rate_s = self.rate_s0 * self.shape_s / self.shape_s0
            self.s = self.shape_s / self.rate_s        
            self.Elns = psi(self.shape_s) - np.log(self.rate_s)
            self.Ks = self.K / self.s
            self.old_s = self.s
            if self.verbose:
                logging.debug("Setting the initial value of s=%.3f using a heuristic calculation from the data" % self.s)            
        else: # Restarting the fitting process with same observations, e.g. because length-scale has been altered for
            # optimisation. Restart from the old estimates.
            self.obs_f = self.obs_f - self.mu0
            mean_X = (self.mean_prob_obs - self.z0)
            
        old_obs_f = self.obs_f
        nIt = 0
        diff = 0
        L = -np.inf
        converged = False
        
        while not converged and nIt<self.max_iter_VB:    
            prev_mean_X = mean_X
                
            # Iterate a few times to get G to stabilise
            diff_G = 0
            G_update_rate = 1.0 # start with full size updates
            for inner_nIt in range(self.max_iter_G):
                oldG = self.G                
                self.expec_fC(G_update_rate=G_update_rate)
                prev_diff_G = diff_G # save last iteration's difference
                diff_G = np.max(np.abs(oldG - self.G))
                # Use a smaller update size if we get stuck oscillating about the solution
                if np.abs(diff_G) - np.abs(prev_diff_G) < 1e-6 and G_update_rate > 0.1:
                    G_update_rate *= 0.9
                if self.verbose:
                    logging.debug("Iterating over G: diff was %.5f in iteration %i" % (diff_G, inner_nIt))
                if diff_G < self.conv_threshold_G:
                    break;
            if inner_nIt >= self.max_iter_G - 1:
                logging.warning("G did not converge: diff was %.5f" % diff_G)
            
            #update the output scale parameter (also called latent function scale/sigmoid steepness)
            self.old_s = self.s 
            if update_s: 
                L_expecFF = solve_triangular(self.cholK, self.obs_C + self.obs_f.dot(self.obs_f.T), trans=True, 
                                             overwrite_b=True, check_finite=False)
                LT_L_expecFF = solve_triangular(self.cholK, L_expecFF, overwrite_b=True, check_finite=False)
                self.rate_s = self.rate_s0 + 0.5 * np.trace(LT_L_expecFF) 
                #Update expectation of s. See approximations for Binary Gaussian Process Classification, Hannes Nickisch
                self.s = self.shape_s / self.rate_s
                self.Elns = psi(self.shape_s) - np.log(self.rate_s)                      
                if self.verbose:
                    logging.debug("Updated inverse output scale: " + str(self.s))
                self.Ks = self.K / self.s            
                                    
            if self.uselowerbound and np.mod(nIt, self.conv_check_freq)==self.conv_check_freq-1:
                oldL = L
                L = self.lowerbound()
                diff = L - oldL
                
                if self.verbose:
                    logging.debug('GPGRID lower bound = %.5f, diff = %.5f at iteration %i' % (L, diff, nIt))
                    
                if diff < - np.abs(L) * self.conv_threshold: # ignore any error of less than ~1%, as we are using approximations here anyway
                    logging.warning('GPGRID Lower Bound = %.5f, changed by %.5f in iteration %i\
                            -- probable approximation error or bug. Output scale=%.3f.' % (L, diff, nIt, self.s))
                    
                converged = diff < np.abs(L) * self.conv_threshold
            elif np.mod(nIt, 3)==2:
                diff = np.max([np.max(np.abs(mean_X - prev_mean_X)), 
                           np.max(np.abs(mean_X*(1-mean_X) - prev_mean_X*(1-prev_mean_X))**0.5)])
                if self.verbose:
                    logging.debug('GPGRID mean_X diff = %.5f at iteration %.i' % (diff, nIt) )
                    
                sdiff = np.abs(self.old_s - self.s) / self.s
                
                if self.verbose:
                    logging.debug('GPGRID s diff = %.5f' % sdiff)
                
                diff = np.max([diff, sdiff])
                    
                converged = (diff < self.conv_threshold) & (nIt > 2)
            nIt += 1
                
        if self.verbose:
            logging.debug("gp grid trained with inverse output scale %.5f" % self.s)

        self.obs_f = self.obs_f + self.mu0
        self.changed_obs = (np.abs(self.obs_f - old_obs_f) > 0.05).reshape(-1)

        # draw samples from a Gaussian with mean f and covariance C
        if self.expec_sigmoid_method=='sample':
            f_samples = norm.rvs(loc=self.obs_f.flatten(), scale=np.sqrt(np.diag(self.obs_C)), size=1000) 
            rho_samples = sigmoid(f_samples)
            rho_neg_samples = 1 - rho_samples
            if expectedlog:
                rho_samples = np.log(rho_samples)
                rho_neg_samples = np.log(rho_neg_samples)
            mean_prob_obs = np.mean(rho_samples, axis=0)
            mean_prob_notobs = np.mean(rho_neg_samples, axis=0)
        else:
            k = 1.0 / np.sqrt(1 + (np.pi * np.diag(self.obs_C) / 8.0))
            k = k[:, np.newaxis]    
            mean_prob_obs = sigmoid(k*self.obs_f)
            mean_prob_notobs = 1 - mean_prob_obs
            if expectedlog:
                mean_prob_obs = np.log(mean_prob_obs)
                mean_prob_notobs = np.log(mean_prob_notobs)

        self.mean_prob_obs = mean_prob_obs
        self.mean_prob_notobs = mean_prob_notobs
        return mean_prob_obs.reshape(-1)
    
    def predict(self, output_coords, variance_method='rough'):
        '''
        Evaluate the function posterior mean and variance at the given co-ordinates using the 2D squared exponential 
        kernel
        '''
        self.outputx = output_coords[0].astype(float)[:, np.newaxis]
        self.outputy = output_coords[1].astype(float)[:, np.newaxis]

        ddx = self.outputx - self.obsx[np.newaxis, :]
        ddy = self.outputy - self.obsy[np.newaxis, :]
        Kpred = self.sq_exp_cov(ddx, ddy)
        Kpred /= self.s
        
        ddx = self.outputx - self.outputx.T
        ddy = self.outputy - self.outputy.T
        Kout = self.sq_exp_cov(ddx, ddy)
        Kout /= self.s
        Kout += 1e-6 * np.eye(len(Kout)) # jitter
        
        self.f = Kpred.dot(self.G.T).dot(self.A)
        V = solve_triangular(self.L, self.G.dot(Kpred.T), lower=True, overwrite_b=True, check_finite=False)
        self.C = Kout - V.T.dot(V)
        self.v = np.diag(self.C)[:, np.newaxis]
        if np.any(self.v < 0):
            self.v[(self.v < 0) & (self.v > -1e-6)] = 0
            if np.any(self.v < 0): # anything still below zero?
                logging.error("Variance has gone negative in GPgrid.predict()")
        self.f = self.f + self.mu0
                
        # Approximate the expected value of the variable transformed through the sigmoid.
        if variance_method == 'rough':
            k = 1.0 / np.sqrt(1 + (np.pi * self.v / 8.0))
            m_post = sigmoid(k*self.f)
            v_post = (sigmoid(self.f + np.sqrt(self.v)) - m_post)**2 + (sigmoid(self.f - np.sqrt(self.v)) - m_post)**2 / 2.0
        elif variance_method == 'sample':
            # draw samples from a Gaussian with mean f and variance v
            f_samples = norm.rvs(loc=self.f, scale=np.sqrt(self.v), size=(len(self.f), 2000))
            rho_samples = sigmoid(f_samples)
            m_post = np.mean(rho_samples, axis=1)[:, np.newaxis]
            v_post = np.mean((rho_samples - m_post)**2, axis=1)[:, np.newaxis]
        return m_post, v_post

    def get_mean_density(self):
        '''
        Return an approximation to the mean density having calculated the latent mean using predict().
        :return:
        '''
        k = 1.0 / np.sqrt(1 + (np.pi * self.v / 8.0))
        m_post = sigmoid(k * (self.f * self.mu0))
        return m_post
    
if __name__ == '__main__':
    import scipy.stats.multivariate_normal as mvn
    # run some tests on the learning algorithm
    
    N = 100.0
    
    # generate test ground truth
    s = 10
    mean = np.zeros(N)
    
    gridsize = 1000.0
    ls = 10.0
    
    x_all = np.arange(N) / N * gridsize 
    y_all = np.arange(N) / N * gridsize
    
    ddx = x_all[:, np.newaxis] - x_all[np.newaxis, :]
    ddy = y_all[:, np.newaxis] - y_all[np.newaxis, :]
            
    Kx = np.exp( -ddx**2 / ls )
    Ky = np.exp( -ddy**2 / ls )
    K = Kx * Ky
    
    K += np.eye(int(N)) * 1e-6
    
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