'''

Uses stochastic variational inference (SVI) to scale to larger datasets with limited memory. At each iteration 
of the VB algorithm, only a fixed number of random data points are used to update the distribution.

'''

import numpy as np
import logging
from gp_classifier_vb import GPClassifierVB
from sklearn.cluster import MiniBatchKMeans
from joblib import Parallel, delayed
import multiprocessing
from scipy.linalg import cholesky, solve_triangular
from scipy.special import psi

def _gradient_terms_for_subset(GPC, invKs_fhat, sigmasq, dim):
    dKdls = GPC.K_mm * GPC.kernel_derfactor(GPC.inducing_coords, GPC.ls, dim)  / GPC.s
    firstterm = invKs_fhat.T.dot(dKdls).dot(invKs_fhat)
    secondterm = np.trace(GPC.invKs_mm_uS.dot(sigmasq).dot(dKdls))
    return 0.5 * (firstterm - secondterm)

class GPClassifierSVI(GPClassifierVB):
    
    data_idx_i = [] # data indices to update in the current iteration, i
    changed_selection = True # indicates whether the random subset of data has changed since variables were initialised
    
    def __init__(self, ninput_features, z0=0.5, shape_s0=2, rate_s0=2, shape_ls=10, rate_ls=0.1, ls_initial=None, 
                 force_update_all_points=False, kernel_func='matern_3_2', max_update_size=10000, 
                 ninducing=500, use_svi=True, delay=1.0, forgetting_rate=0.9, verbose=False):
        
        self.max_update_size = max_update_size # maximum number of data points to update in each SVI iteration
        
        # initialise the forgetting rate and delay for SVI
        self.forgetting_rate = forgetting_rate
        self.delay = delay # delay must be at least 1
        
        # number of inducing points
        self.ninducing = ninducing
        
        # default state before initialisation, unless some inducing coordinates are set by external call 
        self.inducing_coords = None
        self.K_mm = None
        self.invK_mm = None
        self.K_nm = None
        
        # if use_svi is switched off, we revert to the standard (parent class) VB implementation
        self.use_svi = use_svi
        
        self.fixed_sample_idxs = False
        
        super(GPClassifierSVI, self).__init__(ninput_features, z0, shape_s0, rate_s0, shape_ls, rate_ls, ls_initial, 
                                    force_update_all_points, kernel_func, verbose=verbose)      

    # Initialisation --------------------------------------------------------------------------------------------------
        
    def _init_params(self, mu0=None):            
        super(GPClassifierSVI, self)._init_params(mu0)
        if self.use_svi:            
            self._choose_inducing_points()
    
    def _init_covariance(self):
        if not self.use_svi:
            return super(GPClassifierSVI, self)._init_covariance()
        
    def _choose_inducing_points(self):
        # choose a set of inducing points -- for testing we can set these to the same as the observation points.
        nobs = self.obs_f.shape[0]       
        
        self.update_size = self.max_update_size # number of inducing points in each stochastic update
        if self.update_size > nobs:
            self.update_size = nobs  
               
        if self.ninducing > self.obs_coords.shape[0]:
            if self.inducing_coords is not None:
                logging.warning('replacing intial inducing points with observation coordinates because they are smaller.')
            self.ninducing = self.obs_coords.shape[0]
            self.inducing_coords = self.obs_coords
            # invalidate matrices passed in to init_inducing_points() as we need to recompute for new inducing points 
            self.K_mm = None
            self.invK_mm  = None
            self.K_nm = None
        elif self.inducing_coords is None:
            init_size = 300
            if self.ninducing > init_size:
                init_size = self.ninducing
            kmeans = MiniBatchKMeans(init_size=init_size, n_clusters=self.ninducing)
            kmeans.fit(self.obs_coords)
        
            #self.inducing_coords = self.obs_coords[np.random.randint(0, nobs, size=(ninducing)), :]
            self.inducing_coords = kmeans.cluster_centers_
            #self.inducing_coords = self.obs_coords
                        
        self.u_invSm = np.zeros((self.ninducing, 1), dtype=float)# theta_1
        self.u_invS = np.zeros((self.ninducing, self.ninducing), dtype=float) # theta_2
        self.um_minus_mu0 = np.zeros((self.ninducing, 1))
        self.invKs_mm_uS = np.eye(self.ninducing)

        if self.K_mm is None:
            self.K_mm = self.kernel_func(self.inducing_coords, self.ls)
            self.K_mm += 1e-6 * np.eye(len(self.K_mm)) # jitter
        if self.invK_mm is None: 
            self.invK_mm = np.linalg.inv(self.K_mm)
        if self.K_nm is None:
            self.K_nm = self.kernel_func(self.obs_coords, self.ls, self.inducing_coords)
        
        self.shape_s = self.shape_s0 + 0.5 * self.ninducing # update this because we are not using n_locs data points

    # Mapping between latent and observation spaces -------------------------------------------------------------------
        
    def _update_jacobian(self, G_update_rate=1.0):
        if not self.use_svi:
            return super(GPClassifierSVI, self)._update_jacobian(G_update_rate)
                
        if len(self.data_idx_i):
            g_obs_f = self.forward_model(self.obs_f.flatten()[self.data_idx_i]) # first order Taylor series approximation
        else:
            g_obs_f = self.forward_model(self.obs_f.flatten())
        J = np.diag(g_obs_f * (1-g_obs_f))
        if G_update_rate==1 or not len(self.G) or self.G.shape != J.shape or self.changed_selection:
            # either G has not been initialised, or is from different observations, or random subset of data has changed 
            self.G = J
        else:
            self.G = G_update_rate * J + (1 - G_update_rate) * self.G
            
        # set the selected observations i.e. not their locations, but the actual indexes in the input data. In the 
        # standard case, these are actually the same anyway, but this can change if the observations are pairwise prefs.
        self.data_obs_idx_i = self.data_idx_i
            
        return g_obs_f  

    # Log Likelihood Computation ------------------------------------------------------------------------------------- 
        
    def _logpf(self):
        if not self.use_svi:
            return super(GPClassifierSVI, self)._logpf()
            
        _, logdet_K = np.linalg.slogdet(self.Ks_mm * self.s)
        D = len(self.um_minus_mu0)
        logdet_Ks = - D * self.Elns + logdet_K
                
        invK_expecF = self.invKs_mm_uS + self.invKs_mm.dot(self.um_minus_mu0.dot(self.um_minus_mu0.T))
        
        _logpf = 0.5 * (- np.log(2*np.pi)*D - logdet_Ks - np.trace(invK_expecF))
        return _logpf
        
    def _logqf(self):
        if not self.use_svi:
            return super(GPClassifierSVI, self)._logqf()
                
        # We want to do this, but we can simplify it, since the x and mean values cancel:
        _, logdet_C = np.linalg.slogdet(self.u_invS)
        logdet_C = -logdet_C # because we are using the inverse of the covariance
        D = len(self.um_minus_mu0)
        _logqf = 0.5 * (- np.log(2*np.pi)*D - logdet_C - D)    
        return _logqf         
    
    def lowerbound_gradient(self, dim):
        '''
        Gradient of the lower bound on the marginal likelihood with respect to the length-scale of dimension dim.
        '''
        if not self.use_svi:
            return super(GPClassifierSVI, self).lowerbound_gradient(dim)
         
        fhat = self.um_minus_mu0
        invKs_fhat = (self.invK_mm * self.s).dot(fhat)                
         
        GTQ = (self.G.T * self.Q[ np.newaxis, :])
        sigmasq = self.invK_mm.dot(self.K_nm.T).dot(GTQ).dot(self.G).dot(self.K_nm).dot(self.invK_mm)
         
        if self.n_lengthscales == 1 or dim == -1: # create an array with values for each dimension
            dims = range(self.obs_coords.shape[1])
        else: # do it for only the dimension dim
            dims = [dim]        
        
        num_jobs = multiprocessing.cpu_count()
        gradient = Parallel(n_jobs=num_jobs)(delayed(_gradient_terms_for_subset)(self, invKs_fhat, sigmasq, dim) 
                                             for dim in dims)
        if self.n_lengthscales == 1:
            # sum the partial derivatives over all the dimensions
            gradient = np.sum(gradient)
         
        return gradient    
    
    # Training methods ------------------------------------------------------------------------------------------------
          
    def _expec_f(self):
        if self.use_svi:  
            # change the randomly selected observation points
            self._update_sample()        
        
        super(GPClassifierSVI, self)._expec_f()          
            
    def _update_f(self):
        if not self.use_svi:
            return super(GPClassifierSVI, self)._update_f()
        
        # this is done here not update_sample because it needs to be updated every time obs_f is updated
        self.obs_f_i = self.obs_f[self.data_idx_i]    
        
        Ks_nm_i = self.Ks_nm[self.data_idx_i, :]
        
        Q = self.Q[self.data_obs_idx_i][np.newaxis, :]
        Lambda_factor1 = self.invKs_mm.dot(Ks_nm_i.T).dot(self.G.T)
        Lambda_i = (Lambda_factor1 / Q).dot(Lambda_factor1.T)
        
        # calculate the learning rate for SVI
        rho_i = (self.vb_iter + self.delay) ** (-self.forgetting_rate)
        #print "\rho_i = %f " % rho_i
        
        # weighting. Lambda and 
        w_i = np.sum(self.obs_total_counts) / float(np.sum(self.obs_total_counts[self.data_obs_idx_i]))#self.obs_f.shape[0] / float(self.obs_f_i.shape[0])
        
        # S is the variational covariance parameter for the inducing points, u. Canonical parameter theta_2 = -0.5 * S^-1.
        # The variational update to theta_2 is (1-rho)*S^-1 + rho*Lambda. Since Lambda includes a sum of Lambda_i over 
        # all data points i, the stochastic update weights a sample sum of Lambda_i over a mini-batch.  
        self.u_invS = (1 - rho_i) * self.prev_u_invS + rho_i * (w_i * Lambda_i  + self.invKs_mm)
        
        # use the estimate given by the Taylor series expansion
        z0 = self.forward_model(self.obs_f, subset_idxs=self.data_idx_i) + self.G.dot(self.mu0_i - self.obs_f_i)
        y = self.z_i - z0
        
        # Variational update to theta_1 is (1-rho)*S^-1m + rho*beta*K_mm^-1.K_mn.y  
        self.u_invSm = (1 - rho_i) * self.prev_u_invSm + w_i * rho_i * (Lambda_factor1/Q).dot(y)
        
        # Next step is to use this to update f, so we can in turn update G. The contribution to Lambda_m and u_inv_S should therefore be made only once G has stabilised!
        L_u_invS = cholesky(self.u_invS.T, lower=True, check_finite=False)
        B = solve_triangular(L_u_invS, self.invKs_mm.T, lower=True, check_finite=False)
        A = solve_triangular(L_u_invS, B, lower=True, trans=True, check_finite=False, overwrite_b=True)
        self.L_u_invS = L_u_invS
        self.invKs_mm_uS = A.T
        
        #covpair_uS = covpair.dot(np.linalg.inv(self.u_invS))
        
        self.um_minus_mu0 = solve_triangular(L_u_invS, self.u_invSm, lower=True, check_finite=False)
        self.um_minus_mu0 = solve_triangular(L_u_invS, self.um_minus_mu0, lower=True, trans=True, check_finite=False, 
                                             overwrite_b=True)
        
        self.obs_f = self._f_given_u(self.Ks_nm, self.mu0)
                
    def _f_given_u(self, Ks_nm, mu0, Ks_nn=None):
        covpair_uS = Ks_nm.dot(self.invKs_mm_uS)
        fhat = covpair_uS.dot(self.u_invSm) + mu0
        if Ks_nn is not None:
            covpair =  Ks_nm.dot(self.invKs_mm)
            C = Ks_nn + (covpair_uS - covpair.dot(self.Ks_mm)).dot(covpair.T)
            return fhat, C
        else:
            return fhat

       
    def obs_variance(self):
        return np.diag(self._f_given_u(self.Ks_nm, self.mu0, np.diag(np.ones(self.obs_f.shape[0])) / self.s)[1])[:, np.newaxis]

    def _expec_s(self):
        if not self.use_svi:
            return super(GPClassifierSVI, self)._expec_s()
                    
        self.old_s = self.s 
        invK_mm_expecFF = self.invKs_mm_uS / self.s + self.invK_mm.dot(self.um_minus_mu0.dot(self.um_minus_mu0.T))
        self.rate_s = self.rate_s0 + 0.5 * np.trace(invK_mm_expecFF) 
        #Update expectation of s. See approximations for Binary Gaussian Process Classification, Hannes Nickisch
        self.s = self.shape_s / self.rate_s
        self.Elns = psi(self.shape_s) - np.log(self.rate_s)                      
        if self.verbose:
            logging.debug("Updated inverse output scale: " + str(self.s))
            
        self.Ks_mm = self.K_mm / self.s
        self.invKs_mm  = self.invK_mm * self.s
        self.Ks_nm = self.K_nm / self.s            
    
    def _update_sample(self):
        
        # once the iterations over G are complete, we accept this stochastic VB update
        self.prev_u_invSm = self.u_invSm
        self.prev_u_invS = self.u_invS        
        
        self._update_sample_idxs()
        
        self.Ks_mm = self.K_mm / self.s
        self.invKs_mm  = self.invK_mm * self.s
        self.Ks_nm = self.K_nm / self.s            
        
        self.G = 0 # reset because we will need to compute afresh with new sample
        self.z_i = self.z[self.data_obs_idx_i]
        self.mu0_i = self.mu0[self.data_idx_i]
        
    def fix_sample_idxs(self, data_idx_i):
        '''
        Pass in a set of pre-determined sample idxs rather than changing them stochastically inside this implementation.
        '''
        self.data_idx_i = data_idx_i
        self.fixed_sample_idxs = True
        
    def init_inducing_points(self, inducing_coords, K_mm=None, invK_mm=None, K_nm=None):
        self.ninducing = inducing_coords.shape[0]
        self.inducing_coords = inducing_coords
        if K_mm is not None:
            self.K_mm = K_mm
        if invK_mm is not None:
            self.invK_mm = invK_mm
        if K_nm is not None:
            self.K_nm = K_nm
        
    def _update_sample_idxs(self):
        if not self.fixed_sample_idxs:
            nobs = self.obs_f.shape[0]            
            self.data_idx_i = np.sort(np.random.choice(nobs, self.update_size, replace=False))
        self.data_obs_idx_i = self.data_idx_i  
    
    # Prediction methods ---------------------------------------------------------------------------------------------
                
    def _expec_f_output(self, block, blockidxs, reset_block=True):
        if not self.use_svi:
            return super(GPClassifierSVI, self)._expec_f_output(block, blockidxs, reset_block)
            
        if block not in self.K_out or reset_block:
            block_coords = self.output_coords[blockidxs]        
            self.K_out[block] = self.kernel_func(block_coords, self.ls, self.inducing_coords)
            self.K_out[block] /= self.s        
                
        self.f[blockidxs, :], C_out = self._f_given_u(self.K_out[block], self.mu0_output[blockidxs, :], 1.0 / self.s)
        self.v[blockidxs, 0] = np.diag(C_out) 