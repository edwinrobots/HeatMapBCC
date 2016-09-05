'''

Uses stochastic variational inference (SVI) to scale to larger datasets with limited memory. At each iteration 
of the VB algorithm, only a fixed number of random data points are used to update the distribution.

'''

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.special import psi
import logging
from gpgrid import GPGrid

class GPGridSVI(GPGrid):
    
    data_idx_i = [] # data indices to update in the current iteration, i
    changed_selection = True # indicates whether the random subset of data has changed since variables were initialised
    
    def __init__(self, dims, z0=0.5, shape_s0=None, rate_s0=None, s_initial=None, shape_ls=10, rate_ls=0.1, 
                 ls_initial=None, force_update_all_points=False, n_lengthscales=1, max_update_size=1000):
        
        self.max_update_size = max_update_size # maximum number of data points to update in each SVI iteration
        
        # initialise the forgetting rate and delay for SVI
        self.forgetting_rate = 0.9
        self.delay = 1.0
                
        super(GPGridSVI, self).__init__(dims, z0, shape_s0, rate_s0, s_initial, shape_ls, rate_ls, ls_initial, 
                                    force_update_all_points, n_lengthscales)
    
    
    def update_jacobian(self, G_update_rate=1.0, selection=[]):
        if len(selection):
            g_obs_f = self.forward_model(self.obs_f.flatten()[selection]) # first order Taylor series approximation
        else:
            g_obs_f = self.forward_model(self.obs_f.flatten())
        J = np.diag(g_obs_f * (1-g_obs_f))
        if not len(self.G) or self.G.shape != J.shape or self.changed_selection:
            # either G has not been initialised, or is from different observations, or random subset of data has changed 
            self.G = J
        else:
            self.G = G_update_rate * J + (1 - G_update_rate) * self.G
            
        # set the selected observations i.e. not their locations, but the actual indexes in the input data. In the 
        # standard case, these are actually the same anyway, but this can change if the observations are pairwise prefs.
        self.data_obs_idx_i = self.data_idx_i
            
        return g_obs_f
    
    def expec_fC(self, G_update_rate=1.0):
        
        Ks_nm_i = self.Ks_nm[self.data_idx_i, :]
        
        Lambda_factor1 = self.inv_Ks_mm.dot(Ks_nm_i.T).dot(self.G.T)
        Lambda_i = (Lambda_factor1 / self.Q[self.data_obs_idx_i,:].T).dot(Lambda_factor1.T)
        
        # calculate the learning rate for SVI
        rho_i = (self.svi_iter + self.delay) ** (-self.forgetting_rate)
        #print "\rho_i = %f " % rho_i
        
        # weighting. Lambda and 
        w_i = self.obs_f.shape[0] / float(self.obs_f_i.shape[0])
        
        # S is the variational covariance parameter for the inducing points, u. Canonical parameter theta_2 = -0.5 * S^-1.
        # The variational update to theta_2 is (1-rho)*S^-1 + rho*Lambda. Since Lambda includes a sum of Lambda_i over 
        # all data points i, the stochastic update weights a sample sum of Lambda_i over a mini-batch.  
        self.u_invS = (1 - rho_i) * self.prev_u_invS + rho_i * (w_i * Lambda_i  + self.inv_Ks_mm)
        
        # use the estimate given by the Taylor series expansion
        z0 = self.forward_model(self.obs_f, subset_idxs=self.data_idx_i) + self.G.dot(self.mu0_i - self.obs_f_i)
        y = self.z_i - z0
        
        # Variational update to theta_1 is (1-rho)*S^-1m + rho*beta*K_mm^-1.K_mn.y  
        self.u_invSm = (1 - rho_i) * self.prev_u_invSm + \
                                    w_i * rho_i * self.inv_Ks_mm.dot(Ks_nm_i.T).dot(self.G.T).dot(y)
        
        # Next step is to use this to update f, so we can in turn update G. The contribution to Lambda_m and u_inv_S should therefore be made only once G has stabilised!
        self.obs_f = self.Ks_nm.dot(self.u_invSm) + self.mu0
        self.obs_C = self.Ks - self.Ks_nm.dot(np.linalg.inv(self.u_invS)).dot(self.Ks_nm.T) 

    def fit(self, obs_coords, obs_values, totals=None, process_obs=True, update_s=True, mu0=None):
        # Initialise the objects that store the observation data
        if process_obs:
            self.process_observations(obs_coords, obs_values, totals)
            if mu0 is not None:
                self.mu0 = mu0
            else:
                self.init_obs_mu0()            
            
            # Get the correct covariance matrix
            self.K = self.kernel_func(self.obs_distances)
            self.K += 1e-6 * np.eye(len(self.K)) # jitter
            self.cholK = cholesky(self.K, overwrite_a=False, check_finite=False)
            
            # initialise s
            self.shape_s = self.shape_s0 + self.obs_coords.shape[0]/2.0 # reset!
            self.rate_s = (self.rate_s0 + 0.5 * np.sum((self.obs_f - self.mu0)**2)) + self.rate_s0 * self.shape_s / self.shape_s0
            self.s = self.shape_s / self.rate_s        
            self.Elns = psi(self.shape_s) - np.log(self.rate_s)
            
            self.Ks = self.K / self.s
            self.obs_C = self.K / self.s
            
            self.old_s = self.s
            if self.verbose:
                logging.debug("Setting the initial precision scale to s=%.3f" % self.s)
                
            # For SVI
            # choose a set of inducing points -- for testing we will set these to the same as the observation points.
            self.inducing_distances = self.obs_distances
            K_mm = self.K
            invK_mm = np.linalg.inv(K_mm)
            K_nm = self.K
            self.Ks_mm = K_mm / self.s
            self.inv_Ks_mm  = invK_mm * self.s
            self.Ks_nm = K_nm / self.s
            self.prev_u_invSm = np.zeros(self.obs_f.shape, dtype=float) # theta_1
            self.prev_u_invS = np.zeros(self.obs_C.shape, dtype=float) # theta_2
                        
        if not len(self.obs_coords):
            mPr = 0.5
            stdPr = 0.25       
            return mPr, stdPr
             
        if self.verbose: 
            logging.debug("gp grid starting training with length-scales %f, %f..." % (self.ls[0], self.ls[1]) )        
                
        nobs = self.obs_f.shape[0]
        update_size = self.max_update_size
        if update_size > nobs:
            update_size = nobs
        
        G_update_rate = 1.0 # start with full size updates
                    
        nIt = 0
        diff = 0
        L = -np.inf
        converged = False
                
        while not converged and nIt<self.max_iter_VB:    
            prev_obs_f = self.obs_f
            
            self.svi_iter = nIt
            
            # change the randomly selected observation points
            self.data_idx_i = np.arange(update_size)#np.random.choice(nobs, update_size, replace=False)
            
            self.update_jacobian(G_update_rate, self.data_idx_i)            
#            update_obs_size = self.G.shape[0] 
#            if process_obs and (nIt==0 or self.KsG_i.shape[1] != self.G.shape[0]):
#                # Initialise here to speed up dot product -- assume we need to do this whenever there is new data  
#                 self.Cov = np.zeros((update_obs_size, update_obs_size))
#                 self.KsG_i = np.zeros((nobs, update_obs_size))
#                 self.KsG_obs_i = np.zeros((update_size, update_obs_size))
            
#             self.Ks_i = self.Ks[:, self.data_idx_i]    
#             self.Ks_obs_i = self.Ks[self.data_idx_i][:, self.data_idx_i]
            self.obs_f_i = self.obs_f[self.data_idx_i]
            self.z_i = self.z[self.data_obs_idx_i]
            self.mu0_i = self.mu0[self.data_idx_i]
                
            # Iterate a few times to get G to stabilise
            diff_G = 0
            for inner_nIt in range(self.max_iter_G):
                oldG = self.G
                self.update_jacobian(G_update_rate, self.data_idx_i) 
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
                if self.verbose:
                    logging.debug("G did not converge: diff was %.5f" % diff_G)
            
            # once the iterations over G are complete, we accept this stochastic VB update
            self.prev_u_invSm = self.u_invSm
            self.prev_u_invS = self.u_invS
            
            #update the output scale parameter (also called latent function scale/sigmoid steepness)
            self.old_s = self.s 
            if update_s: 
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
                self.Ks_mm = K_mm / self.s      
                self.Ks_nm = K_nm / self.s
                                            
            if self.uselowerbound and np.mod(nIt, self.conv_check_freq)==self.conv_check_freq-1:
                oldL = L
                L = self.lowerbound()
                diff = (L - oldL) / np.abs(L)
                
                if self.verbose:
                    logging.debug('GPGRID lower bound = %.5f, diff = %.5f at iteration %i' % (L, diff, nIt))
                    
                if diff < - self.conv_threshold: # ignore any error of less than ~1%, as we are using approximations here anyway
                    logging.warning('GPGRID Lower Bound = %.5f, changed by %.5f in iteration %i\
                            -- probable approximation error or bug. Output scale=%.3f.' % (L, diff, nIt, self.s))
                    
                converged = diff < self.conv_threshold
            elif np.mod(nIt, self.conv_check_freq)==2:
                diff = np.max([np.max(np.abs(self.obs_f - prev_obs_f)), 
                           np.max(np.abs(self.obs_f*(1-self.obs_f) - prev_obs_f*(1-prev_obs_f))**0.5)])
                if self.verbose:
                    logging.debug('GPGRID_SVI obs_f diff = %.5f at iteration %.i' % (diff, nIt) )
                    
                sdiff = np.abs(self.old_s - self.s) / self.s
                
                if self.verbose:
                    logging.debug('GPGRID s diff = %.5f' % sdiff)
                
                diff = np.max([diff, sdiff])
                    
                converged = (diff < self.conv_threshold) & (nIt > 2)
            nIt += 1
            converged = converged & (nIt >= self.min_iter_VB)
 
        if self.verbose:
            logging.debug("gp grid trained with inverse output scale %.5f" % self.s)
        
    def predict_block(self, block, max_block_size, noutputs):
        
        maxidx = (block + 1) * max_block_size
        if maxidx > noutputs:
            maxidx = noutputs
        blockidxs = np.arange(block * max_block_size, maxidx, dtype=int)
        
        block_coords = self.output_coords[blockidxs]

        distances = np.zeros((block_coords.shape[0], self.obs_coords.shape[0], len(self.dims)))
        for d in range(len(self.dims)):
            distances[:, :, d] = block_coords[:, d:d+1] - self.obs_coords[:, d:d+1].T
        
        Kpred = self.kernel_func(distances)
        Kpred /= self.s
        
        # iterate to get the variance without having to perform cholesky on a large matrix
        v_not_converged = True
        niter_v = 0
        nobs = self.obs_f.shape[0]
        update_size = self.max_update_size
        if update_size > nobs:
            update_size = nobs
                
        while v_not_converged and niter_v < 20:
            # change the randomly selected observation points
            self.data_idx_i = np.random.choice(nobs, update_size, replace=False)
            rho_i = (niter_v + self.delay) ** (-self.forgetting_rate)
            
            self.update_jacobian(selection=self.data_idx_i)
            
            self.Ks_i = Kpred[:, self.data_idx_i]
            self.Ks_obs_i = self.Ks[self.data_idx_i][:, self.data_idx_i]            
            self.obs_f_i = self.obs_f[self.data_idx_i]
            self.z_i = self.z[self.data_obs_idx_i]
            self.mu0_i = self.mu0[self.data_idx_i]
            
            self.KsG_obs_i = self.Ks_obs_i.dot(self.G.T)
            self.KsG_i = self.Ks_i.dot(self.G.T)
            
            self.Cov = self.KsG_obs_i.T.dot(self.G.T) 
            self.Cov[range(self.Cov.shape[0]), range(self.Cov.shape[0])] += self.Q.flatten()[self.data_obs_idx_i]
            
            self.L = cholesky(self.Cov, lower=True, check_finite=False, overwrite_a=True)              
            
            z0 = self.forward_model(self.obs_f, subset_idxs=self.data_idx_i) + self.G.dot(self.mu0_i - self.obs_f_i)            
            B = solve_triangular(self.L, (self.z_i - z0), lower=True, overwrite_b=True, check_finite=False)
            self.A = solve_triangular(self.L, B, lower=True, trans=True, overwrite_b=False, check_finite=False)
            f_i = self.KsG_i.dot(self.A)
            oldf = self.f[blockidxs, 0]
            self.f[blockidxs, 0] = (1 - rho_i) * oldf + rho_i * f_i[:, 0]
        
            V = solve_triangular(self.L, self.KsG_i.T, lower=True, overwrite_b=True, check_finite=False)
            v_i = 1.0 / self.s - np.sum(V**2, axis=0)
            oldv = self.v[blockidxs, 0].flatten()
            self.v[blockidxs, 0] = (1 - rho_i) * oldv + rho_i * v_i
            
            diff_v = np.max(np.abs(self.v[blockidxs, 0] - oldv))
            diff_f = np.max(np.abs(self.f[blockidxs, 0] - oldf))
            if self.verbose:
                logging.debug("GPGrid predict_block(): SVI diff = %.5f, %.5f" % (diff_v, diff_f))
            if diff_v < self.conv_threshold and diff_f < self.conv_threshold:
                v_not_converged = False
                
            niter_v += 1
        
        if np.any(self.v[blockidxs] < 0):
            self.v[(self.v[blockidxs] < 0) & (self.v[blockidxs] > -1e-6)] = 0
            if np.any(self.v[blockidxs] < 0): # anything still below zero?
                logging.error("Variance has gone negative in GPgrid.predict(), %f" % np.min(self.v[blockidxs]))
                
        
        self.f[blockidxs, :] = self.f[blockidxs, :] + self.mu0_output        
        
    def init_output_arrays(self, output_coords, max_block_size, variance_method):
        self.output_coords = np.array(output_coords).astype(float)
        noutputs = self.output_coords.shape[0]

        self.f = np.empty((noutputs, 1), dtype=float)
        self.v = np.empty((noutputs, 1), dtype=float)

        nblocks = int(np.ceil(float(noutputs) / max_block_size))
        
        return nblocks, noutputs   
    
if __name__ == '__main__':
    from scipy.stats import multivariate_normal as mvn
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