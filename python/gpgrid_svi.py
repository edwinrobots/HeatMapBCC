'''

Uses stochastic variational inference (SVI) to scale to larger datasets with limited memory. At each iteration 
of the VB algorithm, only a fixed number of random data points are used to update the distribution.

'''

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.special import psi
import logging
from gpgrid import GPGrid
from sklearn.cluster import MiniBatchKMeans

class GPGridSVI(GPGrid):
    
    data_idx_i = [] # data indices to update in the current iteration, i
    changed_selection = True # indicates whether the random subset of data has changed since variables were initialised
    
    def __init__(self, dims, z0=0.5, shape_s0=None, rate_s0=None, s_initial=None, shape_ls=10, rate_ls=0.1, 
                 ls_initial=None, force_update_all_points=False, n_lengthscales=1, max_update_size=1000, ninducing=500):
        
        self.max_update_size = max_update_size # maximum number of data points to update in each SVI iteration
        
        # initialise the forgetting rate and delay for SVI
        self.forgetting_rate = 0.9
        self.delay = 1.0
        
        # number of inducing points
        self.ninducing = ninducing
                
        super(GPGridSVI, self).__init__(dims, z0, shape_s0, rate_s0, s_initial, shape_ls, rate_ls, ls_initial, 
                                    force_update_all_points, n_lengthscales)
        
    def logpf(self):
        _, logdet_K = np.linalg.slogdet(self.Ks_mm * self.s)
        D = len(self.um)
        logdet_Ks = - D * self.Elns + logdet_K
                
        invK_expecF = self.inv_Ks_mm_uS + self.inv_Ks_mm.dot(self.um.dot(self.um.T))
        
        logpf = 0.5 * (- np.log(2*np.pi)*D - logdet_Ks - np.trace(invK_expecF))
        return logpf
        
    def logqf(self):
        # We want to do this, but we can simplify it, since the x and mean values cancel:
        _, logdet_C = np.linalg.slogdet(self.u_invS)
        logdet_C = -logdet_C # because we are using the inverse of the covariance
        D = len(self.um)
        logqf = 0.5 * (- np.log(2*np.pi)*D - logdet_C - D)    
        return logqf         
    
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
    
    def f_given_u(self, Ks_nn, Ks_nm, mu0):
        covpair =  Ks_nm.dot(self.inv_Ks_mm)
        covpair_uS = Ks_nm.dot(self.inv_Ks_mm_uS)
        fhat = covpair_uS.dot(self.u_invSm) + mu0
        C = Ks_nn + (covpair_uS - covpair.dot(self.Ks_mm)).dot(covpair.T)
        return fhat, C 
            
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
            w_i * rho_i * self.inv_Ks_mm.dot(Ks_nm_i.T).dot(self.G.T/self.Q[self.data_obs_idx_i,:].T).dot(y)
        
        # Next step is to use this to update f, so we can in turn update G. The contribution to Lambda_m and u_inv_S should therefore be made only once G has stabilised!
        L_u_invS = cholesky(self.u_invS.T, lower=True, check_finite=False)
        B = solve_triangular(L_u_invS, self.inv_Ks_mm.T, lower=True, check_finite=False)
        A = solve_triangular(L_u_invS, B, lower=True, trans=True, check_finite=False, overwrite_b=True)
        self.inv_Ks_mm_uS = A.T
        
        #covpair_uS = covpair.dot(np.linalg.inv(self.u_invS))
        
        self.um = solve_triangular(L_u_invS, self.u_invSm, lower=True, check_finite=False)
        self.um = solve_triangular(L_u_invS, self.um, lower=True, trans=True, check_finite=False, overwrite_b=True)
        
        self.obs_f, self.obs_C = self.f_given_u(self.Ks, self.Ks_nm, self.mu0)
        
#         KsG = self.Ks.dot(self.G.T)        
#         Cov = KsG.T.dot(self.G.T)
#         Cov[range(Cov.shape[0]), range(Cov.shape[0])] += self.Q.flatten()
#         
#         self.L = cholesky(Cov, lower=True, check_finite=False, overwrite_a=True)
#         B = solve_triangular(self.L, (self.z - z0), lower=True, overwrite_b=True, check_finite=False)
#         self.A = solve_triangular(self.L, B, lower=True, trans=True, overwrite_b=False, check_finite=False)
#         obs_f = KsG.dot(self.A) + self.mu0 # need to add the prior mean here?
#         V = solve_triangular(self.L, KsG.T, lower=True, overwrite_b=True, check_finite=False)
#         obs_C = self.Ks - V.T.dot(V) 
#         
#         print np.sum(np.abs(obs_f - self.obs_f))

    def choose_inducing_points(self):
        kmeans = MiniBatchKMeans(n_clusters=self.ninducing)
        kmeans.fit(self.obs_coords)
        return kmeans.cluster_centers_
        #return np.random.randint(0, 100, size=(self.ninducing, 2)).astype(float)#self.obs_coords[:self.ninducing, :]

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
                
        nobs = self.obs_f.shape[0]
        update_size = self.max_update_size # number of inducing points in each stochastic update
        if update_size > nobs:
            update_size = nobs  
                          
        if self.ninducing > self.obs_coords.shape[0]:
            self.ninducing = self.obs_coords.shape[0]
            
        if process_obs:
            self.prev_u_invSm = np.zeros((self.ninducing, 1), dtype=float)# theta_1
            self.prev_u_invS = np.zeros((self.ninducing, self.ninducing), dtype=float) # theta_2
                        
        if not len(self.obs_coords):
            mPr = 0.5
            stdPr = 0.25       
            return mPr, stdPr
             
        if self.verbose: 
            logging.debug("gp grid starting training with length-scales %f, %f..." % (self.ls[0], self.ls[1]) )        
                
        # choose a set of inducing points -- for testing we will set these to the same as the observation points.
        self.inducing_coords = self.choose_inducing_points()

        mm_dist = np.zeros((self.ninducing, self.ninducing, len(self.dims)))
        nm_dist = np.zeros((nobs, self.ninducing, len(self.dims)))
        for d in range(len(self.dims)):
            mm_dist[:, :, d] = self.inducing_coords[:, d:d+1].T - self.inducing_coords[:, d:d+1]
            nm_dist[:, :, d] = self.inducing_coords[:, d:d+1].T - self.obs_coords[:, d:d+1].astype(float)
         
        K_mm = self.kernel_func(mm_dist)
        K_mm += 1e-6 * np.eye(len(K_mm)) # jitter 
        
        K_nm = self.kernel_func(nm_dist)
            
        G_update_rate = 1.0 # start with full size updates
                    
        nIt = 0
        diff = 0
        L = -np.inf
        convergedIt = 0
                
        while convergedIt < 5 and nIt<self.max_iter_VB: # require 5 iterations when we are seemingly converged to allow for small jitters in the stochastic updates    
            prev_obs_f = self.obs_f
            
            self.svi_iter = nIt
            
            # change the randomly selected observation points
            self.data_idx_i = np.random.choice(nobs, update_size, replace=False)
            
            invK_mm = np.linalg.inv(K_mm)
            self.Ks_mm = K_mm / self.s
            self.inv_Ks_mm  = invK_mm * self.s
            self.Ks_nm = K_nm / self.s            
            self.Ks = self.K / self.s
            
            self.update_jacobian(G_update_rate, self.data_idx_i)            
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
                invK_mm_expecFF = self.inv_Ks_mm_uS / self.s + invK_mm.dot(self.um.dot(self.um.T))
                self.rate_s = self.rate_s0 + 0.5 * np.trace(invK_mm_expecFF) 
                #Update expectation of s. See approximations for Binary Gaussian Process Classification, Hannes Nickisch
                self.s = self.shape_s / self.rate_s
                self.Elns = psi(self.shape_s) - np.log(self.rate_s)                      
                if self.verbose:
                    logging.debug("Updated inverse output scale: " + str(self.s))
                self.Ks_mm = K_mm / self.s
                self.inv_Ks_mm  = invK_mm * self.s
                self.Ks_nm = K_nm / self.s            
                self.Ks = self.K / self.s
                
#                 sdiff = np.abs(self.old_s - self.s) / self.s
#                 if sdiff > 0.01: # major changes in s -- ignore previous updates
#                     self.svi_iter = -1
                                           
            if self.uselowerbound and np.mod(nIt, self.conv_check_freq)==self.conv_check_freq-1:
                oldL = L
                L = self.lowerbound()
                diff = (L - oldL) / np.abs(L)
                
                if self.verbose:
                    logging.debug('GPGRID lower bound = %.5f, diff = %.5f at iteration %i' % (L, diff, nIt))
                    
                if diff < - self.conv_threshold: # ignore any error of less than ~1%, as we are using approximations here anyway
                    logging.warning('GPGRID Lower Bound = %.5f, changed by %.5f in iteration %i\
                            -- probable approximation error or bug. Output scale=%.3f.' % (L, diff, nIt, self.s))
                    
                convergedIt += int((nIt >= self.min_iter_VB) & (diff < self.conv_threshold))
            elif np.mod(nIt, self.conv_check_freq)==2:
                diff = np.max([np.max(np.abs(self.obs_f - prev_obs_f)), 
                           np.max(np.abs(self.obs_f*(1-self.obs_f) - prev_obs_f*(1-prev_obs_f))**0.5)])
                if self.verbose:
                    logging.debug('GPGRID_SVI obs_f diff = %.5f at iteration %.i' % (diff, nIt) )
                    
                sdiff = np.abs(self.old_s - self.s) / self.s
                
                if self.verbose:
                    logging.debug('GPGRID s diff = %.5f' % sdiff)
                
                diff = np.max([diff, sdiff])
                    
                convergedIt += int((nIt >= self.min_iter_VB) & (diff < self.conv_threshold) & (nIt > 2))
            nIt += 1
 
        if self.verbose:
            logging.debug("gp grid trained with inverse output scale %.5f" % self.s)
        
    def predict_block(self, block, max_block_size, noutputs):
        
        maxidx = (block + 1) * max_block_size
        if maxidx > noutputs:
            maxidx = noutputs
        blockidxs = np.arange(block * max_block_size, maxidx, dtype=int)
        
        block_coords = self.output_coords[blockidxs]

        distances = np.zeros((block_coords.shape[0], self.inducing_coords.shape[0], len(self.dims)))
        for d in range(len(self.dims)):
            distances[:, :, d] = block_coords[:, d:d+1] - self.inducing_coords[:, d:d+1].T
        
        K_out = self.kernel_func(distances)
        K_out /= self.s
        
        self.f[blockidxs, :], C_out = self.f_given_u(1.0 / self.s, K_out, self.mu0_output)
        self.v[blockidxs, 0] = np.diag(C_out)
        
        if np.any(self.v[blockidxs] < 0):
            self.v[(self.v[blockidxs] < 0) & (self.v[blockidxs] > -1e-6)] = 0
            if np.any(self.v[blockidxs] < 0): # anything still below zero?
                logging.error("Variance has gone negative in GPgrid.predict(), %f" % np.min(self.v[blockidxs]))
        
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