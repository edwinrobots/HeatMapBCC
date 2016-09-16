
import ibcc
from gpgrid import GPGrid
import numpy as np
import logging

class HeatMapBCC(ibcc.IBCC):
    # Crowd-supervised GP (CSGP, CrowdGP)
    # GP crowd combination (GPCC)
    #Interpolating BCC (interBcc)
    #BRC (Bayesian report combination, because reporters describe surrounding areas, not just distinct data points like classifiers?)
    #Is it general/extensible enough a name? Does it work with documents,
    #for instance? What are catchy names of other algorithms? "naive
    #Bayes", "EM", "GP"... often two simple words. -- can we find a single
    #word for report combination?
    
    #Something more catchy.
    #Manifold+Classifier?
    #Maps + Events + Bayes = BEM (Bayesian Event Mapping)
    # Semi-supervised
    nx = 0 #grid size. May be able to generalise away from 2-D grids
    ny = 0

    mean_kappa = [] # posterior mean of kappa from the GP
    var_logodds_kappa = [] # posterior SD over kappa
    
    heatGP = {} # spatial GP model for kappa 
    
    obsx = [] # x-coordinates of locations with crowd reports
    obsy = [] # y-coordinates of locations with crowd reports
    
    crowdx = [] # ordered list of x-coordinates of crowd reports 
    crowdy = [] # ordered list of y-coordinates of crowd reports
    crowddict = {}

    output_to_grid = False # use the GP to predict all integer points in the grid.
    outputx = [] # coordinates of output points from the heat-map. If you just want to evaluate the whole grid, leave
    outputy = [] # these as empty lists

    lnkappa_out = [] # lnkappa at the output points given by the GP
    
    oldkappa = [] # density mean from previous VB iteration

    E_t_sparse = [] # target value estimates at the training points only.
    
    optimize_lengthscale_only = True
    
    n_lengthscales = 1
    
    update_s = 4
    
    conv_count = 0 # count the number of iterations where the change was below the convergence threshold

    def __init__(self, nx, ny, nclasses, nscores, alpha0, K, z0=0.5, shape_s0=None, rate_s0=None, shape_ls=10,
                 rate_ls=0.1, force_update_all_points=False, outputx=None, outputy=None, kernel_func='sq_exp'):
        if not outputy:
            outputy = []
        if not outputx:
            outputx = []
        self.nx = int(nx)
        self.ny = int(ny)
        self.outputx = outputx
        self.outputy = outputy
        self.lnkappa = []
        self.post_T = []
        self.update_all_points = force_update_all_points
        self.z0 = z0
        self.shape_s0 = shape_s0
        self.rate_s0 = rate_s0
        self.shape_ls = shape_ls
        self.rate_ls = rate_ls
        self.ls_initial = self.shape_ls / self.rate_ls + np.zeros(nclasses)
        self.kernel_func = kernel_func
        logging.debug('Setting up a 2-D grid. This should be generalised!')
        nu0 = np.ones(nclasses)     
        super(HeatMapBCC, self).__init__(nclasses, nscores, alpha0, nu0, K) 
    
    def desparsify_crowdlabels(self, crowdlabels):
        crowdlabels = np.array(crowdlabels)
        self.crowdx = crowdlabels[:,1]
        self.crowdy = crowdlabels[:,2]

        crowdcoords = zip(self.crowdx, self.crowdy)
        
        self.crowddict = {}
        self.obsx = []
        self.obsy = []
            
        for i in range(len(self.crowdx)):
            coord = crowdcoords[i]
            if not coord in self.crowddict:
                self.crowddict[coord] = len(self.crowddict.values()) # we are adding to the existing list in crowddict, 
                # skipping values of i that are duplicates   
                self.obsx.append(self.crowdx[i])
                self.obsy.append(self.crowdy[i])
        self.obsx = np.array(self.obsx)
        self.obsy = np.array(self.obsy)
        
        linearIdxs = [self.crowddict[l] for l in crowdcoords] # do this to ensure that we get unique values for each coord
        crowdlabels_flat = crowdlabels[:,[0,1,3]]
        crowdlabels_flat[:,1] = linearIdxs
        super(HeatMapBCC,self).desparsify_crowdlabels(crowdlabels_flat)
        self.full_N = self.N # when we re-sparsify, we do not fill in any gaps -- only the given indexes are predicted, unless call to predict grid is made 
        self.nu = [] # make sure we reset kappa so that it is resized correctly -- could avoid this in future to save a few iterations
        return crowdlabels_flat
    
    def preprocess_goldlabels(self, goldlabels=None):
        if np.any(goldlabels):
            goldlabels_flat = goldlabels.flatten()
        else:
            goldlabels_flat = None
        super(HeatMapBCC, self).preprocess_goldlabels(goldlabels_flat)
        
        
    def init_lnkappa(self):
        super(HeatMapBCC, self).init_lnkappa()  
        self.lnkappa = np.tile(self.lnkappa, (1, self.N))
        
        # Initialise the underlying GP with the current set of hyper-parameters           
        if self.nclasses==2:
            gprange = [1]
        else:
            gprange = np.arange(self.nclasses)
        self.oldkappa = np.exp(self.lnkappa)
        for j in gprange:
            #start with a homogeneous grid     
            self.heatGP[j] = GPGrid((self.nx, self.ny), z0=self.z0, shape_s0=self.shape_s0, rate_s0=self.rate_s0,
                                shape_ls=self.shape_ls, rate_ls=self.rate_ls,  ls_initial=self.ls_initial,
                                force_update_all_points=self.update_all_points, n_lengthscales=self.n_lengthscales, 
                                kernel_func=self.kernel_func)   
            self.heatGP[j].verbose = self.verbose
            self.heatGP[j].max_iter_VB = 1
            self.heatGP[j].min_iter_VB = 1
            self.heatGP[j].max_iter_G = 1
            self.heatGP[j].uselowerbound = False # we calculate the lower bound here instead of the GPGrid function
        
    def convergence_measure(self, oldET):
        kappadiff = np.max(np.abs( self.oldkappa - np.exp(self.lnkappa)))
        sdiff = 0
        for j in self.heatGP:
            sdiff_j = np.abs( self.heatGP[j].old_s - self.heatGP[j].s ) / self.heatGP[j].old_s
            if sdiff_j > sdiff:
                sdiff = sdiff_j
            if self.verbose:
                logging.debug('sdiff: %f, s = %f' % (sdiff, self.heatGP[j].s))
        return np.max( [super(HeatMapBCC, self).convergence_measure(oldET), kappadiff, sdiff])
    
    def convergence_check(self):
        if self.change<self.conv_threshold:
            self.conv_count += 1
        locally_converged = super(HeatMapBCC, self).convergence_check()
        if not locally_converged:
            return False
        
        if not self.update_s:
            if self.verbose:
                logging.debug("Switching on the inverse output scale updates.")
            self.update_s = True
            return False
        elif self.update_s < 4: # need to allow at least two more iterations, one to update s, and one more each to 
            # propagate the change through to t, then to pi, then back to t.
            self.update_s += 1
            return False
        elif self.nIts>self.min_iterations:
            return True
        else:
            return False

    def combine_classifications(self, crowdlabels, goldlabels=None, testidxs=None, optimise_hyperparams=False, 
                                maxiter=200, table_format=False):
        if self.table_format_flag:
            logging.error('Error: must use a sparse list of crowdsourced labels for HeatMapBCC')
            return []
        elif crowdlabels.shape[1] != 4:
            logging.error('Must use a sparse list of crowdsourced labels with 4 columns:')
            logging.error('Agent ID, x-cood, y-coord, response value') 
            return []
        super(HeatMapBCC, self).combine_classifications(crowdlabels, goldlabels, testidxs, optimise_hyperparams, 
                                                               maxiter, False)
        if self.output_to_grid:
            logging.debug("Resparsifying to grid")
            E_t_grid, _, _ = self.predict_grid()
            return E_t_grid
        else:
            return self.E_t
    
    def predict(self, outputx, outputy, variance_method='rough'):
        # Initialise containers for results at the output locations 
        nout = len(outputx)
        self.E_t_out = np.zeros((self.nclasses, nout))
        kappa_out = np.zeros((self.nclasses, nout))
        v_kappa_out = np.zeros((self.nclasses, nout))

        # Obtain the density estimates
        if self.nclasses==2:
            gprange = [1]
        else:
            gprange = np.arange(self.nclasses)
        for j in gprange:
            mean_kappa, v_kappa = self.heatGP[j].predict([outputx, outputy], variance_method=variance_method)
            kappa_out[j, :] = mean_kappa.flatten() 
            v_kappa_out[j, :] = v_kappa.flatten() 
        if self.nclasses==2:
            kappa_out[0, :] = 1 - kappa_out[1,:]
            v_kappa_out[0, :] = v_kappa_out[1, :]

        # Set the predictions for the targets
        self.E_t_out[:,:] = kappa_out
        #observation points that coincide with output points should take into account the labels, not just GP
        outputcoords = zip(outputx, outputy)
        obsin_idxs = np.array([self.crowddict[oc] if oc in self.crowddict else -1 for oc in outputcoords], dtype=int)
        obsout_idxs = obsin_idxs > -1
        obsin_idxs = obsin_idxs[obsout_idxs]
        self.E_t_out[:, obsout_idxs] = self.E_t.T[:, obsin_idxs]
        return self.E_t_out, kappa_out, v_kappa_out

    def predict_grid(self):
        E_t_grid = np.zeros((self.nclasses, self.nx, self.ny))
        kappa_grid = np.zeros((self.nclasses, self.nx, self.ny))
        v_kappa_grid = np.zeros((self.nclasses, self.nx, self.ny))
        #Evaluate the function posterior mean and variance at all coordinates in the grid. Use this to calculate
        #values for plotting a heat map. Calculate coordinates:
        nout = self.nx * self.ny
        outputx = np.tile(np.arange(self.nx, dtype=np.float).reshape(self.nx, 1), (1, self.ny)).reshape(nout)
        outputy = np.tile(np.arange(self.ny, dtype=np.float).reshape(1, self.ny), (self.nx, 1)).reshape(nout)

        if self.nclasses==2:
            gprange = [1]
        else:
            gprange = np.arange(self.nclasses)
        for j in gprange:
            kappa_grid_j, v_kappa_grid_j = self.heatGP[j].predict([outputx, outputy])
            kappa_grid[j, :,:] = kappa_grid_j.reshape((self.nx, self.ny))
            v_kappa_grid[j, :, :] = v_kappa_grid_j.reshape((self.nx, self.ny))
        if self.nclasses==2:
            kappa_grid[0,:,:] = 1 - kappa_grid[1,:,:]
            v_kappa_grid[0, :, :] = v_kappa_grid[1, :, :]
        
        E_t_grid[:] = kappa_grid
        
        obs_at_grid_points = (np.mod(self.obsx, 1)==0) & (np.mod(self.obsy, 1)==0)
        obsx_grid = self.obsx[obs_at_grid_points].astype(int)
        obsy_grid = self.obsy[obs_at_grid_points].astype(int)
        E_t_grid[:, obsx_grid, obsy_grid] = self.E_t[obs_at_grid_points, :].T

        return E_t_grid, kappa_grid, v_kappa_grid

    def resparsify_t(self):
        self.E_t_sparse = self.E_t # save the sparse version

        if np.any(self.outputx):
            logging.debug("Resparsifying to specified output points")        
            self.E_t = self.predict(self.outputx, self.outputy)

        return self.E_t
    
    def get_mean_kappa(self, j=1):
        return self.heatGP[j].get_mean_density()

    def get_heat_variance(self, j=1):
        return self.heatGP[j].v
    
    def get_heat_variance_grid(self, j=1):
        return self.heatGP[j].v.reshape((self.nx, self.ny))

    def expec_lnkappa(self):
        if self.nclasses==2:
            gprange = [1]
        else:
            gprange = np.arange(self.nclasses)
        self.oldkappa = np.exp(self.lnkappa)            
        for j in gprange:
            obs_values = self.E_t[:,j]                
            self.heatGP[j].fit([self.obsx, self.obsy], obs_values, update_s=self.update_s)
            self.heatGP[j].verbose = False
            lnkappaj, _ = self.heatGP[j].predict((self.obsx, self.obsy), variance_method='sample', 
                                                        expectedlog=True)
            self.heatGP[j].verbose = self.verbose
            self.lnkappa[j] = lnkappaj.flatten()
#             self.lnkappa[j][self.lnkappa[j] >= 1.0 - 1e-10] = np.log(1.0 - 1e-10) # make sure we don't encounter divide by zeros
            #self.lnkappa[j] = self.heatGP[j].logpz(1.0).flatten()
        if self.nclasses==2:
            kappa0 = 1-np.exp(self.lnkappa[1])
            self.lnkappa[0][kappa0 > 1e-10] = np.log(kappa0[kappa0 > 1e-10])
            #self.lnkappa[0] = self.heatGP[1].logpz(0.0).flatten()
            
    def lnjoint(self, alldata=False):
        lnkappa_all = self.lnkappa 
        if not self.uselowerbound and not alldata:       
            self.lnkappa = self.lnkappa[:, self.testidxs]
        super(HeatMapBCC, self).lnjoint(alldata)
        self.lnkappa = lnkappa_all

    def post_lnjoint_ct(self):
        if self.nclasses==2:
            gprange = [1]
        else:
            gprange = np.arange(self.nclasses)
        self.oldkappa = np.exp(self.lnkappa)            
        self.lnjoint(alldata=True)
        # If we have not already calculated lnpCT for the lower bound, then make sure we recalculate using all data
#         lnpCT = super(HeatMapBCC, self).post_lnjoint_ct()
        lnpCT = 0
        for j in gprange:
            lnpCTj = self.lnpCT[:, j] #- self.lnkappa[j] + self.heatGP[j].logpz(1.0).flatten()
            lnpCT += np.sum(np.multiply(self.E_t[self.testidxs, j], lnpCTj))
        if self.nclasses==2:
            lnpCTj = self.lnpCT[:, 0] #- self.lnkappa[0] + self.heatGP[1].logpz(0.0).flatten()
            lnpCT += np.sum(np.multiply(self.E_t[self.testidxs, 0], lnpCTj))
            
        return lnpCT

    def q_ln_t(self):
        if not self.uselowerbound:
            self.lnjoint(alldata=True)
        
#         qjoint = self.lnpCT - self.lnkappa.T
#         qjoint = qjoint[self.testidxs, :]        
#         if self.nclasses==2:
#             gprange = [1]
#         else:
#             gprange = np.arange(self.nclasses)
#         for j in gprange:
#             pz = self.heatGP[j].logpz(1.0)
#             qjoint[:, j] += pz.flatten()
#         if self.nclasses==2:
#             pz = self.heatGP[1].logpz(0.0)
#             qjoint[:, 0] += pz.flatten()
#              
#         # ensure that the values are not too small
#         largest = np.max(qjoint, 1)[:, np.newaxis]
#         qjoint = qjoint - largest
#         qjoint = np.exp(qjoint)
#         norma = np.sum(qjoint, axis=1)[:, np.newaxis]
#         qT = qjoint / norma
#         lnqT = np.sum( np.multiply(self.E_t[self.testidxs, :], np.log(qT)))
        lnqT = np.sum( np.multiply(self.E_t[self.testidxs, :], np.log(self.E_t[self.testidxs, :]) ) ) #we may need to replace E_t with an expectation WRT the approximation
#         lnqT = 0
#         if self.nclasses==2:
#             gprange = [1]
#         else:
#             gprange = np.arange(self.nclasses)        
#         for j in gprange:
#             lnqT += self.heatGP[j].logpy()
        return lnqT
                
    def post_lnkappa(self):
        lnpKappa = 0
        for j in range(self.nclasses):
            if j in self.heatGP:
                lnpKappa += self.heatGP[j].logps() + self.heatGP[j].logpf()
        return lnpKappa                
                
    def q_lnkappa(self):
        lnqKappa = 0
        for j in range(self.nclasses):
            if j in self.heatGP:
                lnqKappa += self.heatGP[j].logqs() + self.heatGP[j].logqf()
        return lnqKappa
    
    def ln_modelprior(self):
        # get the prior over the alpha0 and nu0 hyper-paramters
        lnp = super(HeatMapBCC,self).ln_modelprior()
        #Gamma distribution over each value. Set the parameters of the gammas.
        lnp_gp = 0
        for j in range(self.nclasses):
            if j in self.heatGP:
                lnp_gp += self.heatGP[j].ln_modelprior()
                logging.debug("GP for class %i using length-scale %f" % (j, self.heatGP[j].ls))
        return lnp + lnp_gp
    
    def set_hyperparams(self, hyperparams):
        if not self.optimize_lengthscale_only:
            ibcc_hyperparams = hyperparams[0:self.nclasses * self.nscores * self.alpha0_length + self.nclasses]
            super(HeatMapBCC, self).set_hyperparams(ibcc_hyperparams)
        
        if self.n_lengthscales==1:
            self.ls_initial[:] = np.exp(hyperparams[ - self.n_lengthscales])
        elif self.n_lengthscales==2:
            self.ls_initial[0] = np.exp(hyperparams[ - self.n_lengthscales])
            self.ls_initial[1] = np.exp(hyperparams[ - self.n_lengthscales + 1])
        lengthscales = self.ls_initial
        logging.debug("Length-scale = %.3f, %.3f" % (self.heatGP[1].ls[0], self.heatGP[1].ls[1]))    
        return self.alpha0, self.nu0, lengthscales

    def get_hyperparams(self):
        if not self.optimize_lengthscale_only:
            hyperparams = super(HeatMapBCC, self).get_hyperparams()
        else:
            hyperparams = []
        for j in range(self.n_lengthscales):
            for gpidx in self.heatGP:
                for d, ls in enumerate(self.heatGP[gpidx].ls):
                    if ls == 1:
                        logging.warning("Changing length-scale of 1 to 2 to avoid optimisation problems.")
                        self.heatGP[1].ls[d] = 2.0            
                hyperparams.append(np.log(self.heatGP[gpidx].ls[j]))
        return hyperparams
