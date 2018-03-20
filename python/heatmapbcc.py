
import ibcc
# from gp_classifier_vb import GPClassifierVB
from gp_classifier_svi import GPClassifierSVI
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
    
    outputx = [] # coordinates of output points from the heat-map. If you just want to evaluate the whole grid, leave
    outputy = [] # these as empty lists

    lnkappa_out = [] # lnkappa at the output points given by the GP
    
    oldkappa = [] # density mean from previous VB iteration

    E_t_sparse = [] # target value estimates at the training points only.
    
    optimize_lengthscale_only = True
    
    n_lengthscales = 1
    
    conv_count = 0 # count the number of iterations where the change was below the convergence threshold

    def __init__(self, nclasses, nscores, alpha0, K, z0=[0.5, 0.5], shape_s0=2, rate_s0=2, shape_ls=10,
                 rate_ls=0.1, kernel_func='matern_3_2'):
        '''
        Class for heatmapBCC. The arguments that are typically needed are described as follows.
        
        Parameters
        ----------
        
        nx : int
            size of area of interest in x dimension
        ny : int
            size of area of interest in y dimension
        nclasses : int
            no. ground truth classes
        nscores : int
            no. discrete values that we can observe from the information sources
        alpha0 : n_classes x n_scores x K numpy array
            confusion matrix hyperparameters
        K : int
            no. information sources
        z0=0.5 : float or n_observations x 1 numpy array 
            prior mean probability of each class, either a scalar (constant value across the area of interest) or
        a list of length n_observations, where n_observations is the number of observed report locations
        shape_s0=2 : float
            shape hyperparameter for the Gamma prior over the GP precision
        rate_s0=2 : float
            rate hyperparameter for the Gamma prior over the GP precision
        '''
        self.lnkappa = []
        self.post_T = []
        self.z0 = z0
        if np.isscalar(self.z0):
            self.z0 = np.ones(self.nclasses) * z0

        self.shape_s0 = shape_s0
        self.rate_s0 = rate_s0
        self.shape_ls = shape_ls
        self.rate_ls = rate_ls
        self.ls_initial = self.shape_ls / self.rate_ls + np.zeros(nclasses)
        self.kernel_func = kernel_func
        nu0 = np.ones(nclasses)
        super(HeatMapBCC, self).__init__(nclasses, nscores, alpha0, nu0, K) 
    
    def _init_lnkappa(self):
        super(HeatMapBCC, self)._init_lnkappa()  
        self.lnkappa = np.tile(self.lnkappa, (1, self.N))
        
        # Initialise the underlying GP with the current set of hyper-parameters           
        if self.nclasses==2:
            gprange = [1]
        else:
            gprange = np.arange(self.nclasses)
        self.oldkappa = np.exp(self.lnkappa)
        for j in gprange:
            self.heatGP[j] = GPClassifierSVI(self.feat_vecs.shape[1], shape_ls=self.shape_ls, rate_ls=self.rate_ls,
                                             ls_initial=self.ls_initial, kernel_func=self.kernel_func)
            self.heatGP[j].verbose = self.verbose
            self.heatGP[j].max_iter_VB = 1
            self.heatGP[j].min_iter_VB = 1
            self.heatGP[j].max_iter_G = 1
            self.heatGP[j].uselowerbound = False # we calculate the lower bound here instead of the GPClassifierVB function
            logging.debug("Length-scale = " + str(self.heatGP[1].ls))
        
    def _convergence_measure(self, oldET):
        kappadiff = np.max(np.abs( self.oldkappa - np.exp(self.lnkappa)))
        sdiff = 0
        for j in self.heatGP:
            sdiff_j = np.abs( self.heatGP[j].old_s - self.heatGP[j].s ) #/ self.heatGP[j].old_s
            if sdiff_j > sdiff:
                sdiff = sdiff_j
            if self.verbose:
                logging.debug('sdiff: %f, s = %f' % (sdiff, self.heatGP[j].s))
        return np.max( [super(HeatMapBCC, self)._convergence_measure(oldET), kappadiff, sdiff])
    
    def _convergence_check(self):
        if self.change<self.conv_threshold:
            self.conv_count += 1
        locally_converged = super(HeatMapBCC, self)._convergence_check()
        if not locally_converged:
            return False
        
        if self.nIts>self.min_iterations:
            return True
        else:
            return False

    def combine_classifications(self, crowdlabels, feature_vectors, goldlabels=None, testidxs=None, optimise_hyperparams=False,
                                maxiter=200, table_format=False):
        '''
        Combine a set of noisy reports to train a GP and make predictions at the observed locations
        
        Parameters
        ----------
        
        crowdlabels : n_obervations x 3 numpy array
            The noisy crowdsourced reports with 3 columns: information source ID, item ID, report value
        feature_vectors : n_items x (n_features+1) numpy array
            The feature vectors for the items, e.g. geo-coordinates. The first column should contain the item IDs, the
            following columns the feature values.
        goldlabels : n_observations x 1 numpy array
            If training labels are available, pass in as a vector with -1 where unavailable.
        optimise_hyperparams : bool
            Optimise the lengthscale
            
        Returns
        -------
        
        E_t : n_observations x n_classes numpy array
            Predictions at the observed locations, where each column indicates probability of corresponding class
        '''
        self.feat_vecs = feature_vectors

        super(HeatMapBCC, self).combine_classifications(crowdlabels, goldlabels, testidxs, optimise_hyperparams,
                                                               maxiter, False)

        return self.E_t

    def _desparsify_crowdlabels(self, crowdlabels):
        crowdlabels = super(HeatMapBCC, self)._desparsify_crowdlabels(crowdlabels)

        idxs = [np.argwhere(self.feat_vecs[:, 0] == i)[0][0] for i in self.observed_idxs]
        self.feat_vecs = self.feat_vecs[idxs, 1:]

        return crowdlabels

    def _expec_lnkappa(self):
        if self.nclasses==2:
            gprange = [1]
        else:
            gprange = np.arange(self.nclasses)
        self.oldkappa = np.exp(self.lnkappa)            
        for j in gprange:
            obs_values = self.E_t[:,j]                
            
            # only process the data on the first iteration, otherwise variables get reset
            self.heatGP[j].fit(self.feat_vecs, np.round(obs_values), process_obs=self.nIts==0)
            
            self.heatGP[j].verbose = False
            lnkappaj, lnkappa_notj, _ = self.heatGP[j].predict(self.feat_vecs, variance_method='sample',
                                                        expectedlog=True)
            self.heatGP[j].verbose = self.verbose
            self.lnkappa[j] = lnkappaj.flatten()
        if self.nclasses==2:
            self.lnkappa[0] = lnkappa_notj.flatten()
            
    def _lnjoint(self, alldata=False):
        lnkappa_all = self.lnkappa 
        if not self.uselowerbound and not alldata:       
            self.lnkappa = self.lnkappa[:, self.testidxs]
        super(HeatMapBCC, self)._lnjoint(alldata)
        self.lnkappa = lnkappa_all

    def _post_lnjoint_ct(self):
        if self.nclasses==2:
            gprange = [1]
        else:
            gprange = np.arange(self.nclasses)
        self.oldkappa = np.exp(self.lnkappa)            
        self._lnjoint(alldata=True)
        # If we have not already calculated lnpCT for the lower bound, then make sure we recalculate using all data
        lnpCT = 0
        for j in gprange:
            lnrhoj, lnnotrhoj = self.heatGP[j]._logpt()
            lnpCTj = self.lnpCT[:, j] - self.lnkappa[j] + lnrhoj.flatten() # this operation is required because we
            # need the p(C, T | f) since the terms p(f) and q(f) will also be included in the lower bound. In contrast,
            # self.lnpCT integrates out f.
            lnpCT += np.sum(np.multiply(self.E_t[self.testidxs, j], lnpCTj))
        if self.nclasses==2:
            lnpCTj = self.lnpCT[:, 0] - self.lnkappa[0] + lnnotrhoj.flatten()
            lnpCT += np.sum(np.multiply(self.E_t[self.testidxs, 0], lnpCTj))
            
        return lnpCT

    def _q_ln_t(self):
        if not self.uselowerbound:
            self._lnjoint(alldata=True)
        
        lnqT = np.sum( np.multiply(self.E_t[self.testidxs, :], np.log(self.E_t[self.testidxs, :]) ) ) #we may need to replace E_t with an expectation WRT the approximation
        return lnqT
                
    def _post_lnkappa(self):
        lnpKappa = 0
        for j in range(self.nclasses):
            if j in self.heatGP:
                lnpKappa += self.heatGP[j]._logps() + self.heatGP[j]._logpf()
        return lnpKappa                
                
    def _q_lnkappa(self):
        lnqKappa = 0
        for j in range(self.nclasses):
            if j in self.heatGP:
                lnqKappa += self.heatGP[j]._logqs() + self.heatGP[j]._logqf()
        return lnqKappa

    def predict(self, feat_vecs, variance_method='rough'):
        '''
        Predict class at new locations.

        Parameters
        ----------

        feat_vecs: numpy array
            Matrix of the feature vectors of output points, where each row is a feature vector or set of coordinates for
            a single point. The first column should contain an ID for the data points. IDs that were seen in training
            will reuse the predictions from the training phase.

        Returns
        -------

        E_t: N x nclasses numpy array
            Predictions at the output locations, where each column indicates probability of corresponding class
        E_rho: N x nclasses numpy array
            Prediction of the underlying class probabilities at each output location
        V_rho: N x nclasses numpy array
            Variance of the underlying class probabilities at each location, useful for indicating regions of model uncertainty

        '''
        # Initialise containers for results at the output locations
        nout = feat_vecs.shape[0]
        self.E_t_out = np.zeros((self.nclasses, nout))
        kappa_out = np.zeros((self.nclasses, nout))
        v_kappa_out = np.zeros((self.nclasses, nout))

        # Obtain the density estimates
        if self.nclasses == 2:
            gprange = [1]
        else:
            gprange = np.arange(self.nclasses)

        for j in gprange:
            mean_kappa, v_kappa = self.heatGP[j].predict(feat_vecs, variance_method=variance_method)
            kappa_out[j, :] = mean_kappa.flatten()
            v_kappa_out[j, :] = v_kappa.flatten()
        if self.nclasses == 2:
            kappa_out[0, :] = 1 - kappa_out[1, :]
            v_kappa_out[0, :] = v_kappa_out[1, :]

        # Set the predictions for the targets
        self.E_t_out[:, :] = kappa_out
        # observation points that coincide with output points should take into account the labels, not just GP
        object_ids = feat_vecs[:, 0]
        obsin_idxs = np.array([np.argwhere(self.observed_idxs==i)[0][0] if i in self.observed_idxs else -1 for i in object_ids], dtype=int)
        obsout_idxs = obsin_idxs > -1
        obsin_idxs = obsin_idxs[obsout_idxs]
        self.E_t_out[:, obsout_idxs] = self.E_t.T[:, obsin_idxs]
        return self.E_t_out, kappa_out, v_kappa_out

    def predict_grid(self, nx, ny):
        """
        Assumes that there are two feature dimensions only.
        :param nx:
        :param ny:
        :return:
        """

        E_t_grid = np.zeros((self.nclasses, nx, ny))
        kappa_grid = np.zeros((self.nclasses, nx, ny))
        v_kappa_grid = np.zeros((self.nclasses, nx, ny))
        # Evaluate the function posterior mean and variance at all coordinates in the grid. Use this to calculate
        # values for plotting a heat map. Calculate coordinates:
        nout = nx * ny
        outputx = np.tile(np.arange(nx, dtype=np.float).reshape(nx, 1), (1, ny)).reshape(nout, 1)
        outputy = np.tile(np.arange(ny, dtype=np.float).reshape(1, ny), (nx, 1)).reshape(nout, 1)

        if self.nclasses == 2:
            gprange = [1]
        else:
            gprange = np.arange(self.nclasses)
        outputxy = np.concatenate((outputx, outputy), axis=1)
        for j in gprange:
            kappa_grid_j, v_kappa_grid_j = self.heatGP[j].predict(outputxy)
            kappa_grid[j, :, :] = kappa_grid_j.reshape((nx, ny))
            v_kappa_grid[j, :, :] = v_kappa_grid_j.reshape((nx, ny))
        if self.nclasses == 2:
            kappa_grid[0, :, :] = 1 - kappa_grid[1, :, :]
            v_kappa_grid[0, :, :] = v_kappa_grid[1, :, :]

        E_t_grid[:] = kappa_grid

        obs_at_grid_points = (np.mod(self.feat_vecs[:, 1], 1) == 0) & (np.mod(self.feat_vecs[:, 2], 1) == 0)
        obsx_grid = self.feat_vecs[obs_at_grid_points, 1].astype(int)
        obsy_grid = self.feat_vecs[obs_at_grid_points, 2].astype(int)
        E_t_grid[:, obsx_grid, obsy_grid] = self.E_t[obs_at_grid_points, :].T

        return E_t_grid, kappa_grid, v_kappa_grid

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
    
    def _set_hyperparams(self, hyperparams):
        if not self.optimize_lengthscale_only:
            ibcc_hyperparams = hyperparams[0:self.nclasses * self.nscores * self.alpha0_length + self.nclasses]
            super(HeatMapBCC, self)._set_hyperparams(ibcc_hyperparams)
        
        if self.n_lengthscales==1:
            self.ls_initial[:] = np.exp(hyperparams[ - self.n_lengthscales])
        elif self.n_lengthscales==2:
            self.ls_initial[0] = np.exp(hyperparams[ - self.n_lengthscales])
            self.ls_initial[1] = np.exp(hyperparams[ - self.n_lengthscales + 1])
        lengthscales = self.ls_initial    
        return self.alpha0, self.nu0, lengthscales

    def _get_hyperparams(self):
        if not self.optimize_lengthscale_only:
            hyperparams = super(HeatMapBCC, self)._get_hyperparams()
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
