
import ibcc
from gpgrid import GPGrid
import numpy as np
import logging
from scipy.stats import gamma, norm

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
    
    heatGP = [] # spatial GP model for kappa 
    
    obsx = [] # x-coordinates of locations with crowd reports
    obsy = [] # y-coordinates of locations with crowd reports
    
    crowdx = [] # ordered list of x-coordinates of crowd reports 
    crowdy = [] # ordered list of y-coordinates of crowd reports
    
    outputx = [] # coordinates of output points from the heat-map. If you just want to evaluate the whole grid, leave
    outputy = [] # these as empty lists
    
    oldf = [] # latent function mean from previous VB iteration 
    
    optimize_lengthscale_only = True 
    
    def __init__(self, nx, ny, nclasses, nscores, alpha0, K, z0=0.5, shape_s0=0.01, rate_s0=0.01, shape_ls=10, \
                 rate_ls=0.1, force_update_all_points=False, outputx=[], outputy=[]):
        self.nx = nx
        self.ny = ny
        self.outputx = outputx
        self.outputy = outputy
        self.N = nx*ny
        self.lnkappa = []
        self.post_T = []
        self.update_all_points = force_update_all_points
        self.z0 = z0
        self.shape_s0 = shape_s0
        self.rate_s0 = rate_s0
        self.shape_ls = shape_ls
        self.rate_ls = rate_ls
        self.ls_initial = self.shape_ls / self.rate_ls + np.ones(nclasses)
        logging.debug('Setting up a 2-D grid. This should be generalised!')
        nu0 = np.ones(nclasses)     
        super(HeatMapBCC, self).__init__(nclasses, nscores, alpha0, nu0, K) 
    
    def desparsify_crowdlabels(self, crowdlabels):
        crowdlabels = np.array(crowdlabels)
        self.crowdx = crowdlabels[:,1]
        self.crowdy = crowdlabels[:,2]

        crowdcoords = zip(self.crowdx, self.crowdy)
        self.crowddict = dict((coord, idx) for coord, idx in zip(crowdcoords, range(len(crowdcoords))) )
        linearIdxs = [self.crowddict[l] for l in crowdcoords] # do this to ensure that we get unique values for each coord
        crowdlabels_flat = crowdlabels[:,[0,1,3]]
        crowdlabels_flat[:,1] = linearIdxs
        crowdlabels = super(HeatMapBCC,self).desparsify_crowdlabels(crowdlabels_flat)
        self.full_N = self.N # make sure that when we re-sparsify, we expand to the full grid size
        linearIdxs = self.crowddict.values()
        self.obsx, self.obsy = np.unravel_index(linearIdxs, dims=(self.nx,self.ny))
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
        self.heatGP = {}           
        if self.nclasses==2:
            gprange = [1]
        else:
            gprange = np.arange(self.nclasses)
        self.oldf = {}
        for j in gprange:
            #start with a homogeneous grid     
            self.heatGP[j] = GPGrid(self.nx, self.ny, z0=self.z0, shape_s0=self.shape_s0, rate_s0=self.rate_s0, 
                                    shape_ls=self.shape_ls, rate_ls=self.rate_ls,  ls_initial=self.ls_initial[j],
                                    force_update_all_points=self.update_all_points)   
            self.heatGP[j].verbose = self.verbose
            self.heatGP[j].max_iter_VB = 1 # do one update each time we call fit()

    def convergence_measure(self, oldET):
        if self.nclasses==2:
            gprange = [1]
        else:
            gprange = np.arange(self.nclasses)
        fdiff = np.zeros(self.nclasses)
        for j in gprange:
            #start with a homogen
            fdiff[j] = np.max(np.abs( (self.oldf[j] - self.heatGP[j].obs_f) / (self.heatGP[j].obs_f+self.oldf[j])))
        return np.max( [np.max(np.sum(np.abs(oldET - self.E_t), 1)), np.max(fdiff)])

    def combine_classifications(self, crowdlabels, goldlabels=None, testidxs=None, optimise_hyperparams=False, 
                                maxiter=200, table_format=False):
        if self.table_format_flag:
            logging.error('Error: must use a sparse list of crowdsourced labels for HeatMapBCC')
            return []
        elif crowdlabels.shape[1] != 4:
            logging.error('Must use a sparse list of crowdsourced labels with 4 columns:')
            logging.error('Agent ID, x-cood, y-coord, response value') 
            return []
        return super(HeatMapBCC, self).combine_classifications(crowdlabels, goldlabels, testidxs, optimise_hyperparams, 
                                                               maxiter, False)
        
    def resparsify_t(self):
        self.var_logodds_kappa = {}
        self.mean_kappa = {}                
        if self.nclasses==2:
            gprange = [1]
        else:
            gprange = np.arange(self.nclasses)
                        
        if self.outputx != []:
            logging.debug("Resparsifying to specified output points")        
            nout = len(self.outputx)
            
            self.mean_kappa[0] = np.ones(nout)
            E_t_full = np.zeros((self.nclasses, nout))
            
            self.lnkappa_out = np.zeros((self.nclasses, nout))
            self.nu_out = np.zeros((self.nclasses, nout))

            for j in gprange:
                self.lnkappa_out[j,:] = self.heatGP[j].predict([self.outputx, self.outputy], expectedlog=True)
                self.var_logodds_kappa[j] = self.heatGP[j].v
            if self.nclasses==2:
                self.var_logodds_kappa[0] = self.var_logodds_kappa[1]
                self.lnkappa_out[0,:] = np.log(1 - np.exp(self.lnkappa_out[1,:]))        
            E_t_full[:,:] = np.exp(self.lnkappa_out)
            #observation points that coincide with output points should take into account the labels, not just GP
            outputcoords = zip(self.outputx, self.outputy)
            obsout_idxs = np.argwhere(np.in1d(outputcoords, self.crowddict, assume_unique=True))
            obsin_idxs = [self.crowddict[outputcoords[idx]] for idx in obsout_idxs]
            if len(obsin_idxs):
                E_t_full[:, obsout_idxs] = self.E_t.T[:,obsin_idxs]
        else:
            logging.debug("Resparsifying to grid")
            E_t_full = np.zeros((self.nclasses, self.nx, self.ny))            
            
            self.lnkappa_grid = np.zeros((self.nclasses, self.nx, self.ny))
            #Evaluate the function posterior mean and variance at all coordinates in the grid. Use this to calculate
            #values for plotting a heat map. Calculate coordinates:
            nout = self.nx * self.ny
            outputx = np.tile(np.arange(self.nx, dtype=np.float).reshape(self.nx, 1), (1, self.ny)).reshape(nout, 1)
            outputy = np.tile(np.arange(self.ny, dtype=np.float).reshape(1, self.ny), (self.nx, 1)).reshape(nout, 1)
            for j in gprange:
                lnkappa_grid_j = self.heatGP[j].predict([outputx, outputy], expectedlog=True)
                self.lnkappa_grid[j:,:] = lnkappa_grid_j.reshape((self.nx, self.ny))
                self.var_logodds_kappa[j] = self.heatGP[j].v.reshape((self.nx, self.ny))
            if self.nclasses==2:
                self.var_logodds_kappa[0] = self.var_logodds_kappa[1]
                self.lnkappa_grid[0,:,:] = np.log(1 - np.exp(self.lnkappa_grid[1,:,:]))
            E_t_full[:] = (np.exp(self.lnkappa_grid) / np.sum(np.exp(self.lnkappa_grid),axis=0))
            E_t_full[:,self.obsx, self.obsy] = self.E_t.T
        self.E_t_sparse = self.E_t  # save the sparse version
        self.E_t = E_t_full                   
        return self.E_t
    
    def get_mean_kappa(self, j=1):
        return self.mean_kappa[j]

    def get_heat_variance(self, j=1):
        return self.var_logodds_kappa[j] 

    def expec_lnkappa(self):
        if self.nclasses==2:
            gprange = [1]
        else:
            gprange = np.arange(self.nclasses)
        for j in gprange:
            obs_values = self.E_t[:,j]
            if len(self.heatGP[j].obs_f):
                self.oldf[j] = self.heatGP[j].obs_f
            else:
                self.oldf[j] = np.zeros(len(obs_values))
                
            self.lnkappa[j] = self.heatGP[j].fit([self.obsx, self.obsy], obs_values, expectedlog=True)
            if np.sum(self.oldf[j])==0:
                self.oldf[j] = self.heatGP[j].obs_mean_prob
        if self.nclasses==2:
            self.lnkappa[0] = np.log(1-np.exp(self.lnkappa[1]))
     
    def lnjoint(self, alldata=False):
        lnkappa_all = self.lnkappa 
        if not self.uselowerbound and not alldata:       
            self.lnkappa = self.lnkappa[:,self.testidxs]
        super(HeatMapBCC, self).lnjoint(alldata)
        self.lnkappa = lnkappa_all

    def q_ln_t(self):
        ET = self.E_t[:, self.obsx, self.obsy]
        lnqT = np.sum( np.multiply( ET,np.log(ET) ) )
        return lnqT
                
    def post_lnkappa(self):
        lnpKappa = 0
        for j in range(self.nclasses):
            lnpKappa += self.heatGP[j].logp_minus_logq()
        return lnpKappa                
                
    def q_lnkappa(self):
        lnqKappa = 0 # this was already included in the post_lnkappa.
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
        lengthscales = []
        for j in range(self.nclasses):
            if j in self.heatGP:
                self.ls_initial[j] = np.exp(hyperparams[-self.nclasses+j])
                lengthscales.append(self.ls_initial[j])
        return self.alpha0, self.nu0, lengthscales

    def get_hyperparams(self):
        if not self.optimize_lengthscale_only:
            hyperparams, constraints, rhobeg, rhoend = super(HeatMapBCC, self).get_hyperparams()
        else:
            constraints = []
            hyperparams = []
            rhobeg = []
            rhoend = []
        for j in range(self.nclasses):
            if j in self.heatGP:
                hyperparams.append(np.log(self.heatGP[j].ls))
                rhobeg.append(hyperparams[-1] - np.log(0.5 * self.heatGP[j].ls))
                rhoend.append(np.log(0.1 * self.heatGP[j].ls))
        return hyperparams, constraints, rhobeg, rhoend
