
import ibcc
from gpgrid import GPGrid
import numpy as np
import logging
from scipy.special import psi
from scipy.stats import gamma

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
    sd_kappa = [] # posterior SD over kappa
    
    heatGP = [] # spatial GP model for kappa 
    gp_hyperparams = {}
    gam_shape_gp = []
    gam_scale_gp = []
    
    obsx = [] # x-coordinates of locations with crowd reports
    obsy = [] # y-coordinates of locations with crowd reports
    
    crowdx = [] # ordered list of x-coordinates of crowd reports 
    crowdy = [] # ordered list of y-coordinates of crowd reports
    
    outputx = [] # coordinates of output points from the heat-map. If you just want to evaluate the whole grid, leave
    outputy = [] # these as empty lists
    
    def __init__(self, nx, ny, nclasses, nscores, alpha0, nu0, K, force_update_all_points=False, outputx=[], outputy=[],
                 gp_hyperparams={'s':4, 'ls':100}):
        self.nx = nx
        self.ny = ny
        self.outputx = outputx
        self.outputy = outputy
        self.N = nx*ny
        self.lnkappa = []
        self.post_T = []
        self.update_all_points = force_update_all_points
        self.gp_hyperparams = gp_hyperparams
        logging.debug('Setting up a 2-D grid. This should be generalised!')     
        super(HeatMapBCC, self).__init__(nclasses, nscores, alpha0, nu0, K) 
    
    def desparsify_crowdlabels(self, crowdlabels):
        crowdlabels = np.array(crowdlabels)
        self.crowdx = crowdlabels[:,1].astype(int)
        self.crowdy = crowdlabels[:,2].astype(int)        
        linearIdxs = np.ravel_multi_index((self.crowdx, self.crowdy), dims=(self.nx,self.ny))
        crowdlabels_flat = crowdlabels[:,[0,1,3]]
        crowdlabels_flat[:,1] = linearIdxs
        crowdlabels = super(HeatMapBCC,self).desparsify_crowdlabels(crowdlabels_flat)
        self.full_N = self.N # make sure that when we re-sparsify, we expand to the full grid size
        linearIdxs = np.unique(linearIdxs)
        self.obsx, self.obsy = np.unravel_index(linearIdxs, dims=(self.nx,self.ny))
        self.nu = [] # make sure we reset kappa so that it is resized correctly -- could avoid this in future to save a few iterations
        return crowdlabels_flat
    
    def preprocess_goldlabels(self, goldlabels=None):
        if np.any(goldlabels):
            goldlabels_flat = goldlabels.flatten()
        else:
            goldlabels_flat = None
        super(HeatMapBCC, self).preprocess_goldlabels(goldlabels_flat)
        
    def createGP(self):
        #function can be overwritten by subclasses
        return GPGrid(self.nx, self.ny, force_update_all_points=self.update_all_points, s=self.gp_hyperparams['s'],
                        ls=self.gp_hyperparams['ls'])        
        
    def init_lnkappa(self):
        super(HeatMapBCC, self).init_lnkappa()  
        
        self.nu = np.tile(self.nu, (1, self.N))
        self.lnkappa = np.tile(self.lnkappa, (1, self.N))
        
        # Initialise the underlying GP with the current set of hyper-parameters
        self.heatGP = {}           
        if self.nclasses==2:
            gprange = [1]
        else:
            gprange = np.arange(self.nclasses)
        for j in gprange:
            #start with a homogeneous grid     
            self.heatGP[j] = self.createGP()

    def combine_classifications(self, crowdlabels, goldlabels=None, testidxs=None, optimise_hyperparams=False, table_format=False):
        if self.table_format_flag:
            logging.error('Error: must use a sparse list of crowdsourced labels for HeatMapBCC')
            return []
        elif crowdlabels.shape[1] != 4:
            logging.error('Must use a sparse list of crowdsourced labels with 4 columns:')
            logging.error('Agent ID, x-cood, y-coord, response value') 
            return []
        return super(HeatMapBCC, self).combine_classifications(crowdlabels, goldlabels, testidxs, optimise_hyperparams, False)
        
    def resparsify_t(self):       
        self.sd_kappa = {}
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
                self.lnkappa_out[j,:] = self.heatGP[j].predict([self.outputx, self.outputy])
                self.sd_kappa[j] = self.heatGP[j].v
            if self.nclasses==2:
                self.sd_kappa[0] = self.sd_kappa[1]        
            E_t_full[:,:] = (np.exp(self.lnkappa_out) / np.sum(np.exp(self.lnkappa_out),axis=0))
            #observation points that coincide with output points should take into account the labels, not just GP
            obsout_idxs = np.argwhere(np.in1d(self.obsx, self.outputx, assume_unique=True))
            E_t_full[:, obsout_idxs] = self.E_t.T[:,obsout_idxs]
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
                lnkappa_grid_j = self.heatGP[j].predict([outputx, outputy])
                self.lnkappa_grid[j:,:] = lnkappa_grid_j.reshape((self.nx, self.ny))
                self.sd_kappa[j] = self.heatGP[j].v.reshape((self.nx, self.ny))
            if self.nclasses==2:
                self.sd_kappa[0] = self.sd_kappa[1]
            E_t_full[:] = (np.exp(self.lnkappa_grid) / np.sum(np.exp(self.lnkappa_grid),axis=0))
            E_t_full[:,self.obsx, self.obsy] = self.E_t.T
        self.E_t_sparse = self.E_t  # save the sparse version
        self.E_t = E_t_full                   
        return self.E_t
    
    def get_mean_kappa(self, j=1):
        return self.mean_kappa[j]

    def get_sd_kappa(self, j=1):
        return self.sd_kappa[j] 
# 
#     def nu_to_latent(self, j=1):
#         mean = self.nu0 / np.sum(self.nu0)
#         fmean = logit(mean, self.heatGP[j].s)
#         
#         totalnu = np.sum(self.nu0)
#         var = self.nu0*(totalnu=self.nu0) / (totalnu**2 * (totalnu+1))
#         
#         fvar = 

    def kappa_moments_to_nu(self, mean_kappa, sd_kappa):
        total_nu = mean_kappa*(1-mean_kappa)/(sd_kappa**2) - 1
        nu_j = total_nu*mean_kappa
        return nu_j, total_nu
    
    def beta_var(self, nu0, nu1):
        var = nu0*nu1 / ((nu0+nu1)**2 * (nu0+nu1+1))
        return var

    def expec_lnkappa(self):
        if self.nclasses==2:
            nu_rest = []
            gprange = [1]
        else:
            gprange = np.arange(self.nclasses)
        for j in gprange:
            obs_values = self.E_t[:,j]
            self.lnkappa[j] = self.heatGP[j].fit([self.obsx, self.obsy], obs_values)
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
    
    def ln_modelprior(self):
        # get the prior over the alpha0 and nu0 hyper-paramters
        lnp = super(HeatMapBCC,self).ln_modelprior()
        # get the prior over the GP hyper-parameters
        # inner length scale, 
        # outer length scale, 
        # sigmoid scale, s
        #Check and initialise the hyper-hyper-parameters if necessary
        if self.gam_scale_gp==[]:
            self.gam_shape_gp = self.gam_shape_gp.astype(float)
            # if the scale was not set, assume current values of alpha0 are the means given by the hyper-prior
            self.gam_scale_gp = self.gp_hyperparams.values()/self.gam_shape_gp
        #Gamma distribution over each value. Set the parameters of the gammas.
        lnp_gp = np.sum(gamma.logpdf(self.gp_hyperparams.values(), self.gam_shape_gp, scale=self.gam_scale_gp))
        return lnp + lnp_gp
