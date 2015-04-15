
import ibcc
from gpgrid import GPGrid
import numpy as np
import logging
from scipy.special import psi
from scipy.stats import gamma

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
    
    def __init__(self, nx, ny, nclasses, nscores, alpha0, nu0, K, calc_full_grid=False, gp_hyperparams={'s':4, 'ls':100}):
        self.nx = nx
        self.ny = ny
        self.N = nx*ny
        self.lnkappa = []
        self.post_T = []
        self.calc_full_grid = calc_full_grid
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
        return crowdlabels_flat
    
    def preprocess_goldlabels(self, goldlabels=None):
        if np.any(goldlabels):
            goldlabels_flat = goldlabels.flatten()
        else:
            goldlabels_flat = None
        super(HeatMapBCC, self).preprocess_goldlabels(goldlabels_flat)
        
    def createGP(self):
        #function can be overwritten by subclasses
        return GPGrid(self.nx, self.ny, calc_full_grid=self.calc_full_grid, s=self.gp_hyperparams['s'], ls=self.gp_hyperparams['ls'])        
        
    def init_lnkappa(self):
        super(HeatMapBCC, self).init_lnkappa()  
        
        self.nu = np.tile(self.nu, (1, self.N))
        self.lnkappa = np.tile(self.lnkappa, (1, self.N))
        
        # Initialise the underlying GP with the current set of hyper-parameters
        self.heatGP = {}           
        for j in range(1, self.nclasses):
            #start with a homogeneous grid     
            self.heatGP[j] = self.createGP()

    def combine_classifications(self, crowdlabels, goldlabels=None, testidxs=None, optimise_hyperparams=False, table_format=False):
        if self.table_format_flag:
            #goldlabels = np.zeros(crowdlabels.shape[0]) -1
            logging.error('Error: must use a sparse list of crowdsourced labels for HeatMapBCC')
            return []
        elif crowdlabels.shape[1] != 4:
            logging.error('Must use a sparse list of crowdsourced labels with 4 columns:')
            logging.error('Agent ID, x-cood, y-coord, response value') 
            return []
        return super(HeatMapBCC, self).combine_classifications(crowdlabels, goldlabels, testidxs, optimise_hyperparams, False)
        
    def resparsify_t(self):       
        self.lnkappa = np.zeros((self.nclasses, self.nx, self.ny))
        self.nu = np.zeros((self.nclasses, self.nx, self.ny))
        
        self.sd_kappa = {}
        self.mean_kappa = {}
        self.mean_kappa[0] = np.ones((self.nx,self.ny))
        
        nu_rest = []        
        for j in range(1,self.nclasses):
            mean_kappa, sd_kappa = self.heatGP[j].post_grid()
            self.nu[j,:,:], total_nu = self.kappa_moments_to_nu(mean_kappa, sd_kappa)
            self.nu[j,:,:] += self.nu0[j]
            total_nu += np.sum(self.nu0)
            self.lnkappa[j,:,:] = psi(self.nu[j,:,:]) - psi(total_nu)
            self.sd_kappa[j] = sd_kappa
            self.mean_kappa[j] = mean_kappa
            self.mean_kappa[0] -= mean_kappa
            if nu_rest==[]:
                nu_rest = total_nu
            nu_rest = nu_rest - self.nu[j,:,:]
                
        self.nu[0,:,:] = nu_rest             
        self.lnkappa[0,:,:] = psi(nu_rest) - psi(total_nu)
        self.sd_kappa[0] = np.sqrt(self.beta_var(self.nu[0,:,:], total_nu-self.nu[0,:,:]))        
        
        E_t_full = np.zeros((self.nclasses, self.nx, self.ny))
        E_t_full[:] = (np.exp(self.lnkappa) / np.sum(np.exp(self.lnkappa),axis=0))
        E_t_full[:,self.obsx, self.obsy] = self.E_t.T
        self.E_t_sparse = self.E_t  # save the sparse version
        self.E_t = E_t_full                   
        return self.E_t
    
    def get_mean_kappa(self, j=1):
        return self.mean_kappa[j]

    def get_sd_kappa(self, j=1):
        return self.sd_kappa[j] 

    def kappa_moments_to_nu(self, mean_kappa, sd_kappa):
        total_nu = mean_kappa*(1-mean_kappa)/(sd_kappa**2) - 1
        nu_j = total_nu*mean_kappa
        return nu_j, total_nu
    
    def beta_var(self, nu0, nu1):
        var = nu0*nu1 / ((nu0+nu1)**2 * (nu0+nu1+1))
        return var

    def expec_lnkappa(self):
        nu_rest = []
        
        for j in range(1,self.nclasses):
            obsPoints = self.E_t[:,j]
            mean_kappa, sd_kappa = self.heatGP[j].train(self.obsx, self.obsy, obsPoints)    
            #convert to pseudo-counts
            nu_j, total_nu = self.kappa_moments_to_nu(mean_kappa, sd_kappa)
            nu_j += self.nu0[j] - 1
            total_nu += np.sum(self.nu0) - 2
            nu_j = nu_j.flatten()
            self.nu[j] = nu_j
            if nu_rest==[]:
                nu_rest = total_nu
            nu_rest -= nu_j
            
            self.lnkappa[j] = psi(nu_j)-psi(total_nu)#np.log(mean_kappa)
        nu_rest = nu_rest.flatten()
        self.nu[0] = nu_rest
        self.lnkappa[0] = psi(nu_rest)-psi(total_nu)#np.log(kappa_rest)
     
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