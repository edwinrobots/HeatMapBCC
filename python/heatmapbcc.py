
import ibcc
from gpgrid import GPGrid
import numpy as np
from scipy.special import psi, gammaln
from scipy.sparse import coo_matrix

class HeatMapBCC(ibcc.IBCC):
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
    nreportpoints = 0 # Number of report locations.
    
    nx = 0 #grid size. May be able to generalise away from 2-D grids
    ny = 0

    mean_kappa = [] # posteriors from the GP
    sd_kappa = []
    
    heatGP = []
    
    def __init__(self, nx, ny, nclasses, nscores, alpha0, nu0, K, table_format=False):
        self.nx = nx
        self.ny = ny
        self.N = nx*ny
        self.lnkappa = []
        self.post_T = []
        print 'Setting up a 2-D grid. This should be generalised!'        
        super(HeatMapBCC, self).__init__(nclasses, nscores, alpha0, nu0, K, table_format) 
    
    def combine_classifications(self, crowdlabels, train_t=None):
        super(HeatMapBCC, self).combine_classifications(crowdlabels, train_t)
        
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
        self.expec_t()   
            
        import gc
        gc.collect()
                     
        return self.E_t
    
    def get_mean_kappa(self, j=1):
        return self.mean_kappa[j]

    def get_sd_kappa(self, j=1):
        return self.sd_kappa[j]
        
    def preprocess_training(self, crowdlabels, train_t=None):
        # Is this necessary? Prevents uncertain labels from crowd!
        crowdlabels = crowdlabels.astype(int) 
        self.crowdlabels = crowdlabels
        if crowdlabels.shape[1]!=4:
            self.table_format_flag = True
        
        if train_t==None:
            if self.table_format_flag:
                #train_t = np.zeros(crowdlabels.shape[0]) -1
                print 'Error: must use a sparse list of crowdsourced labels for HeatMapBCC'
                train_t = []
            else:
                train_t = np.zeros((self.nx,self.ny)) -1#(np.max(crowdlabels[:,1]), np.max(crowdlabels[:,2])) )
        
        self.train_t = train_t
           
    def preprocess_crowdlabels(self):
        #ensure we don't have a matrix by mistake
        if not isinstance(self.crowdlabels, np.ndarray):
            self.crowdlabels = np.array(self.crowdlabels)
        if self.table_format_flag:            
            print 'Must use a sparse list of crowdsourced labels with 4 columns:'
            print 'Agent ID, x-cood, y-coord, response value'
            return        
                
        #De-duplication of the observations 
        self.C = self.crowdlabels    
        linearIdxs = np.ravel_multi_index((self.C[:,1], self.C[:,2]), dims=(self.nx,self.ny))
        linearIdxs = np.unique(linearIdxs)
        self.obsx, self.obsy = np.unravel_index(linearIdxs, dims=(self.nx,self.ny))

    def init_t(self):     
        kappa = self.nu / np.sum(self.nu, axis=0)        
        self.E_t = np.zeros((self.nclasses, self.nx, self.ny)) + kappa     
        
    def createGP(self):
        #function can be overwritten by subclasses
        return GPGrid(self.nx, self.ny)
        
    def init_lnkappa(self):
        if self.lnkappa != []:
            return
        self.nu = np.zeros((self.nclasses, self.nx, self.ny))
        self.lnkappa = np.zeros((self.nclasses, self.nx, self.ny))    
        
        self.heatGP = {}           
        for j in range(self.nclasses):
            self.nu[j,:,:] = self.nu0[j]   
            self.lnkappa[j,:,:] = psi(self.nu0[j]) - psi(np.sum(self.nu0))      
            #start with a homogeneous grid     
            if j>0:  
                self.heatGP[j] = self.createGP()

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
            obsPoints = self.E_t[j, self.obsx, self.obsy]
            mean_kappa, sd_kappa = self.heatGP[j].train(self.obsx, self.obsy, obsPoints)    
            #convert to pseudo-counts
            nu_j, total_nu = self.kappa_moments_to_nu(mean_kappa, sd_kappa)
            nu_j += self.nu0[j] - 1
            total_nu += np.sum(self.nu0) - 2
            self.nu[j, self.obsx, self.obsy] = nu_j
            if nu_rest==[]:
                nu_rest = total_nu
            nu_rest -= nu_j
            
            self.lnkappa[j, self.obsx, self.obsy] = psi(nu_j)-psi(total_nu)#np.log(mean_kappa)
            
        self.nu[0, self.obsx, self.obsy] = nu_rest
        self.lnkappa[0, self.obsx, self.obsy] = psi(nu_rest)-psi(total_nu)#np.log(kappa_rest)
     
    def expec_t(self):       
        lnjoint = np.zeros((self.nclasses, self.nx, self.ny))
        for j in range(self.nclasses):
            data = self.lnPi[j, self.C[:,3], self.C[:,0]].reshape(-1) #responses
            rows = np.array(self.C[:,1]).reshape(-1) #x co-ord
            cols = np.array(self.C[:,2]).reshape(-1) #y co-ord
            
            likelihood_j = coo_matrix((data, (rows,cols)), shape=(self.nx, self.ny)).todense()
            lnjoint[j,:,:] = likelihood_j
        
        lnjoint = lnjoint + self.lnkappa
        
        #ensure that the values are not too small
        largest = np.max(lnjoint, 0)
        lnjoint = lnjoint - largest
            
        joint = np.exp(lnjoint)
        self.E_t = joint/np.sum(joint, axis=0)
            
        train_idxs = self.train_t!=-1
        self.E_t[:, train_idxs] = 0
        for j in range(self.nclasses):            
            #training labels    
            self.E_t[:,self.train_t==j] = 1    
            
        return lnjoint
    
    def post_Alpha(self):#Posterior Hyperparams
        for j in range(self.nclasses):
            for l in range(self.nscores):
                sepcounts = (self.C[:,3]==l)*(self.E_t[j,self.C[:,1],self.C[:,2]])
                counts = np.zeros((self.K, self.C.shape[0]))
                counts[self.C[:,0], np.arange(self.C.shape[0])] = sepcounts
                counts = np.sum(counts, axis=1).reshape(-1)
                self.alpha[j,l,:] = self.alpha0[j,l,:] + counts

    def post_lnjoint_ct(self, lnjoint):
        #not quite right if there are multiple observations at one point.
        ET = self.E_t[:, self.obsx, self.obsy]
        lnjoint = lnjoint[:, self.obsx, self.obsy]
        lnpCT = np.sum(np.sum(np.sum( lnjoint*ET )))                        
        return lnpCT
    
    def post_lnkappa(self):
        n_obs = len(self.obsx)
        lnKappa_obs = self.lnkappa[:,self.obsx,self.obsy]
        nu0_obs = np.tile(self.nu0.reshape(self.nclasses,1), (1, n_obs))
        
        lnpKappa = gammaln(np.sum(nu0_obs, 0))-np.sum(gammaln(nu0_obs), 0) + \
                    np.sum( (nu0_obs-1)*lnKappa_obs, 0)            
        return np.sum(lnpKappa)
 
    def q_lnkappa(self):
        nu_obs = self.nu[:, self.obsx, self.obsy]
        lnKappa_obs = self.lnkappa[:,self.obsx,self.obsy]
        
        lnqKappa = gammaln(np.sum(nu_obs,0))-np.sum(gammaln(nu_obs), 0) + \
                    np.sum( (nu_obs-1)*lnKappa_obs, 0)
        return np.sum(lnqKappa)
        
    def q_ln_t(self):
        ET = self.E_t[:, self.obsx, self.obsy]
        lnqT = np.sum( np.multiply( ET,np.log(ET) ) )
        return lnqT
