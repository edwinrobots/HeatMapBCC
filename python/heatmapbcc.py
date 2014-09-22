
import ibcc
from gpgrid import GPGrid
import numpy as np
from scipy.special import psi, gammaln
from scipy.sparse import coo_matrix

class Heatmapbcc(ibcc.Ibcc):
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
    nReps = 0 # Number of report locations.
    
    nx = 0 #grid size. May be able to generalise away from 2-D grids
    ny = 0

    mPr = [] # posteriors from the GP
    sd_post_T = []
    
    heatGP = []
    
    def __init__(self, nx, ny, nClasses, nScores, alpha0, nu0, K, tableFormat=False):
        self.nx = nx
        self.ny = ny
        self.nObjs = nx*ny
        self.lnKappa = []
        self.post_T = []
        print 'Setting up a 2-D grid. This should be generalised!'        
        super(Heatmapbcc, self).__init__(nClasses, nScores, alpha0, nu0, K, tableFormat) 
    
    def combineClassifications(self, crowdLabels, trainT=None):
        super(Heatmapbcc, self).combineClassifications(crowdLabels, trainT)
        
        self.sd_post_T = {}
        self.mPr = {}
        self.mPr[0] = np.ones((self.nx,self.ny))
        
        nu_rest = []        
        for j in range(1,self.nClasses):
            mPr, sdPr = self.heatGP[j].post_grid()
            self.nu[j,:,:], totalNu = self.kappa_moments_to_nu(mPr, sdPr)
            self.nu[j,:,:] += self.nu0[j]
            totalNu += np.sum(self.nu0)
            self.lnKappa[j,:,:] = psi(self.nu[j,:,:]) - psi(totalNu)
            self.sd_post_T[j] = sdPr
            self.mPr[j] = mPr
            self.mPr[0] -= mPr
            if nu_rest==[]:
                nu_rest = totalNu
            nu_rest = nu_rest - self.nu[j,:,:]
                
        self.nu[0,:,:] = nu_rest             
        self.lnKappa[0,:,:] = psi(nu_rest) - psi(totalNu)
        self.sd_post_T[0] = np.sqrt(self.beta_var(self.nu[0,:,:], totalNu-self.nu[0,:,:]))
        self.expecT()   
            
        import gc
        gc.collect()
                     
        return self.ET
    
    def getmean(self, j=1):
        return self.mPr[j]

    def getsd(self, j=1):
        return self.sd_post_T[j]
        
    def preprocessTraining(self, crowdLabels, trainT=None):
        if trainT==None:
            if (crowdLabels.shape[1]!=4 or self.crowdTable != None):
                #trainT = np.zeros(crowdLabels.shape[0]) -1
                print 'Error: must use a sparse list of crowdsourced labels for HeatMapBCC'
                trainT = []
            else:
                trainT = np.zeros((self.nx,self.ny)) -1#(np.max(crowdLabels[:,1]), np.max(crowdLabels[:,2])) )
        
        self.trainT = trainT
           
    def preprocessCrowdLabels(self, crowdLabels):
        #ensure we don't have a matrix by mistake
        if not isinstance(crowdLabels, np.ndarray):
            crowdLabels = np.array(crowdLabels)
        self.crowdLabels = crowdLabels
        if crowdLabels.shape[1]!=4 or self.crowdTable != None:            
            print 'Must use a sparse list of crowdsourced labels with 4 columns:'
            print 'Agent ID, x-cood, y-coord, response value'
            return        
                
        #De-duplication of the observations 
        self.C = crowdLabels    
        linearIdxs = np.ravel_multi_index((self.C[:,1], self.C[:,2]), dims=(self.nx,self.ny))
        linearIdxs = np.unique(linearIdxs)
        self.obsx, self.obsy = np.unravel_index(linearIdxs, dims=(self.nx,self.ny))

    def initT(self):     
        kappa = self.nu / np.sum(self.nu, axis=0)        
        self.ET = np.zeros((self.nClasses, self.nx, self.ny)) + kappa     
        
    def createGP(self):
        #function can be overwritten by subclasses
        return GPGrid(self.nx, self.ny)
        
    def initLnKappa(self):
        if self.lnKappa != []:
            return
        self.nu = np.zeros((self.nClasses, self.nx, self.ny))
        self.lnKappa = np.zeros((self.nClasses, self.nx, self.ny))    
        
        self.heatGP = {}           
        for j in range(self.nClasses):
            self.nu[j,:,:] = self.nu0[j]   
            self.lnKappa[j,:,:] = psi(self.nu0[j]) - psi(np.sum(self.nu0))      
            #start with a homogeneous grid     
            if j>0:  
                self.heatGP[j] = self.createGP()

    def kappa_moments_to_nu(self, mPr, sdPr):
        totalNu = mPr*(1-mPr)/(sdPr**2) - 1
        nu_j = totalNu*mPr
        return nu_j, totalNu
    
    def beta_var(self, nu0, nu1):
        var = nu0*nu1 / ((nu0+nu1)**2 * (nu0+nu1+1))
        return var

    def expecLnKappa(self):
        nu_rest = []
        
        for j in range(1,self.nClasses):
            obsPoints = self.ET[j, self.obsx, self.obsy]
            mPr, sdPr = self.heatGP[j].train(self.obsx, self.obsy, obsPoints)    
            #convert to pseudo-counts
            nu_j, totalNu = self.kappa_moments_to_nu(mPr, sdPr)
            nu_j += self.nu0[j] - 1
            totalNu += np.sum(self.nu0) - 2
            self.nu[j, self.obsx, self.obsy] = nu_j
            if nu_rest==[]:
                nu_rest = totalNu
            nu_rest -= nu_j
            
            self.lnKappa[j, self.obsx, self.obsy] = psi(nu_j)-psi(totalNu)#np.log(mPr)
            
        self.nu[0, self.obsx, self.obsy] = nu_rest
        self.lnKappa[0, self.obsx, self.obsy] = psi(nu_rest)-psi(totalNu)#np.log(kappa_rest)
     
    def expecT(self):       
        lnjoint = np.zeros((self.nClasses, self.nx, self.ny))
        for j in range(self.nClasses):
            data = self.lnPi[j, self.C[:,3], self.C[:,0]].reshape(-1) #responses
            rows = np.array(self.C[:,1]).reshape(-1) #x co-ord
            cols = np.array(self.C[:,2]).reshape(-1) #y co-ord
            
            likelihood_j = coo_matrix((data, (rows,cols)), shape=(self.nx, self.ny)).todense()
            lnjoint[j,:,:] = likelihood_j
        
        lnjoint = lnjoint + self.lnKappa
        
        #ensure that the values are not too small
        largest = np.max(lnjoint, 0)
        lnjoint = lnjoint - largest
            
        joint = np.exp(lnjoint)
        self.ET = joint/np.sum(joint, axis=0)
            
        trainIdxs = self.trainT!=-1
        self.ET[:, trainIdxs] = 0
        for j in range(self.nClasses):            
            #training labels    
            self.ET[:,self.trainT==j] = 1    
            
        return lnjoint
    
    def postAlpha(self):#Posterior Hyperparams
        for j in range(self.nClasses):
            for l in range(self.nScores):
                sepcounts = (self.C[:,3]==l)*(self.ET[j,self.C[:,1],self.C[:,2]])
                counts = np.zeros((self.K, self.C.shape[0]))
                counts[self.C[:,0], np.arange(self.C.shape[0])] = sepcounts
                counts = np.sum(counts, axis=1).reshape(-1)
                self.alpha[j,l,:] = self.alpha0[j,l,:] + counts

    def postLnJoint(self, lnjoint):
        #not quite right if there are multiple observations at one point.
        ET = self.ET[:, self.obsx, self.obsy]
        lnjoint = lnjoint[:, self.obsx, self.obsy]
        lnpCT = np.sum(np.sum(np.sum( lnjoint*ET )))                        
        return lnpCT
    
    def postLnKappa(self):
        nObs = len(self.obsx)
        lnKappa_obs = self.lnKappa[:,self.obsx,self.obsy]
        nu0_obs = np.tile(self.nu0.reshape(self.nClasses,1), (1, nObs))
        
        lnpKappa = gammaln(np.sum(nu0_obs, 0))-np.sum(gammaln(nu0_obs), 0) + \
                    np.sum( (nu0_obs-1)*lnKappa_obs, 0)            
        return np.sum(lnpKappa)
 
    def qLnKappa(self):
        nu_obs = self.nu[:, self.obsx, self.obsy]
        lnKappa_obs = self.lnKappa[:,self.obsx,self.obsy]
        
        lnqKappa = gammaln(np.sum(nu_obs,0))-np.sum(gammaln(nu_obs), 0) + \
                    np.sum( (nu_obs-1)*lnKappa_obs, 0)
        return np.sum(lnqKappa)
        
    def qLnT(self):
        ET = self.ET[:, self.obsx, self.obsy]
        lnqT = np.sum( np.multiply( ET,np.log(ET) ) )
        return lnqT
