
import ibcc
import numpy as np
from scipy.special import psi, gammaln
from scipy.sparse import coo_matrix
from scipy.linalg import eigh 
from scipy.stats import norm
from gpgrid import infer_gpgrid
    
def _pinv_1d(v, eps=1e-5):
    """
    A helper function for computing the pseudoinverse.

    Parameters
    ----------
    v : iterable of numbers
        This may be thought of as a vector of eigenvalues or singular values.
    eps : float
        Elements of v smaller than eps are considered negligible.

    Returns
    -------
    v_pinv : 1d float ndarray
        A vector of pseudo-inverted numbers.

    """
    return np.array([0 if abs(x) < eps else 1/x for x in v], dtype=float)    
    
def _psd_pinv_decomposed_log_pdet(mat, cond=None, rcond=None,
                                  lower=True, check_finite=True):
    """
    SCIPY 0.14
    Compute a decomposition of the pseudo-inverse and the logarithm of
    the pseudo-determinant of a symmetric positive semi-definite
    matrix.

    The pseudo-determinant of a matrix is defined as the product of
    the non-zero eigenvalues, and coincides with the usual determinant
    for a full matrix.

    Parameters
    ----------
    mat : array_like
        Input array of shape (`m`, `n`)
    cond, rcond : float or None
        Cutoff for 'small' singular values.
        Eigenvalues smaller than ``rcond*largest_eigenvalue``
        are considered zero.
        If None or -1, suitable machine precision is used.
    lower : bool, optional
        Whether the pertinent array data is taken from the lower or upper
        triangle of `mat`. (Default: lower)
    check_finite : boolean, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    M : array_like
        The pseudo-inverse of the input matrix is np.dot(M, M.T).
    log_pdet : float
        Logarithm of the pseudo-determinant of the matrix.

    """
    # Compute the symmetric eigendecomposition.
    # The input covariance matrix is required to be real symmetric
    # and positive semidefinite which implies that its eigenvalues
    # are all real and non-negative,
    # but clip them anyway to avoid numerical issues.

    # TODO: the code to set cond/rcond is identical to that in
    # scipy.linalg.{pinvh, pinv2} and if/when this function is subsumed
    # into scipy.linalg it should probably be shared between all of
    # these routines.

    # Note that eigh takes care of array conversion, chkfinite,
    # and assertion that the matrix is square.
    s, u = eigh(mat, lower=lower, check_finite=check_finite)

    if rcond is not None:
        cond = rcond
    if cond in [None, -1]:
        t = u.dtype.char.lower()
        factor = {'f': 1E3, 'd': 1E6}
        cond = factor[t] * np.finfo(t).eps
    eps = cond * np.max(abs(s))

    if np.min(s) < -eps:
        raise ValueError('the covariance matrix must be positive semidefinite')

    s_pinv = _pinv_1d(s, eps)
    U = np.multiply(u, np.sqrt(s_pinv))
    log_pdet = np.sum(np.log(s[s > eps]))

    return U, log_pdet    

_LOG_2PI = np.log(2 * np.pi)

def _loggausspdf(x, mean, cov):
    """
    SCIPY 0.14
    Parameters
    ----------
    x : ndarray
        Points at which to evaluate the log of the probability
        density function
    mean : ndarray
        Mean of the distribution
    prec_U : ndarray
        A decomposition such that np.dot(prec_U, prec_U.T)
        is the precision matrix, i.e. inverse of the covariance matrix.
    log_det_cov : float
        Logarithm of the determinant of the covariance matrix

    Notes
    -----
    As this function does no argument checking, it should not be
    called directly; use 'logpdf' instead.

    """
    prec_U, log_det_cov = _psd_pinv_decomposed_log_pdet(cov)
    
    dim = x.shape[-1]
    dev = x - mean
    maha = np.sum(np.square(np.dot(dev, prec_U)), axis=-1)
    return -0.5 * (dim * _LOG_2PI + log_det_cov + maha)    


class  Heatmapbcc(ibcc.Ibcc):
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
    
    f_pr = [] #priors from the GP
    Cov_pr = []
    s_pr = []
    
    f = []#posteriors from the GP
    Cov = []
    s = []   
    mPr = []   
    
    sd_post_T = []    
    
    def __init__(self, nx, ny, nClasses, nScores, alpha0, nu0, K, tableFormat=False):
        self.nx = nx
        self.ny = ny
        self.nObjs = nx*ny
        self.lnKappa = []
        self.post_T = []
        print 'Setting up a 2-D grid. This should be generalised!'        
        super(Heatmapbcc, self).__init__(nClasses, nScores, alpha0, nu0, K, tableFormat) 
        
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
        self.crowdLabels = crowdLabels
        if crowdLabels.shape[1]!=4 or self.crowdTable != None:            
            print 'Must use a sparse list of crowdsourced labels with 4 columns:'
            print 'Agent ID, x-cood, y-coord, response value'
            return        
                
        self.C = crowdLabels
        obsGrid = np.zeros((self.nx,self.ny))
        obsGrid[self.C[:,1], self.C[:,2]] = 1
        obsPairs = np.argwhere(obsGrid>0)
        self.obsIdx_x = obsPairs[:,0]
        self.obsIdx_y = obsPairs[:,1]
    
        nobs = len(self.obsIdx_x)
        self.f_pr = np.tile(self.f_pr, nobs)
        self.Cov_pr = np.tile(self.Cov_pr, (nobs,nobs))
    
    def initT(self):     
        kappa = self.nu / np.sum(self.nu, axis=0)        
        self.ET = np.zeros((self.nClasses, self.nx, self.ny)) + kappa     
        
    def initLnKappa(self):
        if self.lnKappa != []:
            return
        self.nu = np.zeros((self.nClasses, self.nx, self.ny))
        self.lnKappa = np.zeros((self.nClasses, self.nx, self.ny))        
        for j in range(self.nClasses):
            self.nu[j,:,:] = self.nu0[j]   
            self.lnKappa[j,:,:] = psi(self.nu0[j]) - psi(np.sum(self.nu0))      
        #start with a homogeneous grid       
        print 'Priors nu0 do not affect the gpgrid in the correct way'
        self.f_pr, self.Cov_pr,_,_,self.s_pr = infer_gpgrid([], [], [], \
                                                    self.nx, self.ny, self.nu0)
              
    
    def expecLnKappa(self):
        
        for j in range(1,self.nClasses):
            obsPoints = self.ET[j, self.obsIdx_x, self.obsIdx_y]
            self.f, self.Cov, mPr, sdPr, self.s = infer_gpgrid( \
                     self.obsIdx_x, self.obsIdx_y, obsPoints, self.nx, self.ny, self.nu0)    
            #convert to pseudo-counts
            totalNu = np.divide(np.multiply(mPr,(1-mPr)), (np.power(sdPr,2))) - 1
            self.nu[j,:,:] = np.multiply(totalNu, mPr)
            totalNu = np.multiply(totalNu, (1-mPr))
            
            self.lnKappa[j,:,:] = np.log(mPr)
            
        self.nu[0,:,:] = totalNu
        self.lnKappa[0,:,:] = np.log(1-mPr)
#         self.lnKappa = np.concatenate(np.log(1-np.transpose(mPr)), np.log(np.transpose(mPr))) 
        self.sd_post_T = sdPr
        self.mPr = mPr
            
    def expecT(self):       
        lnjoint = np.zeros((self.nClasses, self.nx, self.ny))
        for j in range(self.nClasses):
            data = self.lnPi[j, self.C[:,3], self.C[:,0]].reshape(-1)
            rows = np.array(self.C[:,1]).reshape(-1)
            cols = np.array(self.C[:,2]).reshape(-1)
            
            likelihood_j = coo_matrix((data, (rows,cols)), shape=(self.nx, self.ny)).todense()
            lnjoint[j,:,:] = likelihood_j.reshape(1,self.nx, self.ny) + self.lnKappa[j,:,:]     
        
        joint = np.zeros((self.nClasses, self.nx, self.ny))
        pT = np.zeros((self.nClasses, self.nx, self.ny))
        #ensure that the values are not too small
        largest = np.max(lnjoint, 0)
        for j in range(self.nClasses):
            joint[j,:,:] = lnjoint[j,:,:] - largest
            
        joint = np.exp(joint)
        norma = np.sum(joint, axis=0)
        for j in range(self.nClasses):
            pT[j,:,:] = np.divide(joint[j,:,:], norma)
            self.ET[j,:,:] = pT[j,:,:]
            
        trainIdxs = self.trainT!=-1
        self.ET[:, trainIdxs] = 0
        for j in range(self.nClasses):            
            #training labels    
            self.ET[:,self.trainT==j] = 1    
            
        return lnjoint
    
    def expecLnPi(self):#Posterior Hyperparams
        for j in range(self.nClasses):
            for l in range(self.nScores):
                counts = np.matrix( np.transpose(self.C[:,3]==l) \
                                    * self.ET[1,self.C[:,1],self.C[:,2]])
                self.alpha[j,l,:] = self.alpha0[j,l,:] + counts
        self.initLnPi()   

    def postLnJoint(self, lnjoint):
        lnpCT = np.sum(np.sum(np.sum( np.multiply(lnjoint, self.ET) )))                        
        return lnpCT
    
    def postLnKappa(self):
#         lnpKappa = np.tile(gammaln(np.sum(self.nu0))-np.sum(gammaln(self.nu0)), (1,self.nx,self.ny)) \
#                     + np.sum(np.multiply(np.reshape(self.nu0-1,(self.nClasses,1,1)),self.lnKappa))
#         return np.sum(np.sum(lnpKappa))
        
        #lnpKappa = self.lnGaussPdf(self.f_pr.transpose(), self.f.transpose(), self.Cov_pr)
        #lnpKappa = np.sum(lnpKappa)
        
        #f_pr = np.zeros(len(self.f))
        #cov_pr = np.diagflat(np.ones(len(self.f)))
        
#         prior_mean = np.divide(self.nu0, np.sum(self.nu0))
#         f_pr = np.zeros(len(self.f)) + logit(prior_mean,4)
#         prior_var = np.divide(np.prod(self.nu0), np.multiply(np.square(np.sum(self.nu0)),(np.sum(self.nu0)+1)) )
#         cov_pr = np.diagflat(prior_var)
        lnpKappa = _loggausspdf(self.f.transpose(), self.f_pr.transpose(), self.Cov_pr)
        return lnpKappa 

    def lnGaussPdf (self, x, m, v):
        return -np.log(np.sqrt(v)) -np.log(np.sqrt(2*np.pi)) - np.divide(np.square(x-m), (2*v))
             
    def qLnKappa(self):
        
        #lnqKappa = self.lnGaussPdf(self.f.transpose(), self.f.transpose(), self.Cov)
        #lnqKappa = np.sum(lnqKappa)
        
        lnqKappa = _loggausspdf(self.f.transpose(), self.f.transpose(), self.Cov)
        
#         lnqKappa = gammaln(np.sum(self.nu))-np.sum(gammaln(self.nu)) \
#                         + np.sum(np.multiply(self.nu-1,self.lnKappa))
#         return np.sum(np.sum(lnqKappa))
        return lnqKappa
