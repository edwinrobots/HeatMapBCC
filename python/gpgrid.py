def sigmoid(f,s):
    import numpy as np
    g = np.divide(1, 1+np.exp(-s*f))
    return g

def target_var(f,s,v):
    import numpy as np
    mean = sigmoid(f,s)
    topvariance = np.multiply(mean,(1-mean))
    v = np.multiply(v,np.square(s))
    prec = np.divide(1,np.multiply(v,np.square(topvariance))) + np.divide(1,topvariance)
    return np.divide(1,prec)

def latentvariance(std, mean,s):
    import numpy as np
    prec = np.divide(1,np.square(std))
    topvariance = np.multiply(mean,(1-mean))
    prec = prec - np.divide(1,topvariance)
    var = np.divide(1,prec)
    var = np.divide(var, np.square(topvariance))
    var = np.divide(var, np.square(s))
    return var

def process_observations( obs_x, obs_y, obs_points, nx, ny, nu0 ):
    import numpy as np
    if obs_points==[]:
        return [],[],[],[],[]
        
    obs_x = np.array(obs_x, dtype=np.int64)
    obs_y = np.array(obs_y, dtype=np.int64)
    if len(obs_points.shape)==1 or obs_points.shape[1]==1:
        grid_p = np.zeros((nx,ny))
        grid_all = np.zeros((nx,ny))
        obs_points = np.array(obs_points).reshape(-1)
        for i in range(len(obs_x)):
            grid_p[obs_x[i], obs_y[i]] += obs_points[i]
            grid_all[obs_x[i], obs_y[i]] += 1
           
        deduped_idxs = np.argwhere(grid_all>0)
        obs_x = deduped_idxs[:,0]
        obs_y = deduped_idxs[:,1]   
            
        presp = grid_p[obs_x,obs_y]
        allresp = grid_all[obs_x,obs_y]
        
    elif obs_points.shape[1]==2:
        presp = np.array(obs_points[:,0]).reshape(obs_points.shape[0],1)
        allresp = np.array(obs_points[:,1]).reshape(obs_points.shape[0],1)
     
    presp += nu0[1]
    allresp += np.sum(nu0)
        
    z = np.divide(presp, allresp)
    z = z.reshape((z.size,1))
        
    return obs_x, obs_y, presp, allresp, z    

def infer_gpgrid( obs_x, obs_y, obs_points, nx,ny, nu0, nsamps=1000 ):
    import numpy as np

    s=4 # should be learned    
    
    obs_x, obs_y, presp, allresp, z = process_observations(obs_x, obs_y, obs_points, nx, ny, nu0)
        
    #grid_obs = np.zeros( (nx,ny) )
    #grid_obs[obs_x,obs_y] = 1
    
    prior_mean = nu0[1] / np.sum(nu0)
    prior_mean_latent = np.log(prior_mean/(1-prior_mean))
    prior_var = prior_mean*(1-prior_mean)/(nu0[0]+nu0[1]+1)#1/nu0[0] + 1/nu0[1] #should be dealt with through Q?
    #prior_var_latent = latentvariance(prior_var, prior_mean, s) 
    
    if obs_x==[]:
        print "What is correct way to apply priors? Adding pseudo-counts will not apply to points that" + \
        "are  not included in training."
        mPr = prior_mean
        stdPr = np.sqrt(prior_var)        
        f = prior_mean_latent
        var = latentvariance(stdPr, mPr, s)
        return f, var, mPr, stdPr, s   
         
    #X = np.argwhere(grid_obs.transpose().reshape((nx*ny,1)).reshape(-1)).reshape(-1)  
    
    #gridx = np.float64(np.tile( range(1,nx+1), (ny, 1) ))
    #gridy = np.float64(np.tile( range(1,ny+1), (nx, 1) ).transpose())
    
    #xx = gridx.transpose().reshape( (nx*ny,1) )
    #yy = gridy.transpose().reshape( (nx*ny,1) )
    
    #ddx = np.tile(xx, (1,nx*ny) ) - np.tile(xx, (1,nx*ny) ).transpose()
    #ddy = np.tile(yy, (1,nx*ny) ) - np.tile(yy, (1,nx*ny) ).transpose()
    
    #Update to produce training matrices only over known points
    ddx = np.float64(np.tile(obs_x, (len(obs_x),1)).transpose() - np.tile(obs_x, (len(obs_x),1)));
    ddy = np.float64(np.tile(obs_y, (len(obs_y),1)).transpose() - np.tile(obs_y, (len(obs_y),1)));
    
    #length scale
    ls = 100
    
    Kx = np.exp( np.divide(-np.square(ddx), ls) )
    Ky = np.exp( np.divide(-np.square(ddy), ls) )
    K = np.multiply(Kx, Ky)
    #n = nx*ny

    f = np.zeros(len(obs_x))
    #K = K + 1e-6 * np.eye(n) # jitter    
    
    #INCLUDE PRIORS HERE?
    Pr_est = np.divide( presp+1, allresp+2 )
    Q = np.diagflat( np.divide(np.multiply(Pr_est, 1-Pr_est), allresp) )
    
    converged = False    
    nIt = 0
    while not converged and nIt<1000:
        old_f = f
    
        mean_X = sigmoid(f,s)
        G = np.diagflat( s*np.multiply(mean_X, (1-mean_X)) )
        Gtr = G.transpose()
    
        W = K.dot(Gtr).dot( np.linalg.inv(G.dot(K).dot(Gtr)+Q) )
    
        f = prior_mean_latent + W.dot(Gtr).dot(z-prior_mean) #prior_mean_latent + 
        C = K - W.dot(G).dot(K) #possibly needs additional variance term sicne the observation is uncertain?
    
        diff = np.max(np.abs(f-old_f))
        converged = diff<1e-3
        #print 'GPGRID diff = ' + str(diff)
        nIt += 1
        
#     samps = np.random.multivariate_normal(f.reshape(-1), C, nsamps)
    
    #Pr = np.zeros( (len(f), nsamps) )
#     for i in range(nsamps):
#         Pr[:,i] = sigmoid(samps[i,:].transpose(),s)
    
    #stdPr = np.std(Pr,1)
    #mPr = np.mean(Pr,1)        
    partialK = Gtr.dot(np.linalg.inv(G.dot(K).dot(Gtr)+Q) );        
    mPr, stdPr = post_grid(G, Gtr, partialK, nx, ny, obs_x, obs_y, prior_mean, prior_mean_latent, z, f, C, s, ls)
    return f, C, mPr, stdPr, s

def post_peaks(G, Gtr, partialK, nx, ny, obs_x, obs_y, prior_mean, prior_mean_latent, z, f, C,s):
    import numpy as np
       
    v = np.diag(C)
    stdPr = target_var(f,s,v)
    mPr = sigmoid(f,s)
      
    return mPr, stdPr

def post_grid(G, Gtr, partialK, nx, ny, obs_x, obs_y, prior_mean, prior_mean_latent, z, f, C,s, ls):    
    import numpy as np
       
    ddy = np.float64(np.tile(range(ny), (len(obs_y),1)) - np.tile(obs_y, (ny,1)).transpose());
    Ky = np.exp( np.divide(-np.square(ddy), ls) ).transpose()   
      
    fending = Gtr.dot(z-prior_mean)
    
    f = np.zeros((nx, ny))
    C = np.zeros((nx, ny))
      
#     mPr = np.zeros((nx,ny))
#     stdPr = np.zeros((nx,ny))
    #ddx_i = np.float64(np.tile(-obs_x, (ny,1))) 
    ddx = np.float64(np.tile(-obs_x, (nx,1))) + np.array(range(nx)).reshape((nx,1))
    Kx = np.exp( np.divide(-np.square(ddx), ls) )  
    
    for i in range(nx):
        print i
#         for j in range(ny):
#             ddx=np.float64(i-obs_x)
#             ddy=np.float64(j-obs_y)
#  
#             Kx=-np.square(ddx)/ls
#             Ky=-np.square(ddy)/ls
#  
#             Kpred=np.exp(Kx+Ky)#np.multiply(Kx,Ky);           
#             
#             W=Kpred.dot(partialK)
#             
#             f[i,j] = W.dot(Gtr).dot(z-prior_mean)
#             C[i,j] = 1 - W.dot(G).dot(Kpred.transpose())           
        
        #Kx_i = np.exp( np.divide(-np.square(ddx_i+i), ls) )
        Kx_i = Kx[i,:]
        Kpred_i = np.multiply(Ky,Kx_i)
        W_i = Kpred_i.dot(partialK)  
        f[i,:] = W_i.dot(fending).reshape(-1) 
        C[i,:] = -np.sum(np.multiply(W_i.dot(G), Kpred_i), axis=1)       
 
#         nsamps = 1000
#         for j in range(ny):
#             samps = np.random.normal(f[i,j], C[i,j], nsamps)
#             Pr = np.zeros(nsamps)
#             for ss in range(nsamps):
#                 Pr[ss] = sigmoid(samps[ss],s)
#             mPr[i,j] = np.mean(Pr)
#             stdPr[i,j] = np.std(Pr)
          
    f = f + prior_mean_latent 
    C = C + 1
            
    mPr = sigmoid(f,s)
    stdPr = np.sqrt(target_var(f, s, C))
    
    return mPr, stdPr
