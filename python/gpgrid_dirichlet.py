def sigmoid(f,s):
    import numpy as np
    g = np.divide(1, 1+np.exp(-s*f))
    return g

def logit(x,s):
    import numpy as np    
    return np.log(np.divide(x,1-x))

def infer_gpgrid( obs_x, obs_y, obs_points, nx,ny, nu0, nsamps=1000 ):
    import numpy as np

    s=4 # should be learned    
    
    if obs_points==[]:
        obs_x = []
        obs_y = []
    else:
        obs_x = np.array(obs_x, dtype=np.int64)
        obs_y = np.array(obs_y, dtype=np.int64)
        if len(obs_points.shape)==1 or obs_points.shape[1]==1:
            grid_p = np.zeros((nx,ny))
            grid_all = np.zeros((nx,ny))
            obs_points = np.array(obs_points).reshape(-1)
            for i in range(len(obs_x)):
                grid_p[obs_x[i], obs_y[i]] += obs_points[i]
                grid_all[obs_x[i], obs_y[i]] += 1
            presp = grid_p[np.where(grid_p>0)]
            allresp = grid_all[np.where(grid_all>0)]
        elif obs_points.shape[1]==2:
            presp = np.array(obs_points[:,0]).reshape(obs_points.shape[0],1)
            allresp = np.array(obs_points[:,1]).reshape(obs_points.shape[0],1)
            
        z = np.divide(presp, allresp)
        z = z.reshape((z.size,1))
        
    grid_obs = np.zeros( (nx,ny) )
    grid_obs[obs_x,obs_y] = 1
        
    #X = np.argwhere(grid_obs.transpose().reshape((nx*ny,1)).reshape(-1)).reshape(-1)  
    
    #gridx = np.float64(np.tile( range(1,nx+1), (ny, 1) ))
    #gridy = np.float64(np.tile( range(1,ny+1), (nx, 1) ).transpose())
    
    #xx = gridx.transpose().reshape( (nx*ny,1) )
    #yy = gridy.transpose().reshape( (nx*ny,1) )
    
    #ddx = np.tile(xx, (1,nx*ny) ) - np.tile(xx, (1,nx*ny) ).transpose()
    #ddy = np.tile(yy, (1,nx*ny) ) - np.tile(yy, (1,nx*ny) ).transpose()
    
    #Update to produce training matrices only over known points
    ddx = np.tile(obs_x, (len(obs_x),1)) - np.tile(obs_x, (len(obs_x),1)).transpose();
    ddy = np.tile(obs_y, (len(obs_y),1)) - np.tile(obs_y, (len(obs_y),1)).transpose();
    
    Kx = np.exp( np.divide(-np.square(ddx), 1) )
    Ky = np.exp( np.divide(-np.square(ddy), 1) )
    
    K = np.multiply(Kx, Ky)
    n = nx*ny
            
    if obs_points==[]:
        f = np.zeros(K.shape[0])
        var = np.ones(n)
        return f, var, f+0.5, f+(0.5*0.5), s
    
    f = np.zeros(len(obs_points))
    
    #K = K + 1e-6 * np.eye(n) # jitter    
    
    #INCLUDE PRIORS HERE?
    Pr_est = np.divide( presp+1, allresp+2 )
    Q = np.diagflat( np.divide(np.multiply(Pr_est, 1-Pr_est), allresp) )
    
    prior_mean = np.divide(nu0, np.sum(nu0))
    f = f + logit(prior_mean, s)
    prior_var = np.divide(np.prod(nu0), np.multiply(np.square(np.sum(nu0)),(np.sum(nu0)+1)) )
    prior_var = prior_var + np.zeros(len(obs_points))
    
    converged = False    
    nIt = 0
    while not converged and nIt<1000:
        old_f = f
    
        #mean_X = sigmoid(f,s)
        #G = np.diagflat( s*np.multiply(mean_X, (1-mean_X)) )
        G = np.diagflat(s*prior_var)
        Gtr = G.transpose()
    
        W = K.dot(Gtr).dot( np.linalg.inv(G.dot(K).dot(Gtr)+Q) )
    
        f = W.dot(Gtr).dot(z-prior_mean)
        C = K - W.dot(G).dot(K)
    
        diff = np.max(np.abs(f-old_f))
        converged = diff<1e-3
        print 'GPGRID diff = ' + str(diff)
        nIt += 1
        
#     samps = np.random.multivariate_normal(f.reshape(-1), C, nsamps)
    
    Pr = np.zeros( (len(f), nsamps) )
#     for i in range(nsamps):
#         Pr[:,i] = sigmoid(samps[i,:].transpose(),s)
    
    #stdPr = np.std(Pr,1)
    #mPr = np.mean(Pr,1)        
        
    partialK = Gtr.dot(np.linalg.inv(G.dot(K).dot(Gtr)+Q) );    
      
    stdPr = np.zeros((nx,ny))
    mPr = np.zeros((nx,ny))
    f_out = f#np.zeros(n)
    var_out = C#np.zeros(n)
      
    for i in range(nx):
        for j in range(ny):
            ddx = i-obs_x;
            ddy = j-obs_y;
    
            Kx = np.exp( np.divide(-np.square(ddx), 1) )
            Ky = np.exp( np.divide(-np.square(ddy), 1) )
    
            Kpred = np.multiply(Kx,Ky)
    
            W = Kpred.dot(partialK)
    
            f = W.dot(Gtr).dot(z-0.5)
            C = 1 - W.dot(G).dot(Kpred.transpose())
    
            samps = np.random.normal(f, C, nsamps)
    
            for ss in range(nsamps):
                Pr[:, ss] = sigmoid(samps[ss],s)
    
            stdPr[i,j] = np.std(Pr)
            mPr[i,j] = np.mean(Pr)
            
            #f_out[(i*nx)+j] = f
            #var_out[(i*nx)+j] = C
                
    mPr = mPr.reshape( (nx,ny) )
    stdPr = stdPr.reshape( (nx,ny) )  
    
    return f_out, var_out, mPr, stdPr, s

