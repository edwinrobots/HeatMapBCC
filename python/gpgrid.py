def sigmoid(f,s):
    import numpy as np
    g = np.divide(1, 1+np.exp(-s*f))
    return g

def infer_gpgrid( obs_x, obs_y, obs_points, nx,ny, nsamps=1000 ):
    import numpy as np

    s=1 # should be learned    
    
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
        
    X = np.argwhere(grid_obs.transpose().reshape((nx*ny,1)).reshape(-1)).reshape(-1)  
    
    gridx = np.float64(np.tile( range(1,nx+1), (ny, 1) ))
    gridy = np.float64(np.tile( range(1,ny+1), (nx, 1) ).transpose())
    
    xx = gridx.transpose().reshape( (nx*ny,1) )
    yy = gridy.transpose().reshape( (nx*ny,1) )
    
    ddx = np.tile(xx, (1,nx*ny) ) - np.tile(xx, (1,nx*ny) ).transpose()
    ddy = np.tile(yy, (1,nx*ny) ) - np.tile(yy, (1,nx*ny) ).transpose()
    
    Kx = np.exp( np.divide(-np.square(ddx), 100) )
    #Ky=exp(-ddy.^2/100);
    Ky = np.exp( np.divide(-np.square(ddy), 100) )
    
    K = np.multiply(Kx, Ky)
    n = K.shape[0]
            
    f = np.zeros(n)
    
    if obs_points==[]:
        return f, K, f+0.5, f+(0.5*0.5), s
    
    #K = K + 1e-6 * np.eye(n) # jitter    
    
    #INCLUDE PRIORS HERE?
    Pr_est = np.divide( presp+1, allresp+2 )
    Q = np.diagflat( np.divide(np.multiply(Pr_est, 1-Pr_est), allresp) )
    
    converged = False    
    nIt = 0
    while not converged and nIt<1000:
        old_f = f
    
        mean_X = sigmoid(f[X],s)
        G = np.diagflat( s*np.multiply(mean_X, (1-mean_X)) )
        Gtr = G.transpose()
    
        W = K[:,X].dot(Gtr).dot( np.linalg.inv(G.dot(K[X,:][:,X]).dot(Gtr)+Q) )
    
        f = W.dot(Gtr).dot(z-0.5)
        C = K - W.dot(G).dot(K[X,:])
    
        diff = np.max(np.abs(f-old_f))
        converged = diff<1e-3
        print 'GPGRID diff = ' + str(diff)
        nIt += 1
        
    samps = np.random.multivariate_normal(f.reshape(-1), C, nsamps)
    
    Pr = np.zeros( (len(f), nsamps) )
    for i in range(nsamps):
        Pr[:,i] = sigmoid(samps[i,:].transpose(),s)
    
    stdPr = np.std(Pr,1)
    mPr = np.mean(Pr,1)
    
    mPr = mPr.reshape( (nx,ny) )
    stdPr = stdPr.reshape( (nx,ny) )  
    
    return f, C, mPr, stdPr, s

