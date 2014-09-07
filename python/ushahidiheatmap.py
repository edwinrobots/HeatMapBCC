'''
Created on 23 Jun 2014

@author: edwin
'''
import heatmapbcc, time, logging, json
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D #Can use if we want to make a 3D plot instead of flat heat map
from scipy.sparse import coo_matrix
from copy import deepcopy
#from memory_profiler import profile

class Heatmap(object):

    sea_map = []
    combiner = {}

    webdatadir = './web'
    datadir = './data'
    fileprefix = '/mapdata/map_test'
    
    minlat = 18.2#18.0
    maxlat = 18.8#19.4
    minlon = -72.6#-73.1
    maxlon = -72.0#-71.7   
    
    target_threshold = 0.775
    
    nx = 100
    ny = 100
    
    C = []
    K = 1
    rep_ids = []
    
    timestep = 20#579#20
    
    running = True
    
    alpha0 = []
    nu0 = []
    
    combine_cat_three_and_one = False
    
    def __init__(self, nx,ny, minlat=None,maxlat=None, minlon=None,maxlon=None, fileprefix=None, compose_demo_reports=False):
        self.nx = nx
        self.ny = ny
        if minlat != None:
            self.minlat = minlat
        if maxlat != None:
            self.maxlat = maxlat
        if minlon != None:
            self.minlon = minlon
        if maxlon != None:
            self.maxlon = maxlon  
        
        if fileprefix==None:
            self.fileprefix = self.webdatadir + self.fileprefix
        else:
            self.fileprefix = self.webdatadir + fileprefix
                
        self.alpha0 = np.array([[2.0, 1.0], [1.0, 2.0]])
        self.nu0 = np.array([1,1])#np.array([0.5, 0.5])#0.03
        self.rep_ids.append(0)
        
        self.combine_cat_three_and_one = compose_demo_reports
        
    def runBCC(self, j):
        C = self.C[j]
        self.runBCC_subset(C)
        
    def runBCC_up_to_time(self, j, timestep):       
        C = self.C[j]
        C = C[0:timestep+1,:]
        bcc_pred, _ = self.runBCC_subset(C)
        return bcc_pred
            
    def runBCC_subset(self, C, j=1):
        if j not in self.combiner or self.combiner[j]==None:
            self.combiner[j] = heatmapbcc.Heatmapbcc(self.nx, self.ny, 2, 2, self.alpha0, self.nu0, self.K)
            self.combiner[j].minNoIts = 5
            self.combiner[j].maxNoIts = 200
            self.combiner[j].convThreshold = 0.1
                
        bcc_pred = self.combiner[j].combineClassifications(C)
        bcc_pred = bcc_pred[j,:,:].reshape((self.nx,self.ny))
        return bcc_pred, self.combiner[j]

    #@profile
    def loop_iteration(self, j, t):
        bcc_pred = self.runBCC_up_to_time(j,t)
        return bcc_pred   

    def move_reps(self, rgrid, x, y, newx, newy):
        if rgrid[x,y] == None:
            return rgrid
        
        if rgrid[newx,newy]==None:
            rgrid[newx,newy] = []
        elif len(rgrid[newx,newy])>10:
            rgrid[newx,newy] = rgrid[newx,newy][0:10]
            
        for i in range(len(rgrid[x,y])):
            if len(rgrid[newx,newy])<10:
                rgrid[newx, newy].append(rgrid[x,y][i])
        rgrid[x,y] = []
        return rgrid    

    #storing target info
    targetsx = []
    targetsy = []
    targetids = []
    targetversions = []

    def calculate_targets(self, pgrid, rep_id_grid, theta=0):
        targetsx, targetsy, plist,bgrid,rep_ids = self.find_peaks(pgrid, rep_id_grid, theta)
        
        dist = np.zeros((len(targetsx),len(self.targetsx)))
        
        newtargetids = np.zeros(targetsx.shape)-1
        newtargetversions = np.zeros(targetsx.shape)
        
        if len(self.targetids)<1:
            self.targetsx = targetsx
            self.targetsy = targetsy
            self.targetids = np.arange(len(targetsx))
            self.targetversions = np.zeros(self.targetids.shape)
            return plist, bgrid, rep_ids
        
        #see if the peaks are close matches to existing targets
        for t in range(len(targetsx)):
            x = targetsx[t]
            y = targetsy[t]
            
            #calculate distances to old targets
            dist_t = np.sqrt((x-self.targetsx)**2 + (y-self.targetsy)**2)
            dist[t,:] = dist_t
            
        #go through looking for most similar peaks first
        
        nIterations = len(targetsx)
        num_new_ids = 0
        if nIterations>len(self.targetsx):
            num_new_ids = nIterations-len(self.targetsx)
            nIterations = len(self.targetsx)
            
        for _ in range(nIterations):
            closest_old_to_new = np.argmin(dist,axis=1)
            mindist_old_to_new = np.min(dist,axis=1)
        
            least_moved_new = np.argmin(mindist_old_to_new)
            least_moved_old = closest_old_to_new[least_moved_new]
            newtargetids[least_moved_new] = self.targetids[least_moved_old]
            
            if targetsx[least_moved_new]==self.targetsx[least_moved_old] \
                and targetsy[least_moved_new]==self.targetsy[least_moved_old]:
                newtargetversions[least_moved_new] = self.targetversions[least_moved_old]
            else:
                newtargetversions[least_moved_new] = self.targetversions[least_moved_old]+1
            dist[:,least_moved_old] = np.Inf
            
        if num_new_ids>0:
            missingid_idxs = np.argwhere(newtargetids<0)
            missingids = range(nIterations,nIterations+num_new_ids)
            newtargetids[missingid_idxs] = missingids
            
        self.targetids = newtargetids
        self.targetversions = newtargetversions
        self.targetsx = targetsx
        self.targetsy = targetsy        
        
        logging.info("Maximimum target ID is " + str(np.max(self.targetids)))
        
        return plist, bgrid, rep_ids

    def find_peaks(self, pgrid, rep_id_grid, theta=0):
        #Turn a grid of predictions of events, e.g. bcc_pred, into a set of binary points
        #representing the most likely events. 
        #find points > theta        
        if theta==0:
            theta = np.max(pgrid) - 0.12
        
        bgrid = np.array(pgrid>theta, dtype=np.int8)
        
        rgrid = deepcopy(rep_id_grid)
        
        for x in np.arange(bgrid.shape[0]):
            for y in np.arange(bgrid.shape[1]):
                if bgrid[x,y]==0:
                    continue
                
                #move reports from discarded neighbours
                if bgrid[x+1,y]==0:
                    rgrid = self.move_reps(rgrid, x+1, y, x, y)
                if bgrid[x-1,y]==0:
                    rgrid = self.move_reps(rgrid, x-1, y, x, y)
                if bgrid[x,y+1]==0:
                    rgrid = self.move_reps(rgrid, x, y+1, x, y)                                    
                if bgrid[x,y-1]==0:
                    rgrid = self.move_reps(rgrid, x, y-1, x, y)
                if bgrid[x+1,y+1]==0:
                    rgrid = self.move_reps(rgrid, x+1, y+1, x, y)
                if bgrid[x-1,y-1]==0:
                    rgrid = self.move_reps(rgrid, x-1, y-1, x, y)
                if bgrid[x-1,y+1]==0:
                    rgrid = self.move_reps(rgrid, x-1, y+1, x, y)                                    
                if bgrid[x+1,y-1]==0:
                    rgrid = self.move_reps(rgrid, x+1, y-1, x, y)     
                
                #find highest neighbour
                highestx = x
                highesty = y
                highestp = pgrid[x,y]
                
                if highestp <= pgrid[x+1,y]:
                    highestx = x+1
                    highesty = y
                    highestp = pgrid[x+1, y]
                    
                if highestp <= pgrid[x-1,y]:
                    highestx = x-1
                    highesty = y
                    highestp = pgrid[x-1, y]   
 
                if highestp <= pgrid[x,y-1]:
                    highestx = x
                    highesty = y-1
                    highestp = pgrid[x, y-1]  
                    
                if highestp <= pgrid[x,y+1]:
                    highestx = x
                    highesty = y+1
                    highestp = pgrid[x, y+1]
                    
                if highestp <= pgrid[x+1,y+1]:
                    highestx = x+1
                    highesty = y+1
                    highestp = pgrid[x+1, y+1]
                    
                if highestp <= pgrid[x-1,y-1]:
                    highestx = x-1
                    highesty = y-1
                    highestp = pgrid[x-1, y-1]   
 
                if highestp <= pgrid[x+1,y-1]:
                    highestx = x+1
                    highesty = y-1
                    highestp = pgrid[x+1, y-1]  
                    
                if highestp <= pgrid[x-1,y+1]:
                    highestx = x-1
                    highesty = y+1
                    highestp = pgrid[x-1, y+1]                    
                    
                if highestx!=x or highesty!=y:
                    bgrid[x,y] = -1
                    rgrid = self.move_reps(rgrid, x, y, highestx, highesty)
                else:
                    logging.info("target found at " + str(x) + ", " + str(y))
                                   
        bgrid[bgrid==-1] = 0
                                        
        target_list = np.argwhere(bgrid)
        targetsx = target_list[:,0]
        targetsy = target_list[:,1]
        
        target_rep_ids = rgrid[targetsx, targetsy]
        
        p_list = pgrid[targetsx,targetsy]     
        targetsx, targetsy = self.tranlate_points_to_original(targetsx, targetsy)
        return targetsx, targetsy, np.around(p_list,2), bgrid, target_rep_ids
              
    def enlarge_target_blobs(self, target_grid, nsize=5):
        #make neighbouring points also one
        pospoints = np.argwhere(target_grid)
        for row in pospoints:
            xblob = np.arange(row[0]-nsize,row[0]+nsize)
            yblob = np.arange(row[1]-nsize,row[1]+nsize)
            for x in xblob:
                target_grid[x, yblob] = 1
        return target_grid
                    
    def timed_update_loop(self, j=1):
        logging.info("Run BCC at intervals, loading new reports.")
        
        rep_id_grid, rep_list = self.load_ush_data() 
        self.write_coords_json()
        
        #Call this in a new thread that can be updated by POST to the web server. 
        #When a new report is received by POST to web server, the server can kill this thread, call insert_trusted and then restart this method in a new thread
        stepsize = 9.0
        nupdates = self.C[j].shape[0]
        
        #print "!!! Breaking the gradual update so we skip to final update loop"
        #self.timestep = nupdates
        
        while self.timestep<=nupdates:
            logging.info("timed_update_loop timestep " + str(self.timestep))
            starttime = time.time()

            bcc_pred = self.loop_iteration(j, self.timestep)

            if not self.running:
                logging.info("Stopping update loop for the heatmap")
                break

            self.plotresults(bcc_pred, label='Predicted Incidents of type '+str(j))
            self.write_img("", j)
  
            bcc_stdPred = np.sqrt(bcc_pred*(1-bcc_pred))#self.combiner[j].getsd()
            self.plotresults(bcc_stdPred,  label='Uncertainty (S.D.) in Pr(incident) of type '+str(j))
            self.write_img("_sd_",j)
            
            rep_pred, rep_std = self.reportintensity(j, self.timestep)
            self.plotresults(rep_pred, label='Predicted Incidents of type '+str(j))
            self.write_img("_rep_intensity_", j)
            
            self.plotresults(rep_std, label='Uncertainty (S.D.) in Pr(incident) of type '+str(j))
            self.write_img("_rep_intensity__sd_",j)   
            
            p_list, target_grid, target_rep_ids = \
                self.calculate_targets(bcc_pred, rep_id_grid[j], self.target_threshold)
            self.plotresults(self.enlarge_target_blobs(target_grid), 'Predicted target points of type ' + str(j))
            self.write_img("_targets_", j)
            expec_pi = self.combiner[j].alpha / np.sum(self.combiner[j].alpha, 1)
            self.write_targets_json(p_list, target_rep_ids, rep_list[j], pi=expec_pi)
            
            plt.close("all")
            
            endtime = time.time()
            
            waittime = 5 # make sure we have left a pause between updates
            if endtime-starttime < waittime:
                time.sleep(waittime-endtime+starttime)
                
            logging.info("Update loop took " + str(endtime-starttime) + " seconds.")
            
            #once complete, update the current time step
            self.timestep += stepsize
            
    def kill_combiners(self):
        self.running = False
        for j in self.combiner.keys():
            self.combiner[j].keeprunning = False
    
    def reportintensity(self, j, timestep=-1):
        if timestep==-1:
            C = self.C[j]
        else:
            C = self.C[j][0:timestep,:]
        
        data = np.array(C[:,3]).reshape(-1)
        rows = np.array(C[:,1]).reshape(-1)
        cols = np.array(C[:,2]).reshape(-1)
        
        nu0 = np.array([1,1])
        
        repgrid = coo_matrix((data,(rows,cols)), shape=(self.nx,self.ny))
        
        negdata = np.array(C[:,3]).reshape(-1)
        negdata[np.where(data==0)] = -1
        negdata[np.where(data>0)] = 0    
        
        negrepgrid = coo_matrix((negdata,(rows,cols)), shape=(self.nx,self.ny))
        
        nuneg = np.zeros((self.nx,self.ny)) - negrepgrid.todense() + nu0[0]
        nupos = np.zeros((self.nx,self.ny)) + repgrid.todense() + nu0[1]
    
        pred = np.divide(nupos, nupos+nuneg)    
        variance = np.divide(np.multiply(nupos,nuneg), np.multiply(np.square(nupos+nuneg), nupos+nuneg+1))
        std = np.sqrt(variance)    
        return pred, std
    
    def removesea(self, bcc_pred):
        fname = self.datadir + "/seamap.csv"
        if self.sea_map==[]:        
            try:
                self.sea_map = np.genfromtxt(fname)
                if self.sea_map.shape[0] != bcc_pred.shape[0] or self.sea_map.shape[1] != bcc_pred.shape[1]:
                    self.sea_map = [] #reload it as shape has changed               
            except Exception:
                logging.info('Will recreate sea map matrix')
                        
        if self.sea_map != []:
            bcc_pred[np.where(self.sea_map==1)] = 0
            return bcc_pred        
        
        from point_in_polygon import cn_PnPoly
        poly1 = np.genfromtxt(self.datadir+"/haiti_polygon_1.csv", usecols=[0,1], delimiter=',')
        poly2 = np.genfromtxt(self.datadir+"/haiti_polygon_2.csv", usecols=[0,1], delimiter=',')
    
        #translate
        poly1x, poly1y = self.translate_points_to_local(poly1[:,1],poly1[:,0])
        poly2x, poly2y = self.translate_points_to_local(poly2[:,1], poly2[:,0])
        poly1 = np.concatenate((poly1x.reshape((len(poly1x),1)),poly1y.reshape((len(poly1y),1))), axis=1)
        poly2 = np.concatenate((poly2x.reshape((len(poly2x),1)),poly2y.reshape((len(poly2y),1))), axis=1)    
        
        extra_border = 0.02
        latborderfrac = extra_border/(self.maxlat-self.minlat)
        lonborderfrac = extra_border/(self.maxlon-self.minlon)
        xborder = np.ceil(latborderfrac*self.nx)
        yborder = np.ceil(lonborderfrac*self.ny)
        
        # points after which we don't care if it is in Haiti or not 
        #--> this goes over border into Dominican Rep.
        blehx, _ = self.translate_points_to_local(18.2, -72) 
        blehx2, blehy = self.translate_points_to_local(19.8, -72) 
            
        self.sea_map = np.zeros((self.nx,self.ny), dtype=np.int8)
        
        logging.info("Sea map loading...")
        for i in range(self.nx):
            logging.debug("Loading row " + str(i))
            for j in range(self.ny):       
                if i>blehx and i<blehx2 and j>=blehy:
                    continue               
                if not cn_PnPoly([i-xborder,j-yborder], poly1) and not cn_PnPoly([i-xborder,j-yborder], poly2) \
                and not cn_PnPoly([i+xborder,j-yborder], poly1) and not cn_PnPoly([i+xborder,j-yborder], poly2) \
                and not cn_PnPoly([i-xborder,j+yborder], poly1) and not cn_PnPoly([i-xborder,j+yborder], poly2) \
                and not cn_PnPoly([i+xborder,j+yborder], poly1) and not cn_PnPoly([i+xborder,j+yborder], poly2):
                    bcc_pred[i,j] = 0
                    self.sea_map[i,j] = 1
                    
        np.savetxt(fname, self.sea_map)            
        
        return bcc_pred

    def plotresults(self, bcc_pred, label='no idea', interp='none', imgmax=1, imgmin=0):
        bcc_pred = self.removesea(bcc_pred)
                
        dpi = 96
        if self.nx>=500:
            fig = plt.figure(frameon=False, figsize=(self.nx/dpi,self.ny/dpi))
        else:
            fig = plt.figure(frameon=False)
        plt.autoscale(tight=True)
    #     #For showing a 3D plot instead of flat heat map
    #     gridx = np.tile( range(1,self.nx+1),(self.ny,1))
    #     gridy = np.tile(range(1,self.ny+1), (self.nx,1)).transpose()
        
    #     ax = fig.add_subplot(1, 1, 1, projection='3d')    
    #     ax.plot_surface(gridx, gridy, bcc_pred, cstride=1, rstride=1, \
    #                     cmap=plt.get_cmap('spectral'), linewidth=0)
    #     ax.view_init(elev=90, azim=-90)
    
        #Can also try interpolation=nearest or none
        ax = fig.add_subplot(111)
        ax.set_axis_off()    
        
        plt.imshow(bcc_pred, cmap=plt.get_cmap('jet'), aspect=None, origin='lower', \
                   vmin=imgmin, vmax=imgmax, interpolation=interp, filterrad=0.01)
    
        fig.tight_layout(pad=0,w_pad=0,h_pad=0)
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())    
        
    def translate_points_to_local(self, latdata, londata):
        logging.debug('Translating original coords to local values.')
            
        latdata = np.float64(latdata)
        londata = np.float64(londata)
        
        normlatdata = (latdata-self.minlat)/(self.maxlat-self.minlat)
        normlondata = (londata-self.minlon)/(self.maxlon-self.minlon)    
            
        latdata = np.round(normlatdata*self.nx)
        londata = np.round(normlondata*self.ny)
            
        return latdata,londata    
        
    def tranlate_points_to_original(self, x,y):
        logging.debug('Tranlating our local points back to original lat/long')
        
        #normalise
        x = np.divide(np.float64(x),self.nx)
        y = np.divide(np.float64(y),self.ny)
        
        latdata = x*(self.maxlat-self.minlat) + self.minlat
        londata = y*(self.maxlon-self.minlon) + self.minlon
        
        return latdata,londata
        
    def load_ush_data(self):
        dataFile = self.datadir+'/exported_ushahidi.csv'
        self.K = 1
        #load the data
    #     reportIDdata = np.genfromtxt(dataFile, np.str, delimieter=',', skip_header=True, usecols=[])
    #     datetimedata = np.genfromtxt(dataFile, np.datetime64, delimiter=',', skip_header=True, usecols=[2,3])
        latdata = np.genfromtxt(dataFile, np.float64, delimiter=',', skip_header=True, usecols=[4])
        londata = np.genfromtxt(dataFile, np.float64, delimiter=',', skip_header=True, usecols=[5])
        reptypedata = np.genfromtxt(dataFile, np.str, delimiter=',', skip_header=True, usecols=[1])
        rep_list_all = np.genfromtxt(dataFile, np.str, delimiter=',', skip_header=True, usecols=[6])
        latdata,londata = self.translate_points_to_local(latdata,londata)
        rep_id_grid = {}

        rep_list = {}
        C = {}            
        for i, reptypetext in enumerate(reptypedata):        
            typetoks = reptypetext.split('.')
            typetoks = typetoks[0].split(',')
            for typestring in typetoks:
                maintype = typestring[0] #first character should be a number
                try:
                    typeID = int(maintype)
                    #print "Type ID found: " + str(typeID)
                    if typeID==3 and self.combine_cat_three_and_one:
                        typeID = 1
                except ValueError:
                    logging.warning('Not a report category: ' + typestring)
                    continue
                repx = latdata[i]
                repy = londata[i]
                
                if repx>=self.nx or repx<0 or repy>=self.ny or repy<0:
                    continue
                
                try:
                    Crow = np.array([0, repx, repy, 1]) # all report values are 1 since we only have confirmations of an incident, not confirmations of nothing happening
                except ValueError:
                    logging.error('ValueError creating a row of the crowdsourced data matrix.!')        
                if C=={} or typeID not in C:
                    C[typeID] = Crow.reshape((1,4))
                    rep_id_grid[typeID] = np.empty((self.nx, self.ny), dtype=np.object)
                    rep_list[typeID] = [rep_list_all[i]]
                else:
                    C[typeID] = np.concatenate((C[typeID], Crow.reshape(1,4)), axis=0)
                    rep_list[typeID].append(rep_list_all[i])
                    
                if rep_id_grid[typeID][repx, repy] == None:
                    rep_id_grid[typeID][repx, repy] = []
                rep_id_grid[typeID][repx, repy].append(len(rep_list[typeID])-1)               
                     
        self.C = C
        self.combiner = {} #reset as we have reloaded the data
        print "Number of type one reports: " + str(self.C[1].shape[0])
        
        return rep_id_grid, rep_list
  
    def insert_trusted(self, j, x, y, v, rep_id=-1, trust_acc=0, trust_var=0):
        #add a new reporter ID if necessary
        if rep_id == -1:
            rep_id = self.K
            self.K += 1
            
        if rep_id not in self.rep_ids:
            self.rep_ids.append(rep_id)
            alpha0new = np.zeros((2,2,1))
            if trust_acc ==0 and trust_var==0:
                alpha0new[:,:,0] = [[1000,1], [1, 1000]]
            else:
                alpha_sum = (trust_acc*(1-trust_acc))/trust_var - 1
                alpha_correct = alpha_sum * trust_acc
                alpha_wrong = alpha_sum * (1-trust_acc)
                
                logging.info("alpha: " + str(alpha_correct) + ", " + str(alpha_wrong))
                alpha0new[[0,1],[0,1],0] = alpha_correct
                alpha0new[[0,1],[1,0],0] = alpha_wrong
            if len(self.alpha0.shape)<3:
                self.alpha0 = self.alpha0.reshape((2,2,1))
            self.alpha0 = np.concatenate((self.alpha0, alpha0new), axis=2)
                  
        #add a trusted report to self.C, inserting it at the current self.timestep        
        x,y = self.translate_points_to_local(x,y)
        Crow = np.array([rep_id, x, y, v]).reshape((1,4))
        C = np.concatenate((self.C[j][0:self.timestep, :], Crow), axis=0)
        C = np.concatenate((C, self.C[j][self.timestep:, :]), axis=0)
        self.C[j] = C         
            
    def insert_trusted_prescripted(self, j):
        x,y = self.translate_points_to_local(18.52,-72.284)
        self.insert_trusted(1, x, y, 0, 1, 0.9, 0.01)
        self.insert_trusted(1, x+1, y, 0, 1, 0.9, 0.01)
        self.insert_trusted(1, x+2, y, 0, 1, 0.9, 0.01)
        self.insert_trusted(1, x+3, y, 0, 1, 0.9, 0.01)
        self.insert_trusted(1, x+1, y+1, 0, 1, 0.9, 0.01)
        self.insert_trusted(1, x+1, y-1, 0, 1, 0.9, 0.01)
        self.insert_trusted(1, x+2, y+1, 0, 1, 0.9, 0.01)
        self.insert_trusted(1, x+3, y-1, 0, 1, 0.9, 0.01)
        self.insert_trusted(1, x-1, y+1, 0, 1, 0.9, 0.01)
        self.insert_trusted(1, x-1, y-1, 0, 1, 0.9, 0.01)
        self.insert_trusted(1, x+2, y-1, 0, 1, 0.9, 0.01)
        self.insert_trusted(1, x+3, y+1, 0, 1, 0.9, 0.01)
    
    def write_img(self, label,j):
        plt.savefig(self.fileprefix+label+str(j)+'.png', bbox_inches='tight', \
                    pad_inches=0, transparent=True, dpi=96)
    
    def write_json(self, bcc_pred, j, label=""):
        jsonFile = self.fileprefix + label + str(j) + '.json'
        bcc_pred = bcc_pred
        bcc_pred = bcc_pred.tolist()
        with open(jsonFile, 'w') as fp:
            json.dump(bcc_pred, fp)
         
    def write_coords_json(self):
        jsonFile = self.webdatadir+'/mapdata/coords.json'
        with open(jsonFile, 'w') as fp:
            json.dump([self.minlat, self.maxlat, self.minlon, self.maxlon], fp)    
        
    def write_targets_json(self, p_list, target_rep_ids, target_list, j=1, targettypes=None, pi=None):
        jsonfile = self.webdatadir+'/targets_'+str(j)+'.json'
        #for now we will randomly assign some target types!
        if targettypes==None:
            targettypes = np.random.randint(0,4,(self.targetsx.size,1))
            
        targetids = self.targetids#np.arange(0,self.targetsx.size)    
            
        obj = np.concatenate((targetids[:,np.newaxis], self.targetsx[:,np.newaxis], self.targetsy[:,np.newaxis], targettypes), axis=1)
        obj = obj.tolist()
               
        #add list of associated targets
        for i in range(len(obj)):
            target_reports = [] 
            pi_list = []
            rep_ids_i = target_rep_ids[i]
            if rep_ids_i==None:
                logging.warning("no reports associated with this target")
            else:
                for idx in rep_ids_i:
                    target_reports.append(str(target_list[idx]))
                    agentid = self.C[j][idx,0]
                    if pi==None:
                        continue
                    pi_list.append(pi[:,:,agentid].tolist())

            obj[i].append(target_reports)
            #for each report, get a trust value
            obj[i].append(pi_list)
            obj[i].append(self.targetversions[i])
            
            
        with open(jsonfile, 'w') as fp:
            json.dump(obj, fp)
            
        return obj
        
#--------  MAIN TEST PROGRAM--------------------------------------------------
if __name__ == '__main__':
    heatmap = Heatmap(1000,1000)
    
    #High definition with no interpolation
    nx = 1000
    ny = 1000  
    heatmap.load_ush_data() 
    for j in range(1,2):
        rep_pred, rep_std = heatmap.reportintensity(j)
        heatmap.plotresults(rep_pred, label='Predicted Incidents of type '+str(j))
        #write_json(bcc_pred, nx,ny,j,label="_rep_intensity_")
        heatmap.write_img("_rep_intensity_", j)
           
        heatmap.plotresults(rep_std, label='Uncertainty (S.D.) in Pr(incident) of type '+str(j))
        #write_json(stdPred, nx,ny,j, label="_sd_")
        heatmap.write_img("_rep_intensity__sd_",j)   
    
    #Using BCC - lower res so it is tractable to interpolate
    heatmap.nx = 1000
    heatmap.ny = 1000
    heatmap.load_ush_data() # in case nx and ny have changed, need to reload
           
    for j in range(1,2):
        start = time.time()
        bcc_pred,combiner = heatmap.runBCC(j)
        fin = time.time()
        print "bcc heatmap prediction timer (no loops): " + str(fin-start)
                        
        bcc_mpr = combiner.getmean()
        heatmap.plotresults(bcc_pred, label='Predicted Incidents of type '+str(j))
        #write_json(bcc_pred, nx,ny,j)
        heatmap.write_img("", j)
                 
        heatmap.plotresults(bcc_mpr,  label='Incident Rate of type '+str(j))
        #write_json(combiner.mPr, nx,ny,j, label="_mpr_")
        heatmap.write_img("_mpr_",j)
         
        bcc_stdPred = combiner.getsd()
        heatmap.plotresults(bcc_stdPred,  label='Uncertainty (S.D.) in Pr(incident) of type '+str(j))
        #write_json(stdPred, nx,ny,j, label="_sd_")
        heatmap.write_img("_sd_",j)
     
    #insert a trusted report at 18.5333 N, -72.3333 W     
    heatmap.nx = 2000
    heatmap.ny = 2000
    heatmap.load_ush_data()
    heatmap.insert_trusted_prescripted(1)
    for j in range(1,2):
        rep_pred, rep_std = heatmap.reportintensity(j)
        heatmap.plotresults(rep_pred, label='Predicted Incidents of type '+str(j))
        #write_json(bcc_pred, nx,ny,j,label="_rep_intensity_")
        heatmap.write_img("_rep_intensity__expert_", j)
           
        heatmap.plotresults(rep_std, label='Uncertainty (S.D.) in Pr(incident) of type '+str(j))
        #write_json(stdPred, nx,ny,j, label="_sd_")
        heatmap.write_img("_rep_intensity__sd__expert_",j)      
          
    heatmap.nx = 2000
    heatmap.ny = 2000  
    heatmap.load_ush_data() 
    heatmap.insert_trusted_prescripted(1)
    for j in range(1,2):
        bcc_pred2,combiner2 = heatmap.runBCC(j)
        heatmap.plotresults(bcc_pred2, label='Predicted Incidents of type '+str(j))
#         write_json(bcc_pred, nx,ny,j, label="_expert_")
        heatmap.write_img("_expert_",j)        
  
        bcc_mpr2 = combiner2.getmean()
        heatmap.plotresults(bcc_mpr2, label='Incident Rate of type '+str(j))
#         write_json(combiner.mPr, nx,ny,j, label="_mpr_expert_")
        heatmap.write_img("_mpr_expert_",j)
          
        bcc_stdPred2 = combiner2.getsd()

        heatmap.plotresults(bcc_stdPred2, label='Uncertainty (S.D.) in Pr(incident) of type '+str(j))
#         write_json(stdPred, nx,ny,j, label="_sd_expert_")        
        heatmap.write_img("_sd__expert_",j)        