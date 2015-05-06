'''
Created on 9 Sep 2014

@author: edwin
'''
import numpy as np
import logging, json
import os.path

class MapTargets(object):
    '''
    classdocs
    '''

    target_threshold = 0.75#0.775
    radius = 5 #radius in which to look for peaks
    
    #storing target info
    targetsx = []
    targetsy = []
    targetids = []
    changedtargets = []
    targets_to_invalidate = [] # invalid targets that have not been invalidated on the prov store
        
    heatmap = None
    
    targets = {}
    targetversions = {} #the latest version entity for each target id
    targetversion_nos = None
    targets_to_invalidate_version_nos = [] #version numbers of targets waiting to be invalidated
    
    def __init__(self, heatmap):
        self.heatmap = heatmap

    def calculate_targets(self, pgrid, j=1):
        targetsx, targetsy, bgrid = self._find_peaks(pgrid, j)
        
        dist = np.zeros((len(targetsx),len(self.targetsx)))
        
        newtargetids = np.zeros(targetsx.shape, dtype=np.int)-1
        newtargetversions = np.zeros(targetsx.shape, dtype=np.int)
        self.changedtargets = np.ones(targetsx.shape)
        
        if len(self.targetids)<1:
            self.targetsx = targetsx
            self.targetsy = targetsy
            self.targetids = np.arange(len(targetsx))
            self.targetversion_nos = np.zeros(self.targetids.shape, dtype=np.int)
            return bgrid
        
        #see if the peaks are close matches to existing targets
        for t in range(len(targetsx)):
            x = targetsx[t]
            y = targetsy[t]
            
            #calculate distances to old targets
            dist_t = np.sqrt((x-self.targetsx)**2 + (y-self.targetsy)**2)
            dist[t,:] = dist_t
            
        #go through looking for most similar peaks first
        
        nIterations = len(targetsx)
        num_new_ids = nIterations
        if nIterations>len(self.targetsx):
            nIterations = len(self.targetsx)
            
        for _ in range(nIterations):
            closest_old_to_new = np.argmin(dist,axis=1)
            mindist_old_to_new = np.min(dist,axis=1)
        
            least_moved_new = np.argmin(mindist_old_to_new)
            least_moved_old = closest_old_to_new[least_moved_new]
            
            shortestdist = dist[least_moved_new,least_moved_old]
            if shortestdist > 0:
                logging.info("Distance between new and old targets: " + str(shortestdist))
            
            if dist[least_moved_new,least_moved_old] > float(self.radius)/float(self.heatmap.nx):
                break
            
            newtargetids[least_moved_new] = self.targetids[least_moved_old]
            
            if targetsx[least_moved_new]==self.targetsx[least_moved_old] \
                and targetsy[least_moved_new]==self.targetsy[least_moved_old]:
                newtargetversions[least_moved_new] = self.targetversion_nos[least_moved_old]
            else:
                newtargetversions[least_moved_new] = self.targetversion_nos[least_moved_old]+1
                self.changedtargets[least_moved_new] = 1
            dist[:,least_moved_old] = np.Inf
            dist[least_moved_new,:] = np.Inf
            
            num_new_ids -=1
            
        if num_new_ids>0:
            missingid_idxs = np.argwhere(newtargetids<0)
            max_id_so_far = np.max(self.targetids.keys())
            missingids = range(max_id_so_far+1,max_id_so_far+num_new_ids+1)
            newtargetids[missingid_idxs] = missingids
            self.changedtargets[missingid_idxs] = 1
            
        self.targets_to_invalidate = []
        self.targets_to_invalidate_version_nos = []
        for i, tid in enumerate(self.targetids):
            if tid not in newtargetids:
                self.targets_to_invalidate.append(tid)
                self.targets_to_invalidate_version_nos.append(self.targetversion_nos[i]+1)
            
        self.targetids = newtargetids
        self.targetversion_nos = newtargetversions
                
        self.targetsx = targetsx
        self.targetsy = targetsy        
        
        logging.info("Maximimum target ID is " + str(np.max(self.targetids)))
        logging.info("Number of targets found: " + str(len(self.targetids)))
                
        return bgrid

    def _move_reps(self, rgrid, x, y, newx, newy):
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

    def _find_peaks(self, pgrid, j):
        #Turn a grid of predictions of events, e.g. bcc_pred, into a set of binary points
        #representing the most likely events. 
        #find points > theta        
        if self.target_threshold==0:
            theta = np.max(pgrid) - 0.12
        else:
            theta = self.target_threshold
        
        bgrid = np.array(pgrid>theta, dtype=np.int8)
        
        for x in np.arange(bgrid.shape[0]):
            for y in np.arange(bgrid.shape[1]):
                if bgrid[x,y]==0:
                    continue
                
                #find highest neighbour
                highestx = x
                highesty = y
                highestp = pgrid[x,y]
            
                radius = self.radius                
                for i in range(1,radius):
                    for j in range(1,radius):        
                        if highestp <= pgrid[x+i,y] and bgrid[x+i,y]>-1:
                            highestx = x+i
                            highesty = y
                            highestp = pgrid[x+i, y]
                        elif highestp <= pgrid[x-i,y] and bgrid[x-i,y]>-1:
                            highestx = x-i
                            highesty = y
                            highestp = pgrid[x-i, y]   
                        elif highestp <= pgrid[x,y-j] and bgrid[x,y-j]>-1:
                            highestx = x
                            highesty = y-j
                            highestp = pgrid[x, y-j]  
                        elif highestp <= pgrid[x,y+j] and bgrid[x,y+j]>-1:
                            highestx = x
                            highesty = y+j
                            highestp = pgrid[x, y+j]
                        elif highestp <= pgrid[x+i,y+j] and bgrid[x+i,y+j]>-1:
                            highestx = x+i
                            highesty = y+j
                            highestp = pgrid[x+i, y+j]
                        elif highestp <= pgrid[x-i,y-j] and bgrid[x-i,y-j]>-1:
                            highestx = x-i
                            highesty = y-j
                            highestp = pgrid[x-i, y-j]   
                        elif highestp <= pgrid[x+i,y-j] and bgrid[x+i,y-j]>-1:
                            highestx = x+i
                            highesty = y-j
                            highestp = pgrid[x+i, y-j]  
                        elif highestp <= pgrid[x-i,y+j] and bgrid[x-i,y+j]>-1:
                            highestx = x-i
                            highesty = y+j
                            highestp = pgrid[x-i, y+j]  
                            
                        if highestx!=x or highesty!=y:
                            break                      
                    
                if highestx!=x or highesty!=y:
                    bgrid[x,y] = -1
                else:
                    logging.info("target found at " + str(x) + ", " + str(y))
                                   
        bgrid[bgrid==-1] = 0
                                      
        target_list = np.argwhere(bgrid>0)
        targetsx = target_list[:,0]
        targetsy = target_list[:,1]
        
        targetsx, targetsy = self.heatmap.tranlate_points_to_original(targetsx, targetsy)
        return targetsx, targetsy, bgrid
        
    def savelocalrecord(self):
        jsonfile = self.heatmap.datadir+"/provfilelist.json"
        if self.provfilelist==[] and os.path.isfile(jsonfile):
            with open(jsonfile,'r') as fp:
                self.provfilelist = json.load(fp)
        else:
            with open(jsonfile, 'w') as fp:
                json.dump(self.provfilelist, fp, indent=2)
        
        
    def write_targets_json(self, update_number, alpha, C, j=1, targettypes=None):
        jsonfile = self.heatmap.webdatadir+'/targets_'+str(j)+'.json'
        
        #get the data ready
        #for now we will randomly assign some target types!
        if targettypes==None:
            targettypes = np.random.randint(0,4,(self.targetsx.size,1))
            
        #Create the list object with basic attributes: Columns 0 to 3
        listobj = np.concatenate((self.targetids[:,np.newaxis], self.targetsx[:,np.newaxis], \
                              self.targetsy[:,np.newaxis], targettypes), axis=1)
        listobj = listobj.tolist()               
               
        #Add lists of associated reports and confusion matrices
        for i in range(len(listobj)):
            listobj[i][0] = int(listobj[i][0])
            listobj[i][3] = int(listobj[i][3])
            listobj[i].append(int(self.targetversion_nos[i])) #Column 4
            
        with open(jsonfile, 'w') as fp:
            json.dump(listobj, fp, indent=2)