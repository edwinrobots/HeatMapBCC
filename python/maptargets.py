'''
Created on 9 Sep 2014

@author: edwin
'''
import numpy as np
import logging, json
from copy import deepcopy
from prov.model import ProvDocument, Namespace
from provstore.api import Api
import os.path
import time
import datetime

class MapTargets(object):
    '''
    classdocs
    '''

    target_threshold = 0.85#0.775
    radius = 5 #radius in which to look for peaks
    
    #storing target info
    targetsx = []
    targetsy = []
    targetids = []
    targetversions = []
    changedtargets = []
    targets_to_invalidate = [] # invalid targets that have not been invalidated on the prov store
        
    plist = []
    target_rep_ids = []
    rep_list = None
    rep_id_grid = None
    
    heatmap = None
    
    provfilelist = []  
    postedreports = {} #reports that have already been posted to provenance server  
    api = None
    namespace = None
    game_id = 13
    defaultns = 'https://provenance.ecs.soton.ac.uk/atomicorchid/data/%s/'
    targets = {}
    targetversions = {} #the latest version entity for each target id
    targetversion_nos = None
    targets_to_invalidate_version_nos = [] #version numbers of targets waiting to be invalidated
    
    document_id = -1 # the provenance document id in the prov store
    
    def __init__(self, heatmap):
        self.heatmap = heatmap
        self.api = Api(username='atomicorchid', api_key='2ce8131697d4edfcb22e701e78d72f512a94d310')
        self.namespace = Namespace('ao', 'https://provenance.ecs.soton.ac.uk/atomicorchid/ns#')
        
    def calculate_targets(self, pgrid, j=1):
        targetsx, targetsy, bgrid = self.find_peaks(pgrid, j)
        
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
            max_id_so_far = np.max(self.targets.keys())
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

    def find_peaks(self, pgrid, j):
        rep_id_grid = self.rep_id_grid[j]
        #Turn a grid of predictions of events, e.g. bcc_pred, into a set of binary points
        #representing the most likely events. 
        #find points > theta        
        if self.target_threshold==0:
            theta = np.max(pgrid) - 0.12
        else:
            theta = self.target_threshold
        
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
                    rgrid = self.move_reps(rgrid, x, y, highestx, highesty)
                else:
                    logging.info("target found at " + str(x) + ", " + str(y))
                                   
        bgrid[bgrid==-1] = 0
                                      
        hasreports = np.argwhere(rgrid)
        hasreportsgrid = np.zeros(bgrid.shape)
        hasreportsgrid[hasreports[:,0],hasreports[:,1]] = 1
        bgrid = bgrid * hasreportsgrid
        
        target_list = np.argwhere(bgrid>0)
        targetsx = target_list[:,0]
        targetsy = target_list[:,1]
        
        self.target_rep_ids = rgrid[targetsx, targetsy]
                
        p_list = pgrid[targetsx,targetsy]     
        self.plist = np.around(p_list,2), 
        targetsx, targetsy = self.heatmap.tranlate_points_to_original(targetsx, targetsy)
        return targetsx, targetsy, bgrid
        
    def write_targets_prov(self, tlist, C, bundle_id):
        #Initialisation
#         cs = b.agent('CrowdScanner')
        
        if self.document_id == -1:
            d = ProvDocument()
            d.add_namespace(self.namespace)
            d.set_default_namespace(self.defaultns % self.game_id)
            
            provstore_document = self.api.document.create(d, name="Game%s CrowdScanner" % self.game_id, public=True)
            document_uri = provstore_document.url
            logging.info("prov doc URI: " + str(document_uri))
            self.provfilelist.append(provstore_document.id)
            self.savelocalrecord()
            self.document_id = provstore_document.id
        
        b = ProvDocument()  # Create a new document for this update
        b.add_namespace(self.namespace)
        b.set_default_namespace(self.defaultns % self.game_id)            
            
        # cs to be used with all targets
        cs = b.agent('agent/CrowdScanner', (('prov:type', 'ao:IBCCAlgo'), ('prov:type', 'prov:SoftwareAgent')))
        
        timestamp = time.time()  # Record the timestamp at each update to generate unique identifiers        
        startTime = datetime.datetime.fromtimestamp(timestamp)
        endTime = startTime
        activity = b.activity('activity/cs/update_report_%s' % timestamp, startTime, endTime)
        activity.wasAssociatedWith(cs)

        #Add target and report entities
        for i, tdata in enumerate(tlist):
            if self.changedtargets[i]==0:
                continue
            
            #Target entity for target i
            tid = int(tdata[0])
            x = tdata[1]
            y = tdata[2]
#             targettype = tdata[3] #don't record here, it will be revealed and recorded by UAVs
            v = int(tdata[4])
            agentids = tdata[7]
            
            targetattributes = {'ao:longitude': str(x), 'ao:latitude': str(y), }
            #'ao:asset_type':str(targettype)}
            target_v0 = b.entity('cs/target/'+str(tid)+'.'+str(v), targetattributes)            
            #Post the root report if this is the first version
            if v==0:
                self.targets[tid] = b.entity('cs/target/'+str(tid))
            else:
                try:
                    target_v0.wasDerivedFrom(self.targetversions[tid])
                except KeyError:
                    logging.error("Got a key error for key " + str(tid) + ', which is supposed to be version' + str(v))
            self.targetversions[tid] = target_v0                    
            target_v0.specializationOf(self.targets[tid])
            target_v0.wasAttributedTo(cs)
            
            #Report entities for origins of target i
            for j, r in enumerate(self.target_rep_ids[i]):
                if r not in self.postedreports:
                    Crow = C[r,:]
                    x = Crow[1]
                    y = Crow[2]
                    reptext = tdata[5][j].decode('utf8')
                    agentid = agentids[j]
                    
                    reporter_name = 'agent/crowdreporter%s' % agentid
                    b.agent(reporter_name, (('prov:type', 'ao:CrowdReporter'), ('prov:type', 'prov:Person')))
                    
                    reportattributes = {'ao:longitude':str(x), 'ao:latitude':str(y), 'ao:report': reptext}
                    
                    self.postedreports[r] = b.entity('cs/report/'+str(r), reportattributes)
                    self.postedreports[r].wasAttributedTo(reporter_name)
                activity.used(self.postedreports[r])
                target_v0.wasDerivedFrom(self.postedreports[r])
        
        #Invalidate old targets no longer in use
        for i,tid in enumerate(self.targets_to_invalidate):
            target_v = self.targetversions[tid]
            b.wasInvalidatedBy(target_v, activity)
            
        #Post the document to the server
        #bundle = b.bundle('crowd_scanner')
        bundle_id = 'bundle/csupdate/%s' % timestamp
        self.api.add_bundle(self.document_id, b.serialize(), bundle_id)
        
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
        rep_list = self.rep_list[j]        
        pi = alpha / np.sum(alpha, axis=1).reshape((2,1,alpha.shape[2]))
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
              
            target_reports = [] 
            pi_list = []
            rep_ids_i = self.target_rep_ids[i]
            agentids = []
            if rep_ids_i==None:
                logging.warning("No reports associated with target " + str(i))
            else:
                for idx in rep_ids_i:
                    target_reports.append(str(rep_list[idx]))
                    agentid = C[idx,0]
                    #logging.info("reporter:  " + str(agentid))
                    if pi==None:
                        continue
                    if agentid < pi.shape[2]:
                        pi_list.append(pi[:,:,agentid].tolist())
                    else:
                        pi_list.append(self.heatmap.alpha0[:,:,0].tolist())
                    agentids.append(int(agentid))

            listobj[i].append(target_reports) # Column 6
            listobj[i].append(pi_list) # Column 6
            #Include the version number as final entry in the list
            listobj[i].append(agentids) # column 7          
            
        #Write the provenance and save the json
        self.write_targets_prov(listobj, C, update_number)
        with open(jsonfile, 'w') as fp:
            json.dump(listobj, fp, indent=2)