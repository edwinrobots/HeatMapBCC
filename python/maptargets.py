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

class MapTargets(object):
    '''
    classdocs
    '''

    target_threshold = 0.75#0.775
    
    #storing target info
    targetsx = []
    targetsy = []
    targetids = []
    targetversions = []
    
    provdoc = None
    provbundle = None
    provdoc_id = -1
    provfilelist = []    
    
    plist = []
    target_rep_ids = []
    rep_list = None
    rep_id_grid = None
    
    heatmap = None

    def __init__(self, heatmap):
        self.heatmap = heatmap
        
    def calculate_targets(self, pgrid, j=1):
        targetsx, targetsy, bgrid = self.find_peaks(pgrid, j)
        
        dist = np.zeros((len(targetsx),len(self.targetsx)))
        
        newtargetids = np.zeros(targetsx.shape)-1
        newtargetversions = np.zeros(targetsx.shape)
        
        if len(self.targetids)<1:
            self.targetsx = targetsx
            self.targetsy = targetsy
            self.targetids = np.arange(len(targetsx))
            self.targetversions = np.zeros(self.targetids.shape)
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
            
                radius = 20
                
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
        #write to prov store
        api = Api(username='atomicorchid', api_key='2ce8131697d4edfcb22e701e78d72f512a94d310')
        ao = Namespace('ao', 'https://provenance.ecs.soton.ac.uk/atomicorchid/ns#')
        
        self.provdoc = ProvDocument()
        self.provdoc.add_namespace(ao)
        self.provdoc.set_default_namespace('https://provenance.ecs.soton.ac.uk/atomicorchid/data/1/')
                    
        self.provbundle = self.provdoc.bundle('crowd_scanner')#:'+str(bundle_id))
        b = self.provbundle    
#         d = self.provdoc
        cs = b.agent('CrowdScanner')

        rep_entities = {}             
        for i, tdata in enumerate(tlist):
            tid = int(tdata[0])
            x = tdata[1]
            y = tdata[2]
            v = int(tdata[6])
            
            target = b.entity('target/'+str(tid))
            target_v0 = b.entity('target/'+str(tid)+'.'+str(v), {'ao:longitude': str(x), 'ao:latitude': str(y)})
            target_v0.specializationOf(target)
            for r in self.target_rep_ids[i]:
                if r not in rep_entities:
                    Crow = C[r,:]
                    x = Crow[1]
                    y = Crow[2]
                    rep_entities[r] = b.entity('crowdreport/'+str(r), {'ao:longitude':str(x), 'ao:latitude':str(y)})
                target_v0.wasDerivedFrom(rep_entities[r])
            target_v0.wasAttributedTo(cs)
        
        provstore_document = api.document.create(self.provdoc, name='cs-targets', public=True)
        self.provdoc_id = provstore_document.id        
        #provstore_document = api.document.get(self.provdoc_id)
        #provstore_document.add_bundle(self.provdoc,'crowd_scanner:'+str(bundle_id))                
        #self.provdoc_id = provstore_document.id
        document_uri = provstore_document.url
        logging.info("prov doc URI: " + str(document_uri))
        self.provfilelist.append(self.provdoc_id)

        jsonfile = self.heatmap.datadir+"/provfilelist.json"
        if self.provfilelist==[] and os.path.isfile(jsonfile):
            with open(jsonfile,'r') as fp:
                self.provfilelist = json.load(fp)
        
        with open(jsonfile, 'w') as fp:
            json.dump(self.provfilelist, fp)
        
        
    def write_targets_json(self, update_number, alpha, C, j=1, targettypes=None):
        
        rep_list = self.rep_list[j]
        
        pi = alpha / np.sum(alpha, axis=1).reshape((2,1,alpha.shape[2]))
 
        jsonfile = self.heatmap.webdatadir+'/targets_'+str(j)+'.json'
        #for now we will randomly assign some target types!
        if targettypes==None:
            targettypes = np.random.randint(0,4,(self.targetsx.size,1))
            
        targetids = self.targetids#np.arange(0,self.targetsx.size)    
            
        obj = np.concatenate((targetids[:,np.newaxis], self.targetsx[:,np.newaxis], \
                              self.targetsy[:,np.newaxis], targettypes), axis=1)
        obj = obj.tolist()
               
        #add list of associated targets
        for i in range(len(obj)):
            target_reports = [] 
            pi_list = []
            rep_ids_i = self.target_rep_ids[i]
            if rep_ids_i==None:
                logging.warning("no reports associated with this target")
            else:
                for idx in rep_ids_i:
                    target_reports.append(str(rep_list[idx]))
                    agentid = C[idx,0]
                    if pi==None:
                        continue
                    pi_list.append(pi[:,:,agentid].tolist())

            obj[i].append(target_reports)
            #for each report, get a trust value
            obj[i].append(pi_list)
            obj[i].append(self.targetversions[i])
            
            
        self.write_targets_prov(obj, C, update_number)
            
        with open(jsonfile, 'w') as fp:
            json.dump(obj, fp)
            
        return obj    