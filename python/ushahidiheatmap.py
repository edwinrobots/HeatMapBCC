'''
Created on 23 Jun 2014

@author: edwin
'''
import heatmapbcc, maptargets, time, logging, json
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D #Can use if we want to make a 3D plot instead of flat heat map
from scipy.sparse import coo_matrix
import os
import provcleanup
import shutil
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
        
    nx = 100
    ny = 100
    
    C = []
    K = 1
    rep_ids = []
        
    startclean = True #if true, will delete all previous maps before running
    timestep = 65 #max is likely to be 765
    stepsize = 100 #takes around 4 minutes to run through all updates. There will be 7 updates
    finalsnapshot = False

    #If run_script_only is set to true, it will run the update loop when called until all scripted reports have been included.
    #If any subsequent reports arrive, it will run only one update to include them.
    run_script_only=False 
    
    running = False
    
    alpha0 = []
    nu0 = []
    
    #Dictionary defining how we map original categories 
    #to class IDs, using original categories as keys:
    categorymap = {1:1, 3:1,5:{'a':1},6:{'a':1,'b':1}} 

    targetextractor = None
        
    def __init__(self, nx,ny, run_script_only=False, minlat=None,maxlat=None, minlon=None,maxlon=None, fileprefix=None):
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
        self.targetextractor = maptargets.MapTargets(self)
        if self.startclean:
            self.initial_cleanup()
        self.load_ush_data() 
        self.write_coords_json()     
        
        self.run_script_only = run_script_only   
        
    def runBCC(self, j):
        C = self.C[j]
        self.runBCC_subset(C)
        
    def runBCC_up_to_time(self, j, timestep):       
        C = self.C[j]
        C = C[0:timestep,:]
        bcc_pred, _ = self.runBCC_subset(C)
        return bcc_pred
            
    def runBCC_subset(self, C, j=1):
        if j not in self.combiner or self.combiner[j]==None or self.combiner[j].K<self.K:
            self.combiner[j] = heatmapbcc.Heatmapbcc(self.nx, self.ny, 2, 2, self.alpha0, self.nu0, self.K)
            self.combiner[j].minNoIts = 5
            self.combiner[j].maxNoIts = 200
            self.combiner[j].convThreshold = 0.1
            self.combiner[j].useLowerBound = False

        bcc_pred = self.combiner[j].combineClassifications(C)
        bcc_pred = bcc_pred[j,:,:].reshape((self.nx,self.ny))
        return bcc_pred, self.combiner[j]

    #@profile
    def loop_iteration(self, j, t):
        bcc_pred = self.runBCC_up_to_time(j,t)
        return bcc_pred
              
    def enlarge_target_blobs(self, target_grid, nsize=5):
        #make neighbouring points also one
        pospoints = np.argwhere(target_grid)
        for row in pospoints:
            xblob = np.arange(row[0]-nsize,row[0]+nsize)
            yblob = np.arange(row[1]-nsize,row[1]+nsize)
            for x in xblob:
                target_grid[x, yblob] = 1
        return target_grid
          
    def initial_cleanup(self, j=1):
        self.rem_img("", j)
        self.rem_img("_sd_", j)
        self.rem_img("_rep_intensity_", j)
        self.rem_img("_rep_intensity__sd_", j)
        self.rem_img("_targets_", j)
        
        targetsjsonfile = self.webdatadir+'/targets_'+str(j)+'.json'
        self.del_data_file(targetsjsonfile)
        with open(targetsjsonfile, 'w') as fp:
            json.dump([], fp)
        
        provcleanup.cleanup()
                    
    def timed_update_loop(self, j=1):
        logging.info("Run BCC at intervals, loading new reports.")
        
        #Intially, we are running. If this flag is changed by another thread, we'll stop.
        self.running = True 

        #Call this in a new thread that can be updated by POST to the web server. 
        #When a new report is received by POST to web server, the server can kill this thread, call insert_trusted and then restart this method in a new thread
        nupdates = self.C[j].shape[0]
        
        if self.finalsnapshot:
            self.timestep = nupdates
            logging.warning("Breaking the gradual update so we skip to final update loop")
        
        while self.running:
            logging.info("timed_update_loop timestep " + str(self.timestep))
            starttime = time.time()

            bcc_pred = self.loop_iteration(j, self.timestep)

            if not self.running:
                logging.info("Stopping update loop for the heatmap")
                break

            self.plotresults(bcc_pred, label='Predicted Incidents of type '+str(j))
            self.write_img("", j)
  
            bcc_stdPred = self.combiner[j].getsd()
            #bcc_stdPred = np.sqrt(bcc_pred*(1-bcc_pred))#
            #normalise it
            maxunc = np.max(bcc_stdPred)
            minunc = np.min(bcc_stdPred)
            bcc_stdPred = (bcc_stdPred-minunc)/(maxunc-minunc)
            lab = 'Uncertainty (S.D.) in Pr(incident) of type '+str(j)
            self.plotresults(bcc_stdPred, label=lab, removesea=True)
            self.write_img("_sd_",j)
            
            rep_pred, rep_std = self.reportintensity(j, self.timestep)
            self.plotresults(rep_pred, label='Predicted Incidents of type '+str(j))
            self.write_img("_rep_intensity_", j)
            
            maxunc = np.max(rep_std)
            minunc = np.min(rep_std)
            rep_std = (rep_std-minunc)/(maxunc-minunc)
            lab = 'Uncertainty (S.D.) in Pr(incident) of type '+str(j)
            self.plotresults(rep_std, label=lab, removesea=True)
            self.write_img("_rep_intensity__sd_",j)   
                        
            target_grid = self.targetextractor.calculate_targets(bcc_pred)
            lab = 'Predicted target points of type ' + str(j)
            self.plotresults(self.enlarge_target_blobs(target_grid), lab)
            self.write_img("_targets_", j)
            self.targetextractor.write_targets_json(self.timestep, self.combiner[j].alpha, self.C[j])
            
            plt.close("all")
            
            endtime = time.time()
            
            waittime = 5 # make sure we have left a pause between updates
            if endtime-starttime < waittime:
                time.sleep(waittime-endtime+starttime)
                
            logging.info("Update loop took " + str(endtime-starttime) + " seconds.")
            
            #once complete, update the current time step
            nupdates = self.C[j].shape[0]
            if self.timestep >= nupdates and self.run_script_only:
                    self.running = False
            else:
                self.timestep += self.stepsize
            

            if self.timestep > nupdates:
                self.timestep = nupdates

        self.running = False
            
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

    def plotresults(self, bcc_pred, label='no idea', interp='none', imgmax=1, imgmin=0, removesea=False):
        if removesea:
            bcc_pred = self.removesea(bcc_pred)
                
        dpi = 96
        if self.nx>=500:
            fig = plt.figure(frameon=False, figsize=(float(self.nx)/dpi,float(self.ny)/dpi))
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
                
        cmap = plt.get_cmap('jet')                
        cmap._init()
        
        prior = self.nu0/float(np.sum(self.nu0))
        prior = prior[0]
        
        changepoint = prior*cmap.N + 3
                
        if removesea:
            alphas = np.linspace(0, 0.75, cmap.N+3)
        else:
            alphas1 = np.linspace(1, 0.6, changepoint-20)
            alphas2 = np.linspace(0.6, 0, 20)
            alphas3 = np.linspace(0, 0.6, 20)
            alphas4 = np.linspace(0.6, 1, changepoint - 23)
            alphas = np.concatenate((alphas1, alphas2, alphas3, alphas4))
        cmap._lut[:,-1] = alphas        
        
        plt.imshow(bcc_pred, cmap=cmap, aspect=None, origin='lower', \
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
        instantiatedagents = []

        for i, reptypetext in enumerate(reptypedata):        
            typetoks = reptypetext.split('.')
            typetoks = typetoks[0].split(',')
            
            if "1a." in reptypetext:
                agentID = 0
                maintype = 1
            elif "1b." in reptypetext:
                agentID = 1
                maintype = 1
            elif "1c." in reptypetext:
                agentID = 2
                maintype = 1
            elif "1d." in reptypetext:
                agentID = 3
                maintype = 1
            elif "3a." in reptypetext:
                agentID = 4
                maintype = 1
            elif "3b." in reptypetext:
                agentID = 5
                maintype = 1
            elif "3c." in reptypetext:
                agentID = 6
                maintype = 1
            elif "3d." in reptypetext:
                agentID = 7
                maintype = 1
            elif "3e." in reptypetext:
                agentID = 8
                maintype = 1
            elif "5a." in reptypetext:
                agentID = 9
                maintype = 1
            elif "6a." in reptypetext:
                agentID = 10
                maintype = 1
            elif "6b." in reptypetext:
                agentID = 11
                maintype = 1
            elif "1." in reptypetext:
                agentID = 12
                maintype = 1
            elif "3." in reptypetext:
                agentID = 13
                maintype = 1  
            else: #we don't care about these categories in the demo anyway, but can add them easily here
                agentID = 14
                for typestring in typetoks:
                    if len(typestring)>1:
                        sectype = typestring[1] # second character should be a letter is available
                    else:
                        sectype = 0
                    try:
                        maintype = int(typestring[0]) #first character should be a number
                    except ValueError:
                        logging.warning('Not a report category: ' + typestring)
                        continue                   
                    if maintype in self.categorymap:
                        mappedID = self.categorymap[maintype]
                        if type(mappedID) is dict:
                            if sectype in mappedID:
                                maintype = mappedID[sectype]
                        else:
                            maintype = mappedID

            repx = latdata[i]
            repy = londata[i]
            if repx>=self.nx or repx<0 or repy>=self.ny or repy<0:
                continue
            
            try:
                Crow = np.array([agentID, repx, repy, 1]) # all report values are 1 since we only have confirmations of an incident, not confirmations of nothing happening
            except ValueError:
                logging.error('ValueError creating a row of the crowdsourced data matrix.!')        

            if C=={} or maintype not in C:
                C[maintype] = Crow.reshape((1,4))
                rep_id_grid[maintype] = np.empty((self.nx, self.ny), dtype=np.object)
                rep_list[maintype] = [rep_list_all[i]]
            else:
                C[maintype] = np.concatenate((C[maintype], Crow.reshape(1,4)), axis=0)
                rep_list[maintype].append(rep_list_all[i])
                
            if rep_id_grid[maintype][repx, repy] == None:
                rep_id_grid[maintype][repx, repy] = []
            rep_id_grid[maintype][repx, repy].append(len(rep_list[maintype])-1)               
            
            if agentID not in instantiatedagents:
                instantiatedagents.append(agentID)
                
        instantiatedagents = np.array(instantiatedagents)
        for j in C.keys():
            for r in range(C[j].shape[0]):
                C[j][r,0] = np.argwhere(instantiatedagents==C[j][r,0])[0,0]
        self.K = len(instantiatedagents)
                     
        self.C = C
        self.combiner = {} #reset as we have reloaded the data
        print "Number of type one reports: " + str(self.C[1].shape[0])
        
        self.targetextractor.rep_list = rep_list
        self.targetextractor.rep_id_grid = rep_id_grid
  
    def insert_trusted(self, j, x, y, v, rep_id=-1, trust_acc=0, trust_var=0):
        #add a new reporter ID if necessary
        if rep_id == -1 or rep_id>=self.K:
            if len(self.alpha0.shape)<3:
                self.alpha0 = np.tile(self.alpha0.reshape((2,2,1)),(1,1,self.K))            
            
            rep_id = self.K
            self.K += 1
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
    
            self.alpha0 = np.concatenate((self.alpha0, alpha0new), axis=2)
                  
        #add a trusted report to self.C, inserting it at the current self.timestep        
        x,y = self.translate_points_to_local(x,y)
        logging.info("Received new report at local point " + str(x) + ", " + str(y))
        Crow = np.array([rep_id, x, y, v]).reshape((1,4))
        C = np.concatenate((self.C[j][0:self.timestep, :], Crow), axis=0)
        C = np.concatenate((C, self.C[j][self.timestep:, :]), axis=0)
        self.timestep += 1
        self.C[j] = C
        
        if v==1:
            self.targetextractor.rep_list.append("Aid agency confirms emergency")
        else:
            self.targetextractor.rep_list[j].append("Aid agency reports that there are no emergencies in this area.")
             
        if self.targetextractor.rep_id_grid[j][x, y] == None:
            self.targetextractor.rep_id_grid[j][x, y] = []
        self.targetextractor.rep_id_grid[j][x, y].append(len(self.targetextractor.rep_list[j])-1)                  
            
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
    
    def rem_img(self, label, j):
        filename = self.fileprefix+label+str(j)+'.png'
        self.del_data_file(filename)
        shutil.copyfile(self.fileprefix+"_blank.png", filename)
        
    def del_data_file(self, filename):
        try:
            os.remove(filename)
        except OSError:
            logging.info("no file to delete: " + filename)
            return 
    
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