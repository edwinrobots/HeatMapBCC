'''
Created on 23 Jun 2014

@author: edwin
'''
import heatmapbcc, logging, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
import os
import shutil
import pandas as pd

class Heatmap(object):

    combiner = {}

    webdatadir = './web'
    datadir = './data'
    fileprefix = '/mapdata/map_test'
    
    minlat = 26.0
    maxlat = 28.6
    minlon = 82.0
    maxlon = 87.0 
        
    nx = 1000
    ny = 1000
    
    C = []
    K = 1
    rep_ids = []
        
    startclean = True #if true, will delete all previous maps before running
    timestep = 65 #max is likely to be 765
    stepsize = 100 #takes around 4 minutes to run through all updates. There will be 7 updates

    running = False
    
    alpha0 = []
    nu0 = []
    
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
                
        self.alpha0 = np.array([[2.0, 1.2, 1.1, 1.0, 1.0, 1.0], [1.2, 2.0, 1.2, 1.1, 1.0, 1.0], [1.1, 1.2, 2.0, 1.2, 1.1, 1.0], 
                                [1.0, 1.1, 1.2, 2.0, 1.2, 1.1], [1.0, 1.0, 1.1, 1.2, 2.0, 1.2], [1.0, 1.0, 1.0, 1.1, 1.2, 2.0]])
        self.nu0 = np.array([1,1,1,1,1,1], dtype=float)#np.array([0.5, 0.5])#0.03
        self.rep_ids.append(0)
        if self.startclean:
            self.initial_cleanup()
        self.load_kll_data()
        #self.load_ush_data() 
        self.write_coords_json()     
        
        self.run_script_only = run_script_only   
        
    def runBCC(self, j):
        C = self.C[j]
        self.runBCC_subset(C)
        
    def runBCC_up_to_time(self, j, timestep):       
        C = self.C[j]
        logging.info("!!! At this point we need to fetch the latest data from Zooniverse !!!")
        bcc_pred, _ = self.runBCC_subset(C)
        return bcc_pred
            
    def runBCC_subset(self, C, j=1):
        if j not in self.combiner or self.combiner[j]==None or self.combiner[j].K<self.K:
            self.combiner[j] = heatmapbcc.HeatMapBCC(self.nx, self.ny, len(self.nu0), self.alpha0.shape[1], self.alpha0, self.nu0, self.K)
            self.combiner[j].min_iterations = 5
            self.combiner[j].max_iterations = 200
            self.combiner[j].conv_threshold = 0.1
            self.combiner[j].uselowerbound = False

        bcc_pred = self.combiner[j].combine_classifications(C)
        bcc_pred = bcc_pred.reshape((len(self.nu0), self.nx,self.ny))
        cs = np.cumsum(bcc_pred, axis=0)
        nclasses = len(self.nu0)
#         result = bcc_pred * np.arange(len(self.nu0))[:, np.newaxis, np.newaxis]
#         result = np.sum(result, axis=0)
        result = np.zeros((self.nx, self.ny))
        for c in np.fliplr(range(nclasses)):
            result[cs[c,:,:]>0.9] = c # remove anything that is still highly uncertain
        result = result / (len(self.nu0)-1)
        
        return result, self.combiner[j]

    #@profile
    def loop_iteration(self, j, t):
        bcc_pred = self.runBCC_up_to_time(j,t)
        return bcc_pred
              
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
                    
    def timed_update_loop(self, j=1):
        logging.info("Run BCC at intervals, loading new reports.")
        
        #Intially, we are running. If this flag is changed by another thread, we'll stop.
        self.running = True 

        #Call this in a new thread that can be updated by POST to the web server. 
        #When a new report is received by POST to web server, the server can kill this thread, call insert_trusted and then restart this method in a new thread
        while self.running:
            logging.info("timed_update_loop timestep " + str(self.timestep))
            
            # BCC PREDICTIONS
            bcc_pred = self.loop_iteration(j, self.timestep)
            self.plotresults(bcc_pred, label='Predicted Incidents of type '+str(j))
            self.write_img("", j)
            
            # BCC UNCERTAINTY
            bcc_stdPred = self.combiner[j].get_sd_kappa(0)
            for c in range(1, len(self.nu0)):
                bcc_stdPred += self.combiner[c].get_sd_kappa(c)
            maxunc = np.max(bcc_stdPred) #normalise it
            minunc = np.min(bcc_stdPred)
            bcc_stdPred = (bcc_stdPred-minunc)/(maxunc-minunc)
            lab = 'Uncertainty (S.D.) in Pr(incident) of type '+str(j)
            self.plotresults(bcc_stdPred, label=lab)
            self.write_img("_sd_",j)
            
            #once complete, update the current time step
            logging.info("Change this as we won't be using a fixed step size with live data.")
            self.timestep += self.stepsize
            
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
    
    def plotresults(self, bcc_pred, label='no idea', interp='none', imgmax=1, imgmin=0):
        dpi = 96
        if self.nx>=500:
            fig = plt.figure(frameon=False, figsize=(float(self.nx)/dpi,float(self.ny)/dpi))
        else:
            fig = plt.figure(frameon=False)
        plt.autoscale(tight=True)
        #Can also try interpolation=nearest or none
        ax = fig.add_subplot(111)
        ax.set_axis_off()    
                
        cmap = plt.get_cmap('jet')                
        cmap._init()
        
        prior = self.nu0/float(np.sum(self.nu0))
        prior = prior[0]
        
        alphas1 = np.linspace(0, 0.3, np.ceil(0.2*cmap.N)+3)
        alphas2 = np.linspace(0.3, 0.6, 0.8*cmap.N)
        alphas = np.concatenate((alphas1, alphas2))
        cmap._lut[:,-1] = alphas        
        
        # bin the results so we get contours rather than blurred map
        contours = bcc_pred.copy()
        contours[bcc_pred<0.2] = 0
        contours[(bcc_pred<0.4) & (bcc_pred>=0.2)] = 0.25
        contours[(bcc_pred<0.6) & (bcc_pred>=0.4)] = 0.6
        contours[(bcc_pred<0.8) & (bcc_pred>=0.6)] = 0.8
        contours[bcc_pred>=0.8] = 1
         
        plt.imshow(contours, cmap=cmap, aspect=None, origin='lower', \
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
        
    def load_kll_data(self):
        dataFile = self.datadir + '/1430580842.csv'#/1430513460.csv'#'/nepal_1_5_2151_kathmandhulivinglabs.csv'
        alldata = pd.read_csv(dataFile, parse_dates=False, index_col=False, usecols=[5,6,7], skipinitialspace=True, quotechar='"')
        latdata = alldata['LATITUDE']
        londata = alldata['LONGITUDE']#pd.read_csv(dataFile, quotechar='"', skipinitialspace=True, dtype=np.float64, sep=',', usecols=[7])
        catdata = alldata['CATEGORY']
        latdata,londata = self.translate_points_to_local(latdata,londata)
        
        # output a four column list.
        # column 0 is category/agent/report source
        # column 1 is location x
        # column 2 is location y
        # column 3 is  value, 1 for positive report of damage/danger/emergency etc. 0 for all clear.
        C = {}
        C[1] = np.zeros((0, 4))
        
        categories = np.array([], dtype=np.str)
        
        nreports = 0
        for i, cat in enumerate(catdata):
            lat = latdata[i]
            lon = londata[i]
            
            if lat>=self.nx or lat<0 or lon>=self.ny or lon<0:
                logging.warning("Coords outside area of interest: %f %f" % (lat,lon))
                continue
            
            toks = str.split(cat.strip(), ',')
            for tok in toks:
                if not len(tok):
                    continue
                if not tok in categories:
                    categories = np.append(categories, tok)
                
                catidx = np.argwhere(categories==tok)[0][0]
            
                crow = np.array([catidx, lat, lon, 5]).reshape((1,4))
                C[1] = np.concatenate((C[1], crow), axis=0)
                nreports += 1
            
        C[1] = C[1][0:nreports, :]
        self.K = len(categories)
        self.C = C
        self.combiner = {} #reset as we have reloaded the data
        
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
    logging.basicConfig(level=logging.INFO)   
    heatmap = Heatmap(1000, 1000, run_script_only=True)
    heatmap.timed_update_loop(1)