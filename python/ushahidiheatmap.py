'''
Created on 23 Jun 2014

@author: edwin
'''
import heatmapbcc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import coo_matrix
import time
sea_map = []

webdatadir = '../web'
datadir = '../data'

def runBCC(C,nx,ny,nReporters):
    nu0 = np.array([0.5, 0.5])#0.03
    
    alpha0 = np.array([[30.0, 1.0], [1.0, 30.0]])
    if nReporters==2:
        alpha0new = np.zeros((2,2,2))
        alpha0new[:,:,0] = alpha0#[[150, 100], [100, 150]]
        alpha0 = alpha0new
        alpha0[:,:,1] = [[1000,1], [1, 1000]]      
    combiner = heatmapbcc.Heatmapbcc(nx, ny, 2, 2, alpha0, nu0, nReporters)
    combiner.minNoIts = 5
    combiner.maxNoIts = 200
    combiner.convThreshold = 0.1
    bcc_pred = combiner.combineClassifications(C)
    #bcc_pred = np.exp(combiner.lnKappa)
    bcc_pred = bcc_pred[1,:,:].reshape((nx,ny))
    return bcc_pred,combiner

def reportIntensity(C, nx, ny):
    data = np.array(C[:,3]).reshape(-1)
    rows = np.array(C[:,1]).reshape(-1)
    cols = np.array(C[:,2]).reshape(-1)
    
    nu0 = np.array([0.05, 0.03])
    
    repgrid = coo_matrix((data,(rows,cols)), shape=(nx,ny))
    
    negdata = np.array(C[:,3]).reshape(-1)
    negdata[np.where(data==0)] = -1
    negdata[np.where(data>0)] = 0    
    
    negrepgrid = coo_matrix((negdata,(rows,cols)), shape=(nx,ny))
    
    nuneg = np.zeros((nx,ny)) - negrepgrid.todense() + nu0[0]
    nupos = np.zeros((nx,ny)) + repgrid.todense() + nu0[1]

    pred = np.divide(nupos, nupos+nuneg)    
    variance = np.divide(np.multiply(nupos,nuneg), np.multiply(np.square(nupos+nuneg), nupos+nuneg+1))
    std = np.sqrt(variance)    
    return pred, std

def removesea(sea_map, bcc_pred, nx, ny, maxlat, minlat, maxlon, minlon):
    
    if sea_map != []:
        bcc_pred[np.where(sea_map==1)] = 0
#         for i in range(nx):
#             print i
#             for j in range(ny):
#                 print '   ' + str(j)
#                 if sea_map[i,j]==1:
#                     bcc_pred[sea_map] = 0
        return bcc_pred, sea_map
    
    from point_in_polygon import cn_PnPoly
    poly1 = np.genfromtxt(datadir+"/haiti_polygon_1.csv", usecols=[0,1], delimiter=',')
    poly2 = np.genfromtxt(datadir+"/haiti_polygon_2.csv", usecols=[0,1], delimiter=',')

    #translate
    poly1x, poly1y = translate_points_to_local(poly1[:,1],poly1[:,0],nx,ny)
    poly2x, poly2y = translate_points_to_local(poly2[:,1], poly2[:,0], nx,ny)
    poly1 = np.concatenate((poly1x.reshape((len(poly1x),1)),poly1y.reshape((len(poly1y),1))), axis=1)
    poly2 = np.concatenate((poly2x.reshape((len(poly2x),1)),poly2y.reshape((len(poly2y),1))), axis=1)    
    
    extra_border = 0.02
    latborderfrac = extra_border/(maxlat-minlat)
    lonborderfrac = extra_border/(maxlon-minlon)
    xborder = np.ceil(latborderfrac*nx)
    yborder = np.ceil(lonborderfrac*ny)
    
    # points after which we don't care if it is in Haiti or not 
    #--> this goes over border into Dominican Rep.
    blehx, _ = translate_points_to_local(18.2, -72, nx, ny) 
    blehx2, blehy = translate_points_to_local(19.8, -72, nx, ny) 
        
    sea_map = np.zeros((nx,ny), dtype=np.int8)
    
    print "Sea map loading..."
    for i in range(nx):
        print str(i)
        for j in range(ny):       
            if i>blehx and i<blehx2 and j>=blehy:
                continue               
            if not cn_PnPoly([i-xborder,j-yborder], poly1) and not cn_PnPoly([i-xborder,j-yborder], poly2) \
            and not cn_PnPoly([i+xborder,j-yborder], poly1) and not cn_PnPoly([i+xborder,j-yborder], poly2) \
            and not cn_PnPoly([i-xborder,j+yborder], poly1) and not cn_PnPoly([i-xborder,j+yborder], poly2) \
            and not cn_PnPoly([i+xborder,j+yborder], poly1) and not cn_PnPoly([i+xborder,j+yborder], poly2):
                bcc_pred[i,j] = 0
                sea_map[i,j] = 1
    return bcc_pred, sea_map

def plotResults(nx, ny, bcc_pred, sea_map, label='no idea', interp='none'):
            
    bcc_pred,sea_map = removesea(sea_map, bcc_pred,nx,ny,maxlat,minlat,maxlon,minlon)
            
    dpi = 96
    if nx>=500:
        fig = plt.figure(frameon=False, figsize=(nx/dpi,ny/dpi))
    else:
        fig = plt.figure(frameon=False)
    plt.autoscale(tight=True)
    #fig.se#
    #gridx = np.tile( range(1,nx+1),(ny,1))
    #gridy = np.tile(range(1,ny+1), (nx,1)).transpose()
    
#     ax = fig.add_subplot(1, 1, 1, projection='3d')    
#     ax.plot_surface(gridx, gridy, bcc_pred, cstride=1, rstride=1, \
#                     cmap=plt.get_cmap('spectral'), linewidth=0)
#     ax.view_init(elev=90, azim=-90)

    #Can also try interpolation=nearest or none
    ax = fig.add_subplot(111)
    ax.set_axis_off()    
    imgmax = 1
    imgmin = 0
    plt.imshow(bcc_pred, cmap=plt.get_cmap('jet'), aspect=None, origin='lower', \
               vmin=imgmin, vmax=imgmax, interpolation=interp, filterrad=0.01)

                           
    fig.tight_layout(pad=0,w_pad=0,h_pad=0)
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())    
    return sea_map
    
def translate_points_to_local(latdata,londata,nx,ny):
    print 'Translating original coords to local values.'
        
    latdata = np.float64(latdata)
    londata = np.float64(londata)
    
#     maxlat = np.max(latdata)
#     minlat = np.min(latdata)
    normlatdata = np.divide(latdata-minlat, maxlat-minlat)
    
#     maxlon = np.max(londata)
#     minlon = np.min(londata)
    normlondata = np.divide(londata-minlon, maxlon-minlon)    
        
    latdata = np.round(np.multiply(normlatdata, nx))
    londata = np.round(np.multiply(normlondata, ny))
        
    return latdata,londata    
    
def tranlate_points_to_original(x,y,nx,ny,maxlat,minlat,maxlon,minlon):
    print 'Tranlating our local points back to original lat/long'
    
    #normalise
    x = np.divide(np.float64(x),nx)
    y = np.divide(np.float64(y),ny)
    
    latdata = np.multiply(x, maxlat-minlat) + minlat
    londata = np.multiply(y, maxlon-minlon) + minlon
    
    return latdata,londata
    
def loadUshData(nx,ny):
    dataFile = datadir+'/exported_ushahidi.csv'
    
    nreporters = 1
    
    #load the data
#     reportIDdata = np.genfromtxt(dataFile, np.str, delimieter=',', skip_header=True, usecols=[])
#     datetimedata = np.genfromtxt(dataFile, np.datetime64, delimiter=',', skip_header=True, usecols=[2,3])
    latdata = np.genfromtxt(dataFile, np.float64, delimiter=',', skip_header=True, usecols=[4])
    londata = np.genfromtxt(dataFile, np.float64, delimiter=',', skip_header=True, usecols=[5])
    reptypedata = np.genfromtxt(dataFile, np.str, delimiter=',', skip_header=True, usecols=[1])
        
    latdata,londata = translate_points_to_local(latdata,londata,nx,ny)
         
    C = {}            
    for i, reptypetext in enumerate(reptypedata):        
        typetoks = reptypetext.split('.')
        typetoks = typetoks[0].split(',')
        for typestring in typetoks:
            maintype = typestring[0] #first character should be a number
            try:
                typeID = int(maintype)
                #print "Type ID found: " + str(typeID)
            except ValueError:
                print 'Not a report category: ' + typestring
                continue
            repx = latdata[i]
            repy = londata[i]
            
            if repx>=nx or repx<0 or repy>=ny or repy<0:
                continue
            
            try:
                Crow = np.array([0, repx, repy, 1]) # all report values are 1 since we only have confirmations of an incident, not confirmations of nothing happening
            except ValueError:
                print 'ValueError creating a row of the crowdsourced data matrix.!'        
            if C=={} or typeID not in C:
                C[typeID] = Crow.reshape((1,4))
            else:
                C[typeID] = np.concatenate((C[typeID], Crow.reshape(1,4)), axis=0)      

    return nreporters, C   

def writeImg(label,j):
    plt.savefig(fileprefix+label+str(j)+'.png', bbox_inches='tight', \
                pad_inches=0, transparent=True, dpi=96)

def writeToJson(bcc_pred, nx, ny, j, label=""):
    import json
    jsonFile = fileprefix + label + str(j) + '.json'
    bcc_pred = bcc_pred
    bcc_pred = bcc_pred.tolist()
    with open(jsonFile, 'w') as fp:
        json.dump(bcc_pred, fp)
     
def writeCoordsToJson(minlat,maxlat,minlon,maxlon):
    import json
    jsonFile = webdatadir+'/mapdata/coords.json'
    with open(jsonFile, 'w') as fp:
        json.dump([minlat,maxlat,minlon,maxlon], fp)    
        
def insertTrusted(nx,ny,C):
    x,y = translate_points_to_local(18.52,-72.284,nx,ny)
    Crow = np.array([[1, x, y, 0],[1, x+1, y, 0],[1, x+2, y, 0],[1, x+3, y, 0], \
                    [1, x+1, y+1, 0],[1, x+1, y-1, 0],[1, x+2, y+1, 0],[1, x+3, y-1, 0], \
                    [1, x-1, y+1, 0],[1, x-1, y-1, 0],[1, x+2, y-1, 0],[1, x+3, y+1, 0]]) 
    C[1] = np.concatenate((C[1], Crow), axis=0)
    return C 
        
#--------  MAIN --------------------------------------------------
if __name__ == '__main__':
   
    fileprefix = webdatadir+'/mapdata/map_big_nosea_speed'
   
    sea_map = []
   
    minlat = 18.0
    maxlat = 20.0
    minlon = -73.8
    maxlon = -71.7   
    writeCoordsToJson(minlat,maxlat,minlon,maxlon)
#     
#     #High definition with no interpolation
#     nx = 2000
#     ny = 2000  
#     _, C = loadUshData(nx,ny)     
#     for j in range(1,2):
#         rep_pred, rep_std = reportIntensity(C[j], nx, ny)
#         sea_map = plotResults(nx, ny, rep_pred, sea_map, label='Predicted Incidents of type '+str(j))
#         #writeToJson(bcc_pred, nx,ny,j,label="_rep_intensity_")
#         writeImg("_rep_intensity_", j)
#           
#         plotResults(nx, ny, rep_std, sea_map, label='Uncertainty (S.D.) in Pr(incident) of type '+str(j))
#         #writeToJson(stdPred, nx,ny,j, label="_sd_")
#         writeImg("_rep_intensity__sd_",j)   
    
    #Using BCC - lower res so it is tractable to interpolate
    nx = 1000
    ny = 1000
    _, C = loadUshData(nx,ny)       
    for j in range(1,2):
        start = time.clock()
        
        bcc_pred,combiner = runBCC(C[j],nx,ny,1)
        
        fin = time.clock()
        print "bcc heatmap prediction timer (no loops): " + str(fin-start)
                        
#         bcc_mpr = combiner.getmean()
#         sea_map = plotResults(nx, ny, bcc_pred, sea_map, label='Predicted Incidents of type '+str(j))
#         #writeToJson(bcc_pred, nx,ny,j)
#         writeImg("", j)
#                 
#         plotResults(nx, ny, bcc_mpr, sea_map, label='Incident Rate of type '+str(j))
#         #writeToJson(combiner.mPr, nx,ny,j, label="_mpr_")
#         writeImg("_mpr_",j)
#         
#         bcc_stdPred = combiner.getsd()
#         plotResults(nx, ny, bcc_stdPred, sea_map, label='Uncertainty (S.D.) in Pr(incident) of type '+str(j))
#         #writeToJson(stdPred, nx,ny,j, label="_sd_")
#         writeImg("_sd_",j)
     
#     #insert a trusted report at 18.5333 N, -72.3333 W     
#     nx = 2000
#     ny = 2000
#     _, C = loadUshData(nx,ny)
#     C = insertTrusted(nx,ny,C)
#     for j in range(1,2):
#         rep_pred, rep_std = reportIntensity(C[j], nx, ny)
#         sea_map = plotResults(nx, ny, rep_pred, sea_map, label='Predicted Incidents of type '+str(j))
#         #writeToJson(bcc_pred, nx,ny,j,label="_rep_intensity_")
#         writeImg("_rep_intensity__expert_", j)
#           
#         plotResults(nx, ny, rep_std, sea_map, label='Uncertainty (S.D.) in Pr(incident) of type '+str(j))
#         #writeToJson(stdPred, nx,ny,j, label="_sd_")
#         writeImg("_rep_intensity__sd__expert_",j)      
#          
#     nx = 2000
#     ny = 2000  
#     _, C = loadUshData(nx,ny) 
#     C = insertTrusted(nx,ny,C)
#     for j in range(1,2):
#         bcc_pred2,combiner2 = runBCC(C[j],nx,ny,2)
#         sea_map = plotResults(nx, ny, bcc_pred2, sea_map, label='Predicted Incidents of type '+str(j))
# #         writeToJson(bcc_pred, nx,ny,j, label="_expert_")
#         writeImg("_expert_",j)        
#  
#         bcc_mpr2 = combiner2.getmean()
#         plotResults(nx, ny, bcc_mpr2, sea_map, label='Incident Rate of type '+str(j))
# #         writeToJson(combiner.mPr, nx,ny,j, label="_mpr_expert_")
#         writeImg("_mpr_expert_",j)
#          
#         bcc_stdPred2 = combiner2.getsd()

#         plotResults(nx, ny, bcc_stdPred2, sea_map, label='Uncertainty (S.D.) in Pr(incident) of type '+str(j))
# #         writeToJson(stdPred, nx,ny,j, label="_sd_expert_")        
#         writeImg("_sd__expert_",j)        