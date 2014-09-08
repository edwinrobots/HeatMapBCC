'''
Created on 5 Sep 2014

@author: edwin
'''
from provstore.api import Api
import json, logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    api = Api(username='atomicorchid', api_key='2ce8131697d4edfcb22e701e78d72f512a94d310')
    datadir = './data'
    
    jsonfile = datadir+"/provfilelist.json"
    with open(jsonfile,'r') as fp:
        provfilelist = json.load(fp)
    
    for i in provfilelist:
        try:
            api.delete_document(i)
        except:
            logging.warning('Could not delete ' + str(i) + ', maybe it no longer exists.')
        
    with open(jsonfile, 'w') as fp:
        json.dump([], fp)    

    logging.info("completed cleanup of old provenance docs.")

