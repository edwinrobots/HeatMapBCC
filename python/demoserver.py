'''
Created on 31 Jul 2014

@author: edwin
'''

import SimpleHTTPServer, SocketServer, time, logging, threading, cgi
from ushahidiheatmap import Heatmap
from socket import error as socket_error

logging.basicConfig(level=logging.INFO)    
map_thread = []
global heatmap

class Demoserver(SimpleHTTPServer.SimpleHTTPRequestHandler):
    
    def do_GET(self):
        #this is just default handling
        SimpleHTTPServer.SimpleHTTPRequestHandler.do_GET(self);

    def do_POST(self):
        logging.info("======= Post started =======")
        logging.info(self.headers)
        logging.info("Path = " + self.path)
        
        if "start_heatmap" in self.path:
            logging.info("Starting the timed update loop")
            start_new_map_thread()
            return
        
        if "reset" in self.path:
            kill_map_thread()
            init_heatmap()
            return
        
        logging.info("======= Processing POST values =======")
        # Parse the form data posted
        form = cgi.FieldStorage(
            fp=self.rfile, 
            headers=self.headers,
            environ={'REQUEST_METHOD':'POST',
                     'CONTENT_TYPE':self.headers['Content-Type'],
                     })
        logging.info("Loaded CGI form.")

        typeid = int(form["type"].value)
        x = float(form["lat"].value)
        y = float(form["lon"].value)
        
        logging.info("Report received at " + str(x) + ", " + str(y))
        
        v = int(form["c_val"].value)
        
        global heatmap
        if "trust_acc" in form and "trust_var" in form:
            trust_acc = float(form["trust_acc"].value)
            trust_var = float(form["trust_var"].value)
            if "rep_id" in form:
                rep_id = int(form["rep_id"].value)
                if rep_id==0:
                    rep_id = int(form["reporter"].value)
            else:
                rep_id = -1
            heatmap.insert_trusted(typeid, x, y, v, rep_id, trust_acc, 1.0/trust_var)
        else:
            heatmap.insert_trusted(typeid, x, y, v)
        
        start_new_map_thread()
            
        self.send_response(201, "New report added")
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write("Post completed: new report added") 

def kill_map_thread():
    heatmap.kill_combiners()
    logging.info("Stopped the heatmap generation thread.")
    
def start_new_map_thread():
    logging.info("Starting heatmap updater thread")
    global heatmap
    #heatmap = Heatmap(map_nx,map_ny)
    if heatmap.running == False:
        map_thr = threading.Thread(target=heatmap.timed_update_loop)
        map_thr.start()
        logging.info("Started heatmap updater thread")
    else:
        logging.info("Heatmap was already running so no new thread started.")

def run_web_server():
    logging.debug("Running the web server thread...")
    port = 8000
    httpd = []
    while httpd==[] and port<8100:
        logging.debug("Trying port " + str(port))
        try:
            httpd = SocketServer.TCPServer( ('127.0.0.1', port), Demoserver )
        except socket_error as serr:
            print serr
            port += 1    
    print "Using port " + str(port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('^C received, shutting down server')
        httpd.socket.close()
        heatmap.running = False
    
def web_server_restarter():
    #restart the web server thread if it crashes
    restart = True
    while restart:
        thr_server = threading.Thread(target=run_web_server)
        thr_server.start()
        logging.info("Started the web server thread.")
        while(thr_server.is_alive()):
            restart = False
            time.sleep(10)
        restart = True    
        
def init_heatmap():
    global heatmap
    global map_nx
    global map_ny
    map_nx = 500
    map_ny = 500
    heatmap = Heatmap(map_nx,map_ny,run_script_only=True)
    heatmap.running = False
      
init_heatmap()
thr_server_maintainer = threading.Thread(target=web_server_restarter)
thr_server_maintainer.start()


