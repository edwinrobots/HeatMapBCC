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
        logging.info(self.path)
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
    heatmap = Heatmap(1000,1000, compose_demo_reports=True)
    map_thr = threading.Thread(target=heatmap.timed_update_loop)
    map_thr.start()
    logging.info("Started heatmap updater thread")

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
    
thr_server_maintainer = threading.Thread(target=web_server_restarter)
thr_server_maintainer.start()
global heatmap
heatmap = Heatmap(1000,1000, compose_demo_reports=True)
heatmap.timed_update_loop()    
