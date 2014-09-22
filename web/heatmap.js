var mapReloadTime = 5000
var heatmapLayer = [];
var map = [];
var layerName = [];
var timerRunning = 0;

var coords = [];
var bounds = [];

//Please set this in the page that is loading the content
var mapdatastr = [];//"mapdata/map_test";
var maptypestr = "_rep_intensity_";

var overlayOptions = [];

var markers = {};
var openWindows = {}

var layerId = 1;

var repList = {};
var piList = {};

function loadOverlayImage(restartTimer, layerName, plotTargetsFlag){    
    var peaksx = [];
    var peaksy = [];
    var peaksval = [];    
    
//     if (layerName=="emergency"){
//         layerId = 1;
//     }else if(layerName=="vital"){
//         layerId = 2;
//     }else if(layerName=="health"){
//         layerId = 3;
//     }else if(layerName=="security"){
//         layerId = 4;
//     }else if(layerName=="infra"){
//         layerId = 5;
//     }else if(layerName=="natural"){
//         layerId = 6;
//     }else if(layerName=="services"){
//         layerId = 7;
//     }else if(layerName=="people"){
//         layerId = 8;
//     }
    layerId = 1; //fix to emergencies for now

    filename = mapdatastr+maptypestr+layerId+".png?" + new Date().getTime();
    
    $("span[class='filename_disp']").text(filename)
    
    //alert("About to clear heatmap");
    if (typeof heatmapLayer.map==='object' && heatmapLayer.getMap()!=null){
        heatmapLayer.setMap(null);
        //alert("cleared that shit");
    }
    //alert("About to set new heatmap: " + filename);
    try{           
        $.getJSON("mapdata/coords.json",  function(data){
            coords = data;
            bounds = new google.maps.LatLngBounds(
                new google.maps.LatLng(coords[0],coords[2]),
                new google.maps.LatLng(coords[1],coords[3]));
                
            heatmapLayer = new google.maps.GroundOverlay(filename, bounds, overlayOptions); 
            
            if (!restartTimer){
                return
            }
    
            setTimeout(function(){
                layerName = getLayerName();
                setHeatMapLayer(true, plotTargetsFlag);
            }, mapReloadTime);
        });
    } catch (e) {
        alert("Loading map failed. " + e.msg);
    }
}

function addReportInfo(infoBox, reps, pis){
    for (var i=0; i<reps.length && i<10; i++){
        trust = pis[i][1][1]/(pis[i][0][1]+pis[i][1][1]) ;
        pi_display = '<div style="width:56px;height:25px"><div style="float:left;border:1px solid #304888; width:25px;height:25px"><div style="float:left; width:25px; height:' + trust*25 + 'px; background-color:green"></div></div><div style="float:right;width:25px;height:25px">'+Math.round(trust*100)/100+'</div></div>';
        infoBox.content = infoBox.content + "<tr><td>" + reps[i] + "</td><td>" + pi_display + "</td></tr>";   
    }
}

function drawTargetMarker(tid, locx, locy, img){
        //create a google maps marker
    
        if (typeof markers[tid] !== 'undefined'){
            markers[tid].setMap(null);
        }
    
        markers[tid] = new google.maps.Marker({
            position: new google.maps.LatLng(locx, locy),
            map: overlayOptions.map,
            title: 'Target',
            icon: img
        }); 
        
        google.maps.event.addListener(markers[tid], 'click', function() {
                        
            var infoWindow = new google.maps.InfoWindow({
                content: "<table border='1'><tr><td><b>Target ID: " + tid + "</b></td><td>Trust</td></tr>",
                position: new google.maps.LatLng(locx,locy)
            });
            addReportInfo(infoWindow, repList[tid], piList[tid])       
            infoWindow.content = infoWindow.content + "</table>"
            
            infoWindow.open(map);
            for (var i in openWindows){
                    
                openWindows[i].close();
                delete openWindows[i];
            }
            openWindows[tid] = infoWindow;
        });
}

function plotTargets(targetData){
    removeReports();
    for (var i=0; i<targetData.length; i++){
        target = targetData[i];
        
        //extract the location
        tid = target[0];
        locx = target[1];
        locy = target[2];
        typeid = target[3];
        repList[tid] = target[5]; //strongly associated reports
        piList[tid] = target[6];
        
        img = "images/question.png";//"http://maps.google.com/mapfiles/kml/shapes/caution.png";
        
        drawTargetMarker(tid, locx, locy, img);
    }        
}

function setHeatMapLayer(restartTimer, plotTargetsFlag) {
    //val = $("input[name='layerSelect']:checked").attr("id")
    //layerName = val;
    if (!plotTargetsFlag){
        loadOverlayImage(restartTimer, layerName);
    }else{
        //load the targets
        $.getJSON("targets_"+layerId+".json?" + new Date().getTime(),  function(data){
            plotTargets(data);
            loadOverlayImage(restartTimer, layerName, true);
        });
    }
}

function removeReports(){
    for (key in markers){
        markers[key].setMap(null);
    }      
}

function switchOverlay(restartTimer){
    maptype = $("input[name='maptype']:checked").attr("id");
    method = $("input[name='method']:checked").attr("id");
    reportsource = $("input[name='reportsource']:checked").attr("id");
        
    plotTargetsFlag = false;
    
    removeReports();
    
    if (method=="reports"){
        maptypestr = "_rep_intensity_";
        methoddisp = "Reports";
    } else if (method=="bcc"){
        maptypestr = ""; //add nothing
        methoddisp = "Bayesian Heatmap";
    } else if (method=="targets"){
        maptypestr = "";
        plotTargetsFlag = true
        methoddisp = "UAV Targets with Heatmap"
    }
    
    if (maptype=="pred"){
        //do nothing
        maptypedisp = " of Emergencies";
    } else if (maptype=="unc"){    
        maptypestr += "_sd_";
        maptypedisp = ": Uncertainty";
    }
    
    if (reportsource=="crowdonly"){
        //do nothing
    } else if (reportsource=="trusted"){
        maptypestr += "_expert_";
    }
    if (restartTimer===undefined){
        setHeatMapLayer(false, plotTargetsFlag);
    }else{
        setHeatMapLayer(restartTimer, plotTargetsFlag);
    }
    $("span[id='statustext']").text(methoddisp + maptypedisp)
}

function getLayerName(){
    return layerName;
}