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

function loadOverlayImage(restartTimer, layerName){    
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
        heatmapLayer = new google.maps.GroundOverlay(filename, bounds, overlayOptions); 
    } catch (e) {
        alert("Loading map failed. " + e.msg);
    }
    
    if (!restartTimer){
        return
    }
    
    setTimeout(function(){
        layerName = getLayerName();
        loadOverlayImage(true, layerName);
    }, mapReloadTime);
}

function addReportInfo(infoBox, reps){
    for (var i=0; i<reps.length && i<10; i++){
        pi_display = "put html string for conf matrix here"
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
                content: "<b>Target ID: " + tid + "</b><table border='1'>"
            });
            addReportInfo(infoWindow, repList[tid])       
            infoWindow.content = infoWindow.content + "</table>"
            
            infoWindow.open(map,markers[tid]);
            for (var i in openWindows){
                    
                openWindows[i].close();
                delete openWindows[i];
            }
            openWindows[tid] = infoWindow;
        });
}

function plotTargets(targetData){
    for (var i=0; i<targetData.length; i++){
        target = targetData[i];
        
        //extract the location
        tid = target[0];
        locx = target[1];
        locy = target[2];
        typeid = target[3];
        repList[tid] = target[4]; //strongly associated reports
        //piList[tid] = target[
        
        img = "http://maps.google.com/mapfiles/kml/shapes/caution.png";
        
        drawTargetMarker(tid, locx, locy, img);
    }        
}

function setHeatMapLayer(restartTimer, plotTargetsFlag) {
    //val = $("input[name='layerSelect']:checked").attr("id")
    //layerName = val;
    $.getJSON("mapdata/coords.json",  function(data){
        coords = data;
        
        bounds = new google.maps.LatLngBounds(
            new google.maps.LatLng(coords[0],coords[2]),
            new google.maps.LatLng(coords[1],coords[3]));
        if (!plotTargetsFlag){
            loadOverlayImage(restartTimer, layerName);
        }else{
            //load the targets
            $.getJSON("targets_"+layerId+".json",  function(data){
                plotTargets(data);
                loadOverlayImage(restartTimer, layerName);
            });
        }
    });
}

function removeReports(){
    for (key in markers){
        markers[key].setMap(null);
    }    
    for (key in openWindows){
        openWindows[key].close();
        openWindows[key] = null;
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
    } else if (method=="bcc"){
        maptypestr = ""; //add nothing
    } else if (method=="targets"){
        maptypestr = "";
        plotTargetsFlag = true
    }
    
    if (maptype=="pred"){
        //do nothing
    } else if (maptype=="unc"){    
        maptypestr += "_sd_";
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
}

function getLayerName(){
    return layerName;
}