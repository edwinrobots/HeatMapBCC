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

function setHeatMapLayer(restartTimer) {
        
    //val = $("input[name='layerSelect']:checked").attr("id")
    //layerName = val;
    
    $.getJSON("mapdata/coords.json",  function(data){
        coords = data;
        
        bounds = new google.maps.LatLngBounds(
            new google.maps.LatLng(coords[0],coords[2]),
            new google.maps.LatLng(coords[1],coords[3]));
                
        loadOverlayImage(restartTimer, layerName);
    });
}

function switchOverlay(restartTimer){
    maptype = $("input[name='maptype']:checked").attr("id");
    method = $("input[name='method']:checked").attr("id");
    reportsource = $("input[name='reportsource']:checked").attr("id");
        
    if (method=="reports"){
        maptypestr = "_rep_intensity_";
    } else if (method=="bcc"){
        maptypestr = ""; //add nothing
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
        setHeatMapLayer(false);
    }else{
        setHeatMapLayer(restartTimer);
    }
}

function getLayerName(){
    return layerName;
}