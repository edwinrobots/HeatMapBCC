var heatmapLayer = [];
var map = [];
var layerName = [];
var timeoutOverridden = 0;

var coords = [];
var bounds = [];

var mapdatastr = "mapdata/map_big2_nosea";
var maptypestr = "_rep_intensity_";

var overlayOptions = [];

function loadOverlayImage(layerName){    
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
          
    filename = mapdatastr+maptypestr+layerId+".png";
    
    $("span[class='filename_disp']").text(filename)
    
    //alert(filename);
    if (typeof heatmapLayer.map==='object'){
        heatmapLayer.setMap(null);
    }
    try{
        heatmapLayer = new google.maps.GroundOverlay(filename, bounds, overlayOptions); 
    } catch (e) {
        alert("Loading map failed. " + e.msg);
    }
    
//     setTimeout(function(){
//         if (timeoutOverridden>0){
//             timeoutOverridden -= 1;
//             return;
//         }
//         layerName = getLayerName();
//         loadOverlayImage(layerName);
//     }, 10000);
}


function setHeatMapLayer() {
        
    //val = $("input[name='layerSelect']:checked").attr("id")
    //layerName = val;
    timeoutOverridden += 1;
    
    $.getJSON("mapdata/coords.json",  function(data){
        coords = data;
        
        bounds = new google.maps.LatLngBounds(
            new google.maps.LatLng(coords[0],coords[2]),
            new google.maps.LatLng(coords[1],coords[3]));
                
        loadOverlayImage(layerName);
    });
}

function switchOverlay(){
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
        
    setHeatMapLayer();
}

function getLayerName(){
    return layerName;
}

function getEventColour(cat){
    if (cat=="1."){
        col = '#FF0000'
        img = "http://maps.google.com/mapfiles/kml/shapes/caution.png"
    }
    else if (cat=="1a."){
        col = '#FF0000'
        img = "http://maps.google.com/mapfiles/kml/shapes/earthquake.png"
    }    
    else if (cat=="1c."){
        col = '#FF0000'
        img = "http://maps.google.com/mapfiles/kml/shapes/falling_rocks.png"
    }
    else if (cat=="1b."){
        col = '#FF0000'
        img = "http://maps.google.com/mapfiles/kml/shapes/hospitals.png"
    }
    else if (cat=="1d."){
        col = '#FF0000'
        img = "http://maps.google.com/mapfiles/kml/shapes/firedept.png"
    }
    else if (cat=="2."){
        col = '#00FF00'
        img = "http://maps.google.com/mapfiles/kml/shapes/convenience.png"
    }
    else if (cat=="2a."){
        col = '#00FF00'
        img = "http://maps.google.com/mapfiles/kml/shapes/convenience.png"
    }
    else if (cat=="2b."){
        col = '#00FF00'
        img = "http://maps.google.com/mapfiles/kml/shapes/convenience.png"
    }
    else if (cat=="2c."){
        col = '#00FF00'
        img = "http://maps.google.com/mapfiles/kml/shapes/convenience.png"
    }
    else if (cat=="2d."){
        col = '#00FF00'
        img = "http://maps.google.com/mapfiles/kml/shapes/homegardenbusiness.png"
    }
    else if (cat=="2e."){
        col = '#00FF00'
        img = "http://maps.google.com/mapfiles/kml/shapes/gas_stations.png"
    }
    else if (cat=="2f."){
        col = '#00FF00'
        img = "http://maps.google.com/mapfiles/kml/shapes/mechanic.png"
    }
    else if (cat=="3."){
        col = '#008080'
        img = "http://maps.google.com/mapfiles/kml/pal3/icon38.png"
    }
    else if (cat=="3a."){
        col = '#008080'
        img = "http://maps.google.com/mapfiles/kml/pal3/icon39.png"
    }
    else if (cat=="3b."){
        col = '#008080'
        img = "http://maps.google.com/mapfiles/kml/pal3/icon51.png"
    }
    else if (cat=="3d."){
        col = '#008080'
        img = "http://maps.google.com/mapfiles/kml/pal3/icon51.png"
    }
    else if (cat=="4."){
        col = '#F00080'
        img = "http://maps.google.com/mapfiles/kml/shapes/police.png"
    }
    else if (cat=="4a."){
        col = '#F00080'
        img = "http://maps.google.com/mapfiles/kml/shapes/police.png"
    }
    else if (cat=="5."){
        col = '#808000'
        img = "http://maps.google.com/mapfiles/kml/pal3/icon21.png"
    }
    else if (cat=="5a."){
        col = '#808000'
        img = "http://maps.google.com/mapfiles/kml/pal3/icon21.png"
    }
    else if (cat=="5b."){
        col = '#808000'
        img = "http://maps.google.com/mapfiles/kml/pal3/icon21.png"
    }
    else if (cat=="5c."){
        col = '#808000'
        img = "http://maps.google.com/mapfiles/kml/pal3/icon21.png"
    }
    else if (cat=="6."){
        col = '#8000F0'
        img = "http://maps.google.com/mapfiles/kml/shapes/water.png"
    }
    else if (cat=="6a."){
        col = '#8000F0'
        img = "http://maps.google.com/mapfiles/kml/shapes/water.png"
    }
    else if (cat=="6c."){
       col = '#8000F0'
       img = "http://maps.google.com/mapfiles/kml/shapes/water.png"
    }
    else if (cat=="6b."){
       col = '#8000F0'
       img = "http://maps.google.com/mapfiles/kml/shapes/man.png"        
    }
    else if (cat=="7."){
        col = '#0000FF'
        img = "http://maps.google.com/mapfiles/kml/pal4/icon6.png"
    }
    else if (cat=="7a."){
        col = '#0000FF'
        img = "http://maps.google.com/mapfiles/kml/pal4/icon4.png"
    }
    else if (cat=="7b."){
        col = '#0000FF'
        img = "http://maps.google.com/mapfiles/kml/pal4/icon3.png"
    }
    else if (cat=="7c."){
        col = '#0000FF'
        img = "http://maps.google.com/mapfiles/kml/pal4/icon5.png"
    }
    else if (cat=="7d."){
        col = '#0000FF'
        img = "http://maps.google.com/mapfiles/kml/pal5/icon11.png"
    }
    else if (cat=="8."){
        col = '#505050'
        img = "http://maps.google.com/mapfiles/kml/pal3/icon49.png"
    }
    else if (cat=="8a."){
        col = '#505050'
        img = "http://maps.google.com/mapfiles/kml/pal3/icon48.png"
    }
    else if (cat=="8d."){
        col = '#505050'
        img = "http://maps.google.com/mapfiles/kml/pal2/icon6.png"
    }
    else if (cat=="8e."){
        col = '#505050'
        img = "http://maps.google.com/mapfiles/kml/pal2/icon6.png"
    }
    else {
//         alert("Unknown category: " + cat);
        col = '#000000'
        img = "http://maps.google.com/mapfiles/kml/pal3/icon49.png"
    }
    
    return [col,img];
}