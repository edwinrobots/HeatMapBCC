function initialize() {

  mapCentre = new google.maps.LatLng(19, -72.5) //52.953250, -1.187569)

  var mapOptions = {
    zoom: 9,
    center: mapCentre,
    mapTypeId: google.maps.MapTypeId.SATELLITE
  };

  var map = new google.maps.Map(document.getElementById('map-canvas'),
      mapOptions);

  markers = {}
  openWindows = {}
  
  catOption = getQueryVariable('cat');
  //show category annotations or not?
  if (catOption=="showAnn"){
      showAnn = true;
  }else{
      showAnn = false;
  }
  
  for (var id in reportTitles){  
    timedReport(id, map, markers, openWindows, showAnn)
  }
  
//   eventMarkings = {}
//    
//   for (var idx in eventTimes){  
//     timedEvent(idx, map, eventMarkings, openWindows, eventTimes[idx])
//   }
}

function getQueryVariable(variable)
{
       var query = window.location.search.substring(1);
       var vars = query.split("&");
       for (var i=0;i<vars.length;i++) {
               var pair = vars[i].split("=");
               if(pair[0] == variable){return pair[1];}
       }
       return(false);
}

/*Need a function that reads an update from file, then either adds or removes the corresponding marking. 
This requires us to keep a list of markings.*/

function timedReport(id, map, markers, openWindows, showAnn){
    setTimeout(function(){
        drawRep(id, map, new google.maps.LatLng(reportLat[id], reportLon[id]), markers, openWindows, showAnn)
    }, reportTimes[id]*1000)
    
    setTimeout(function(){
        removeEvent(markers[id])
    }, (reportTimes[id]*1000) + 20000)
}

function removeEvent(marking){

//     for (var i=0; i<marking.length; i++){
    marking.setMap(null);
//     }
}

function addReportInfo(infoBox, reportId, showAnn){
    
    type = reportCats[reportId]; // used for category text
    
    if (showAnn){
        cats = "<br/><b>Categories: </b>"
        for (var t=0; t<type.length; t++){
            cats += "<br/>" + type[t]
        }
    
        certainty = Math.random()/2 + 0.5;
        certainty = certainty.toFixed(1);
        certaintyText = "<br/><b>Approx. certainty: </b>" + certainty   
    }else{
        cats = ""
        certaintyText = ""
    }    
    infoBox.content = infoBox.content + "<b>Title: </b>" + reportTitles[reportId] + cats + certaintyText;   
}

function drawRep(id, map, loc, markers, openWindows, showAnn){
    catIdx = reportCatIdx[id]; //used for icon only

    if (showAnn){
        settings = getEventColour(catIdx);
        col = settings[0]
        img = settings[1]
        
        markers[id] = new google.maps.Marker({
            position: loc,
            map: map,
            title: 'Event!',
            icon: img
        });       
    }else{
        markers[id] = new google.maps.Marker({
            position: loc,
            map: map,
            title: 'Event!'
        });
    }    
    
    google.maps.event.addListener(markers[id], 'click', function() {
                    
        var infoWindow = new google.maps.InfoWindow({
            content: ''
        });
        addReportInfo(infoWindow, id, showAnn)         
        
            infoWindow.open(map,markers[id]);
            for (var i in openWindows){
                
                openWindows[i].close();
                delete openWindows[i];
            }
            openWindows[id] = infoWindow;
        });
    return markers[id]
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

function loadScript() {
  var script = document.createElement('script');
  script.type = 'text/javascript';
  script.src = 'http://maps.googleapis.com/maps/api/js?key=AIzaSyAeqPac68BRWyDN4QSQDVU1RnkrIZVuhvc&sensor=false&' +
      'callback=initialize';
  document.body.appendChild(script);
}