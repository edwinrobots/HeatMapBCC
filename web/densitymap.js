var mapReloadTime = 5000
var heatmapLayer = [];
var map = [];
var layerName = [];
var timerRunning = 0;

var coords = [84.0344, 27.1709, 86.0148, 28.5954]//[84.551, 27.162377777777774, 85.75159, 28.30438];
var bounds = [];

//Please set this in the page that is loading the content
var mapdatastr = "images/transparency"//"mapdata/popdensity";
var overlayOptions = [];

function loadOverlayImage(){
    filename = mapdatastr+".png?" + new Date().getTime();
    
	bounds = new google.maps.LatLngBounds(
                new google.maps.LatLng(coords[1],coords[0]),
                new google.maps.LatLng(coords[3],coords[2]));
    
    heatmapLayer = new google.maps.GroundOverlay(filename, bounds, overlayOptions);
}