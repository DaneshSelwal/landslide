// =====================================================
// 1. RENAME IMPORTED ASSETS
// =====================================================

// -------- Raster layers --------
var aspect     = image;     
var dem        = image2;    
var lulc       = image4;    
var ndvi       = image5;    
var rainfall   = image6;    
var slope      = image7;    

// ---- NEW distance-based layers ----
var dfriver    = image8;    // DFRIVER_CR
var dfroad     = image9;    // DFROAD_CR
var dflineament= image10;   // LINEAMENT_CR

// -------- Vector layers --------
var lithology = table;    
var boundary  = table2;   
var ls_points = table3;   


// =====================================================
// 2. STUDY AREA
// =====================================================

var studyArea = boundary.geometry();
Map.centerObject(studyArea, 9);
Map.addLayer(studyArea, {color:'black'}, 'Study Area');


// =====================================================
// 3. CLIP ALL RASTERS
// =====================================================

aspect      = aspect.clip(studyArea);
dem         = dem.clip(studyArea);
slope       = slope.clip(studyArea);
ndvi        = ndvi.clip(studyArea);
lulc        = lulc.clip(studyArea);
rainfall    = rainfall.clip(studyArea);
dfriver     = dfriver.clip(studyArea);
dfroad      = dfroad.clip(studyArea);
dflineament = dflineament.clip(studyArea);


// =====================================================
// 4. BASE VISUALISATION (OLD + NEW)
// =====================================================

// ---- Existing layers ----
Map.addLayer(dem, {min:500,max:6000,
  palette:['blue','green','yellow','brown','white']}, 'DEM');

Map.addLayer(slope, {min:0,max:60,
  palette:['white','yellow','orange','red']}, 'Slope');

Map.addLayer(aspect, {min:0,max:360,
  palette:['blue','cyan','green','yellow','orange','red']}, 'Aspect');

Map.addLayer(ndvi, {min:0,max:255,
  palette:['brown','yellow','green']}, 'NDVI');

Map.addLayer(lulc, {min:1,max:10,
  palette:['red','green','blue','yellow','cyan','magenta','brown','orange']}, 'LULC');

Map.addLayer(rainfall, {min:0,max:3000,
  palette:['white','blue','purple']}, 'Rainfall');

Map.addLayer(ls_points, {color:'black'}, 'Landslide Points');

// ---- NEW distance layers ----
Map.addLayer(dfriver, {
  min: 0, max: 2000,
  palette: ['red','yellow','green']
}, 'Distance from River');

Map.addLayer(dfroad, {
  min: 0, max: 2000,
  palette: ['red','yellow','green']
}, 'Distance from Road');

Map.addLayer(dflineament, {
  min: 0, max: 2000,
  palette: ['red','yellow','green']
}, 'Distance from Lineament');


// =====================================================
// 5. LITHOLOGY → NUMERIC → RASTER
// =====================================================

var lithoClasses = lithology.aggregate_array('DOMINANT_R').distinct();

var lithologyNum = lithology.map(function(f){
  var id = lithoClasses.indexOf(f.get('DOMINANT_R'));
  return f.set('litho_id', id);
});

var lithologyRaster = lithologyNum.reduceToImage({
  properties:['litho_id'],
  reducer: ee.Reducer.first()
}).rename('lithology').clip(studyArea);


// =====================================================
// 6. PREDICTOR STACK (UPDATED ✅)
// =====================================================

var predictors = ee.Image.cat([
  dem.rename('dem'),
  slope.rename('slope'),
  aspect.rename('aspect'),
  ndvi.rename('ndvi'),
  lulc.rename('lulc'),
  rainfall.rename('rainfall'),
  dfriver.rename('dfriver'),
  dfroad.rename('dfroad'),
  dflineament.rename('dflineament'),
  lithologyRaster
]).float();

print('Predictor stack:', predictors);


// =====================================================
// 7. TRAINING & NON‑LANDSLIDE SAMPLES
// =====================================================

var landslidePts = ls_points.map(function(f){
  return f.set('class', 1);
});

var bufferLS = landslidePts.map(function(f){
  return f.buffer(100);
});

var nonLandslidePts = ee.FeatureCollection.randomPoints({
  region: studyArea.difference(bufferLS.geometry(),1),
  points: landslidePts.size(),
  seed: 42
}).map(function(f){
  return f.set('class', 0);
});

var samples = landslidePts.merge(nonLandslidePts);


// =====================================================
// 8. SAMPLE RASTER VALUES
// =====================================================

var sampleData = predictors.sampleRegions({
  collection: samples,
  properties: ['class'],
  scale: 30,
  tileScale: 4
});


// =====================================================
// 9. TRAIN / TEST SPLIT
// =====================================================

var withRandom = sampleData.randomColumn('rand');
var train = withRandom.filter(ee.Filter.lt('rand',0.7));
var test  = withRandom.filter(ee.Filter.gte('rand',0.7));


// =====================================================
// 10. RANDOM FOREST TRAINING
// =====================================================

var rf = ee.Classifier.smileRandomForest({
  numberOfTrees: 500,
  bagFraction: 0.7,
  minLeafPopulation: 2,
  seed: 42
}).train({
  features: train,
  classProperty: 'class',
  inputProperties: predictors.bandNames()
});


// =====================================================
// 11. ACCURACY ASSESSMENT
// =====================================================

var validated = test.classify(rf);
var cm = validated.errorMatrix('class', 'classification');

print('Confusion Matrix:', cm);
print('Overall Accuracy:', cm.accuracy());
print('Kappa:', cm.kappa());


// =====================================================
// 12. RF PROBABILITY OUTPUT
// =====================================================

var rfProb = rf.setOutputMode('PROBABILITY');

var susceptibilityProb = predictors
  .classify(rfProb)
  .clip(studyArea);

Map.addLayer(susceptibilityProb, {
  min: 0, max: 1,
  palette: ['green','yellow','red']
}, 'RF Probability (0–1)');


// =====================================================
// 13. PROBABILITY‑BASED 3 CLASSES
// =====================================================

var susceptibility3 = susceptibilityProb
  .where(susceptibilityProb.lte(0.33), 1)
  .where(susceptibilityProb.gt(0.33)
    .and(susceptibilityProb.lte(0.66)), 2)
  .where(susceptibilityProb.gt(0.66), 3)
  .rename('LSM_Class');


// =====================================================
// 14. FINAL MAP VISUALIZATION
// =====================================================

var lsmVis = {
  min: 1,
  max: 3,
  palette: [
    '#2ca25f', // LOW
    '#ffeb3b', // MEDIUM
    '#de2d26'  // HIGH
  ]
};

Map.addLayer(susceptibility3, lsmVis,
  'Landslide Susceptibility (RF + Distance Factors)');


// =====================================================
// 15. LEGEND
// =====================================================

var legend = ui.Panel({
  style: {position:'bottom-left', padding:'8px'}
});

legend.add(ui.Label({
  value: 'Landslide Susceptibility',
  style: {fontWeight:'bold', fontSize:'14px'}
}));

function makeRow(color, name) {
  return ui.Panel({
    widgets: [
      ui.Label({style:{
        backgroundColor: color,
        padding:'8px',
        margin:'0 4px 4px 0'
      }}),
      ui.Label({value:name})
    ],
    layout: ui.Panel.Layout.Flow('horizontal')
  });
}

legend.add(makeRow('#2ca25f','Low'));
legend.add(makeRow('#ffeb3b','Medium'));
legend.add(makeRow('#de2d26','High'));

Map.add(legend);


// =====================================================
// 16. EXPORT FINAL MAP
// =====================================================

Export.image.toDrive({
  image: susceptibility3,
  description: 'Landslide_Susceptibility_RF_With_Distance_Factors',
  folder: 'GEE_Exports',
  region: studyArea,
  scale: 30,
  maxPixels: 1e13
});
