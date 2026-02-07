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
print('Number of features:', predictors.bandNames().size());
print('Feature names:', predictors.bandNames());


// =====================================================
// 7. TRAINING & NON‑LANDSLIDE SAMPLES
// =====================================================

// ---- Real Landslide Points (Class = 1) ----
var landslidePts = ls_points.map(function(f){
  return f.set('class', 1).set('point_type', 'landslide');
});

// ---- Buffer around landslide points ----
var bufferLS = landslidePts.map(function(f){
  return f.buffer(100);
});

// ---- Artificially Generated Non-Landslide Points (Class = 0) ----
var nonLandslidePts = ee.FeatureCollection.randomPoints({
  region: studyArea.difference(bufferLS.geometry(),1),
  points: landslidePts.size(),
  seed: 42
}).map(function(f){
  return f.set('class', 0).set('point_type', 'non_landslide');
});

// ---- Merge all samples ----
var samples = landslidePts.merge(nonLandslidePts);

// ---- Print sample counts ----
print('═══════════════════════════════════════════');
print('SAMPLE COUNTS:');
print('Real Landslide Points (Class 1):', landslidePts.size());
print('Artificial Non-Landslide Points (Class 0):', nonLandslidePts.size());
print('Total Samples:', samples.size());
print('═══════════════════════════════════════════');


// =====================================================
// 8. SAMPLE RASTER VALUES (WITH COORDINATES)
// =====================================================

var sampleData = predictors.sampleRegions({
  collection: samples,
  properties: ['class', 'point_type'],
  scale: 30,
  tileScale: 4,
  geometries: true  // Keep geometry for coordinates
});

// ---- Add Lat/Lon coordinates to each sample ----
var sampleDataWithCoords = sampleData.map(function(f) {
  var coords = f.geometry().coordinates();
  return f.set({
    'longitude': coords.get(0),
    'latitude': coords.get(1)
  });
});

print('Sample Data with Features:', sampleDataWithCoords.first());
print('Total samples with feature values:', sampleDataWithCoords.size());


// =====================================================
// 9. TRAIN / TEST SPLIT
// =====================================================

var withRandom = sampleDataWithCoords.randomColumn('rand');
var train = withRandom.filter(ee.Filter.lt('rand',0.7));
var test  = withRandom.filter(ee.Filter.gte('rand',0.7));

print('Training samples (70%):', train.size());
print('Testing samples (30%):', test.size());


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

print('═══════════════════════════════════════════');
print('MODEL ACCURACY:');
print('Confusion Matrix:', cm);
print('Overall Accuracy:', cm.accuracy());
print('Kappa:', cm.kappa());
print('═══════════════════════════════════════════');


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
// 16. EXPORT FINAL MAP (RASTER)
// =====================================================

Export.image.toDrive({
  image: susceptibility3,
  description: 'Landslide_Susceptibility_RF_With_Distance_Factors',
  folder: 'GEE_Exports',
  region: studyArea,
  scale: 30,
  maxPixels: 1e13
});


// =====================================================
// 17. EXPORT ALL SAMPLE POINTS TO CSV ✅ (NEW)
// =====================================================

// ---- Define columns for export ----
var exportColumns = [
  'longitude',
  'latitude', 
  'dem',
  'slope',
  'aspect',
  'ndvi',
  'lulc',
  'rainfall',
  'dfriver',
  'dfroad',
  'dflineament',
  'lithology',
  'point_type',
  'class'  // Last column: 1 = Landslide, 0 = Non-Landslide
];

// ---- Export ALL samples (Landslide + Non-Landslide) ----
Export.table.toDrive({
  collection: sampleDataWithCoords,
  description: 'All_Sample_Points_With_Features',
  folder: 'GEE_Exports',
  fileNamePrefix: 'all_sample_points_features',
  fileFormat: 'CSV',
  selectors: exportColumns
});

// ---- Export only LANDSLIDE points (Class = 1) ----
var landslideOnly = sampleDataWithCoords.filter(ee.Filter.eq('class', 1));
Export.table.toDrive({
  collection: landslideOnly,
  description: 'Landslide_Points_Only_With_Features',
  folder: 'GEE_Exports',
  fileNamePrefix: 'landslide_points_features',
  fileFormat: 'CSV',
  selectors: exportColumns
});

// ---- Export only NON-LANDSLIDE points (Class = 0) ----
var nonLandslideOnly = sampleDataWithCoords.filter(ee.Filter.eq('class', 0));
Export.table.toDrive({
  collection: nonLandslideOnly,
  description: 'NonLandslide_Points_Only_With_Features',
  folder: 'GEE_Exports',
  fileNamePrefix: 'non_landslide_points_features',
  fileFormat: 'CSV',
  selectors: exportColumns
});

// ---- Export TRAINING data ----
Export.table.toDrive({
  collection: train,
  description: 'Training_Data_70_Percent',
  folder: 'GEE_Exports',
  fileNamePrefix: 'training_data_70pct',
  fileFormat: 'CSV',
  selectors: exportColumns
});

// ---- Export TESTING data ----
Export.table.toDrive({
  collection: test,
  description: 'Testing_Data_30_Percent',
  folder: 'GEE_Exports',
  fileNamePrefix: 'testing_data_30pct',
  fileFormat: 'CSV',
  selectors: exportColumns
});

print('═══════════════════════════════════════════');
print('CSV EXPORTS READY - Check Tasks Tab ▶️');
print('═══════════════════════════════════════════');
