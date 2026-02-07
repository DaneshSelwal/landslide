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

Map.addLayer(dfriver, {min: 0, max: 2000,
  palette: ['red','yellow','green']}, 'Distance from River');

Map.addLayer(dfroad, {min: 0, max: 2000,
  palette: ['red','yellow','green']}, 'Distance from Road');

Map.addLayer(dflineament, {min: 0, max: 2000,
  palette: ['red','yellow','green']}, 'Distance from Lineament');


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
// 6. PREDICTOR STACK
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


// =====================================================
// 7. TRAINING & NON‑LANDSLIDE SAMPLES
// =====================================================

var landslidePts = ls_points.map(function(f){
  return f.set('class', 1).set('point_type', 'landslide');
});

var bufferLS = landslidePts.map(function(f){
  return f.buffer(100);
});

var nonLandslidePts = ee.FeatureCollection.randomPoints({
  region: studyArea.difference(bufferLS.geometry(),1),
  points: landslidePts.size(),
  seed: 42
}).map(function(f){
  return f.set('class', 0).set('point_type', 'non_landslide');
});

var samples = landslidePts.merge(nonLandslidePts);

print('═══════════════════════════════════════════');
print('SAMPLE COUNTS:');
print('Real Landslide Points (Class 1):', landslidePts.size());
print('Artificial Non-Landslide Points (Class 0):', nonLandslidePts.size());
print('Total Samples:', samples.size());


// =====================================================
// 8. SAMPLE RASTER VALUES
// =====================================================

var sampleData = predictors.sampleRegions({
  collection: samples,
  properties: ['class', 'point_type'],
  scale: 30,
  tileScale: 4,
  geometries: true
});

var sampleDataWithCoords = sampleData.map(function(f) {
  var coords = f.geometry().coordinates();
  return f.set({
    'longitude': coords.get(0),
    'latitude': coords.get(1)
  });
});


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
print('Producers Accuracy:', cm.producersAccuracy());
print('Consumers Accuracy:', cm.consumersAccuracy());


// =====================================================
// 12. ROC CURVE AND AUC CALCULATION ✅ (NEW)
// =====================================================

// ---- Get probability predictions on test data ----
var rfProb = rf.setOutputMode('PROBABILITY');

var testWithProb = test.classify(rfProb, 'probability');

// ---- Function to calculate ROC points ----
function calculateROC(testData, numThresholds) {
  
  var thresholds = ee.List.sequence(0, 1, null, numThresholds);
  
  var rocPoints = thresholds.map(function(threshold) {
    threshold = ee.Number(threshold);
    
    // Classify based on threshold
    var classified = testData.map(function(f) {
      var prob = ee.Number(f.get('probability'));
      var predicted = prob.gte(threshold);
      return f.set('predicted', predicted);
    });
    
    // Calculate TP, FP, TN, FN
    var TP = classified.filter(ee.Filter.and(
      ee.Filter.eq('class', 1),
      ee.Filter.eq('predicted', 1)
    )).size();
    
    var FP = classified.filter(ee.Filter.and(
      ee.Filter.eq('class', 0),
      ee.Filter.eq('predicted', 1)
    )).size();
    
    var TN = classified.filter(ee.Filter.and(
      ee.Filter.eq('class', 0),
      ee.Filter.eq('predicted', 0)
    )).size();
    
    var FN = classified.filter(ee.Filter.and(
      ee.Filter.eq('class', 1),
      ee.Filter.eq('predicted', 0)
    )).size();
    
    // Calculate TPR (Sensitivity) and FPR (1-Specificity)
    var TPR = ee.Number(TP).divide(ee.Number(TP).add(FN));
    var FPR = ee.Number(FP).divide(ee.Number(FP).add(TN));
    
    return ee.Feature(null, {
      'threshold': threshold,
      'TPR': TPR,
      'FPR': FPR,
      'TP': TP,
      'FP': FP,
      'TN': TN,
      'FN': FN
    });
  });
  
  return ee.FeatureCollection(rocPoints);
}

// ---- Calculate ROC with 50 threshold points ----
var rocData = calculateROC(testWithProb, 50);

print('═══════════════════════════════════════════');
print('ROC DATA (sample points):', rocData.limit(10));


// =====================================================
// 13. AUC CALCULATION (Trapezoidal Rule) ✅
// =====================================================

function calculateAUC(rocFC) {
  
  // Sort by FPR
  var sorted = rocFC.sort('FPR');
  var rocList = sorted.toList(sorted.size());
  
  var indices = ee.List.sequence(0, sorted.size().subtract(2));
  
  var auc = indices.map(function(i) {
    i = ee.Number(i).int();
    var current = ee.Feature(rocList.get(i));
    var next = ee.Feature(rocList.get(i.add(1)));
    
    var x1 = ee.Number(current.get('FPR'));
    var x2 = ee.Number(next.get('FPR'));
    var y1 = ee.Number(current.get('TPR'));
    var y2 = ee.Number(next.get('TPR'));
    
    // Trapezoidal area
    var width = x2.subtract(x1).abs();
    var height = y1.add(y2).divide(2);
    
    return width.multiply(height);
  });
  
  return ee.Number(auc.reduce(ee.Reducer.sum()));
}

var AUC = calculateAUC(rocData);

print('═══════════════════════════════════════════');
print('🎯 AUC (Area Under ROC Curve):', AUC);
print('═══════════════════════════════════════════');


// =====================================================
// 14. ROC CURVE VISUALIZATION (Chart) ✅
// =====================================================

var rocChart = ui.Chart.feature.byFeature({
  features: rocData,
  xProperty: 'FPR',
  yProperties: ['TPR']
}).setOptions({
  title: 'ROC Curve - Random Forest Landslide Model',
  hAxis: {
    title: 'False Positive Rate (1 - Specificity)',
    viewWindow: {min: 0, max: 1}
  },
  vAxis: {
    title: 'True Positive Rate (Sensitivity)',
    viewWindow: {min: 0, max: 1}
  },
  lineWidth: 2,
  pointSize: 3,
  series: {
    0: {color: 'blue'}
  },
  legend: {position: 'none'}
});

print('═══════════════════════════════════════════');
print('📊 ROC CURVE:');
print(rocChart);


// =====================================================
// 15. ADDITIONAL METRICS AT OPTIMAL THRESHOLD ✅
// =====================================================

// Find optimal threshold (Youden's J statistic: max(TPR - FPR))
var rocWithJ = rocData.map(function(f) {
  var tpr = ee.Number(f.get('TPR'));
  var fpr = ee.Number(f.get('FPR'));
  var J = tpr.subtract(fpr);
  return f.set('J', J);
});

var optimalPoint = rocWithJ.sort('J', false).first();

print('═══════════════════════════════════════════');
print('📌 OPTIMAL THRESHOLD (Youden J):');
print('Optimal Threshold:', optimalPoint.get('threshold'));
print('TPR at optimal:', optimalPoint.get('TPR'));
print('FPR at optimal:', optimalPoint.get('FPR'));
print('Youden J statistic:', optimalPoint.get('J'));


// =====================================================
// 16. AUC INTERPRETATION ✅
// =====================================================

print('═══════════════════════════════════════════');
print('📊 AUC INTERPRETATION:');
print('AUC = 0.5: No discrimination (random)');
print('AUC = 0.5-0.7: Poor discrimination');
print('AUC = 0.7-0.8: Acceptable discrimination');
print('AUC = 0.8-0.9: Excellent discrimination');
print('AUC = 0.9-1.0: Outstanding discrimination');
print('═══════════════════════════════════════════');


// =====================================================
// 17. RF PROBABILITY OUTPUT & SUSCEPTIBILITY MAP
// =====================================================

var susceptibilityProb = predictors
  .classify(rfProb)
  .clip(studyArea);

Map.addLayer(susceptibilityProb, {
  min: 0, max: 1,
  palette: ['green','yellow','red']
}, 'RF Probability (0–1)');


// =====================================================
// 18. PROBABILITY‑BASED 3 CLASSES
// =====================================================

var susceptibility3 = susceptibilityProb
  .where(susceptibilityProb.lte(0.33), 1)
  .where(susceptibilityProb.gt(0.33)
    .and(susceptibilityProb.lte(0.66)), 2)
  .where(susceptibilityProb.gt(0.66), 3)
  .rename('LSM_Class');


// =====================================================
// 19. FINAL MAP VISUALIZATION
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
// 20. LEGEND
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
// 21. EXPORT FINAL MAP
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
// 22. EXPORT ROC DATA TO CSV ✅
// =====================================================

Export.table.toDrive({
  collection: rocData,
  description: 'ROC_Curve_Data',
  folder: 'GEE_Exports',
  fileNamePrefix: 'roc_curve_data',
  fileFormat: 'CSV',
  selectors: ['threshold', 'FPR', 'TPR', 'TP', 'FP', 'TN', 'FN']
});


// =====================================================
// 23. EXPORT SAMPLE DATA TO CSV
// =====================================================

var exportColumns = [
  'longitude', 'latitude', 
  'dem', 'slope', 'aspect', 'ndvi', 'lulc', 'rainfall',
  'dfriver', 'dfroad', 'dflineament', 'lithology',
  'point_type', 'class'
];

Export.table.toDrive({
  collection: sampleDataWithCoords,
  description: 'All_Sample_Points_With_Features',
  folder: 'GEE_Exports',
  fileNamePrefix: 'all_sample_points_features',
  fileFormat: 'CSV',
  selectors: exportColumns
});

// Export test data with probabilities for external ROC plotting
var testExportColumns = [
  'longitude', 'latitude',
  'dem', 'slope', 'aspect', 'ndvi', 'lulc', 'rainfall',
  'dfriver', 'dfroad', 'dflineament', 'lithology',
  'point_type', 'class', 'probability'
];

Export.table.toDrive({
  collection: testWithProb.map(function(f) {
    var coords = f.geometry().coordinates();
    return f.set({
      'longitude': coords.get(0),
      'latitude': coords.get(1)
    });
  }),
  description: 'Test_Data_With_Probabilities',
  folder: 'GEE_Exports',
  fileNamePrefix: 'test_data_with_probabilities',
  fileFormat: 'CSV',
  selectors: testExportColumns
});

print('═══════════════════════════════════════════');
print('✅ ALL EXPORTS READY - Check Tasks Tab ▶️');
print('═══════════════════════════════════════════');
