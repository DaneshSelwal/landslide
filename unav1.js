// =====================================================
// GROUNDWATER MAPPING - SYNTHETIC TRAINING DATA + RF
// =====================================================
// FIXED VERSION - Uses scoring system for robust point generation
// =====================================================

// =====================================================
// 1. RENAME IMPORTED ASSETS
// =====================================================

var geomorphology = image;
var ndvi          = image2;
var rainfall      = image3;
var soil          = image4;
var drainage      = image5;
var lineament     = image6;
var lulc          = image7;
var slope         = image8;
var boundary      = table;
var lithology     = image9;

var studyArea = boundary.geometry();
Map.centerObject(studyArea, 10);
Map.addLayer(studyArea, {color: 'black'}, 'Study Area', false);


// =====================================================
// 2. CLIP ALL RASTERS
// =====================================================

geomorphology = geomorphology.clip(studyArea);
ndvi          = ndvi.clip(studyArea);
rainfall      = rainfall.clip(studyArea);
soil          = soil.clip(studyArea);
drainage      = drainage.clip(studyArea);
lineament     = lineament.clip(studyArea);
lulc          = lulc.clip(studyArea);
slope         = slope.clip(studyArea);
lithology     = lithology.clip(studyArea);


// =====================================================
// 3. CHECK RASTER VALUE RANGES (DIAGNOSTIC)
// =====================================================

print('═══════════════════════════════════════════');
print('📊 RASTER VALUE RANGES:');
print('═══════════════════════════════════════════');

var checkRange = function(img, name) {
  var stats = img.reduceRegion({
    reducer: ee.Reducer.minMax().combine({
      reducer2: ee.Reducer.mean(),
      sharedInputs: true
    }),
    geometry: studyArea,
    scale: 100,
    maxPixels: 1e9,
    bestEffort: true
  });
  print(name + ':', stats);
};

checkRange(slope, 'Slope');
checkRange(drainage, 'Drainage');
checkRange(lineament, 'Lineament');
checkRange(soil, 'Soil');
checkRange(geomorphology, 'Geomorphology');
checkRange(rainfall, 'Rainfall');
checkRange(ndvi, 'NDVI');
checkRange(lulc, 'LULC');
checkRange(lithology, 'Lithology');


// =====================================================
// 4. NORMALIZE ALL RASTERS TO 0-1 SCALE
// =====================================================
// This ensures all layers contribute equally to the score

print('═══════════════════════════════════════════');
print('🔧 NORMALIZING RASTERS TO 0-1 SCALE...');
print('═══════════════════════════════════════════');

function normalizeImage(img, bandName) {
  var minMax = img.reduceRegion({
    reducer: ee.Reducer.minMax(),
    geometry: studyArea,
    scale: 100,
    maxPixels: 1e9,
    bestEffort: true
  });
  
  var bandNames = img.bandNames();
  var firstBand = ee.String(bandNames.get(0));
  
  var minKey = firstBand.cat('_min');
  var maxKey = firstBand.cat('_max');
  
  var minVal = ee.Number(minMax.get(minKey));
  var maxVal = ee.Number(minMax.get(maxKey));
  
  var range = maxVal.subtract(minVal);
  
  // Handle case where min = max (constant image)
  var normalized = ee.Image(ee.Algorithms.If(
    range.eq(0),
    img.multiply(0).add(0.5),
    img.subtract(minVal).divide(range)
  )).rename(bandName);
  
  return normalized.clamp(0, 1);
}

// Normalize each layer
// IMPORTANT: Adjust the sign based on relationship with groundwater potential
// Higher value = MORE favorable for groundwater → use as is
// Higher value = LESS favorable for groundwater → invert (1 - normalized)

var slopeNorm = ee.Image(1).subtract(normalizeImage(slope, 'slope_norm'));  // INVERT: steep slope = bad
var drainageNorm = normalizeImage(drainage, 'drainage_norm');               // Near drainage = good
var lineamentNorm = normalizeImage(lineament, 'lineament_norm');            // Near lineament = good
var soilNorm = normalizeImage(soil, 'soil_norm');                           // High permeability = good
var geoNorm = normalizeImage(geomorphology, 'geo_norm');                    // Favorable geomorphology = good
var rainfallNorm = normalizeImage(rainfall, 'rainfall_norm');               // High rainfall = good
var ndviNorm = normalizeImage(ndvi, 'ndvi_norm');                           // High NDVI = good
var lulcNorm = normalizeImage(lulc, 'lulc_norm');                           // Depends on classes
var lithoNorm = normalizeImage(lithology, 'litho_norm');                    // Depends on classes


// =====================================================
// 5. CREATE GROUNDWATER FAVORABILITY SCORE
// =====================================================
// Sum of normalized layers weighted by importance

print('═══════════════════════════════════════════');
print('📊 CREATING FAVORABILITY SCORE...');
print('═══════════════════════════════════════════');

// Weights for scoring (adjust based on hydrogeological importance)
var w_slope = 0.15;
var w_drainage = 0.12;
var w_lineament = 0.15;
var w_soil = 0.12;
var w_geo = 0.15;
var w_rainfall = 0.08;
var w_ndvi = 0.05;
var w_lulc = 0.08;
var w_litho = 0.10;

var favorabilityScore = slopeNorm.multiply(w_slope)
  .add(drainageNorm.multiply(w_drainage))
  .add(lineamentNorm.multiply(w_lineament))
  .add(soilNorm.multiply(w_soil))
  .add(geoNorm.multiply(w_geo))
  .add(rainfallNorm.multiply(w_rainfall))
  .add(ndviNorm.multiply(w_ndvi))
  .add(lulcNorm.multiply(w_lulc))
  .add(lithoNorm.multiply(w_litho))
  .rename('favorability');

// Visualize the score
Map.addLayer(favorabilityScore, {
  min: 0, max: 1,
  palette: ['red', 'orange', 'yellow', 'lightgreen', 'darkgreen']
}, 'Favorability Score (0-1)', false);


// =====================================================
// 6. CALCULATE PERCENTILE THRESHOLDS
// =====================================================

print('═══════════════════════════════════════════');
print('📊 CALCULATING PERCENTILE THRESHOLDS...');
print('═══════════════════════════════════════════');

// Get percentile values for defining high/low potential
var percentiles = favorabilityScore.reduceRegion({
  reducer: ee.Reducer.percentile([10, 20, 25, 75, 80, 90]),
  geometry: studyArea,
  scale: 100,
  maxPixels: 1e9,
  bestEffort: true
});

print('Percentile Values:', percentiles);

var p20 = ee.Number(percentiles.get('favorability_p20'));
var p80 = ee.Number(percentiles.get('favorability_p80'));

print('Low threshold (20th percentile):', p20);
print('High threshold (80th percentile):', p80);


// =====================================================
// 7. DEFINE HIGH/LOW POTENTIAL ZONES USING PERCENTILES
// =====================================================

// HIGH POTENTIAL: Top 20% of favorability scores
var highPotentialMask = favorabilityScore.gte(p80);

// LOW POTENTIAL: Bottom 20% of favorability scores
var lowPotentialMask = favorabilityScore.lte(p20);

// Visualize the masks
Map.addLayer(highPotentialMask.selfMask(), {palette: ['blue']}, 'High Potential Mask', false);
Map.addLayer(lowPotentialMask.selfMask(), {palette: ['red']}, 'Low Potential Mask', false);


// =====================================================
// 8. CALCULATE MASK AREAS (DIAGNOSTIC)
// =====================================================

var pixelArea = ee.Image.pixelArea();

var highMaskArea = highPotentialMask.multiply(pixelArea).reduceRegion({
  reducer: ee.Reducer.sum(),
  geometry: studyArea,
  scale: 100,
  maxPixels: 1e10,
  bestEffort: true
});

var lowMaskArea = lowPotentialMask.multiply(pixelArea).reduceRegion({
  reducer: ee.Reducer.sum(),
  geometry: studyArea,
  scale: 100,
  maxPixels: 1e10,
  bestEffort: true
});

var totalStudyArea = pixelArea.reduceRegion({
  reducer: ee.Reducer.sum(),
  geometry: studyArea,
  scale: 100,
  maxPixels: 1e10,
  bestEffort: true
});

print('═══════════════════════════════════════════');
print('📐 MASK AREA DIAGNOSTICS:');
print('═══════════════════════════════════════════');
print('Total Study Area (sq km):', ee.Number(totalStudyArea.get('area')).divide(1e6));
print('High Potential Mask Area (sq km):', ee.Number(highMaskArea.values().get(0)).divide(1e6));
print('Low Potential Mask Area (sq km):', ee.Number(lowMaskArea.values().get(0)).divide(1e6));
print('═══════════════════════════════════════════');


// =====================================================
// 9. CONVERT MASKS TO VECTORS FOR POINT GENERATION
// =====================================================

print('═══════════════════════════════════════════');
print('🔧 CONVERTING MASKS TO VECTORS...');
print('═══════════════════════════════════════════');

// Convert masks to vectors with error handling
var highPotentialVectors = highPotentialMask.selfMask().reduceToVectors({
  geometry: studyArea,
  scale: 100,
  geometryType: 'polygon',
  maxPixels: 1e10,
  bestEffort: true,
  eightConnected: true
});

var lowPotentialVectors = lowPotentialMask.selfMask().reduceToVectors({
  geometry: studyArea,
  scale: 100,
  geometryType: 'polygon',
  maxPixels: 1e10,
  bestEffort: true,
  eightConnected: true
});

// Get geometries
var highPotentialArea = highPotentialVectors.geometry();
var lowPotentialArea = lowPotentialVectors.geometry();

// Check if geometries are valid
print('High Potential Vectors count:', highPotentialVectors.size());
print('Low Potential Vectors count:', lowPotentialVectors.size());


// =====================================================
// 10. GENERATE SYNTHETIC TRAINING POINTS
// =====================================================

var numPoints = 300;  // Adjust based on your study area size

print('═══════════════════════════════════════════');
print('🎯 POINT GENERATION SETTINGS:');
print('Requested points per class:', numPoints);
print('Total requested points:', numPoints * 2);
print('═══════════════════════════════════════════');

// Generate random points with error handling
var highPotentialPts = ee.FeatureCollection.randomPoints({
  region: highPotentialArea,
  points: numPoints,
  seed: 42
}).map(function(f) {
  return f.set({'class': 1, 'potential': 'High'});
});

var lowPotentialPts = ee.FeatureCollection.randomPoints({
  region: lowPotentialArea,
  points: numPoints,
  seed: 123
}).map(function(f) {
  return f.set({'class': 0, 'potential': 'Low'});
});

// Combine samples
var samples = highPotentialPts.merge(lowPotentialPts);


// =====================================================
// 11. PRINT POINT COUNTS
// =====================================================

print('═══════════════════════════════════════════');
print('📍 SYNTHETIC TRAINING DATA GENERATED:');
print('═══════════════════════════════════════════');
print('High Potential Points (Class 1):', highPotentialPts.size());
print('Low Potential Points (Class 0):', lowPotentialPts.size());
print('Total Samples:', samples.size());
print('═══════════════════════════════════════════');

// Visualize training points
Map.addLayer(highPotentialPts, {color: 'blue'}, 'High Potential Training Points');
Map.addLayer(lowPotentialPts, {color: 'red'}, 'Low Potential Training Points');


// =====================================================
// 12. CREATE PREDICTOR STACK
// =====================================================

var predictors = ee.Image.cat([
  geomorphology.rename('geomorphology'),
  ndvi.rename('ndvi'),
  rainfall.rename('rainfall'),
  soil.rename('soil'),
  drainage.rename('drainage'),
  lineament.rename('lineament'),
  lulc.rename('lulc'),
  slope.rename('slope'),
  lithology.rename('lithology')
]).float();

print('═══════════════════════════════════════════');
print('Predictor Stack:', predictors);
print('Band Names:', predictors.bandNames());
print('Number of Bands:', predictors.bandNames().size());
print('═══════════════════════════════════════════');


// =====================================================
// 13. SAMPLE RASTER VALUES
// =====================================================

var sampleData = predictors.sampleRegions({
  collection: samples,
  properties: ['class', 'potential'],
  scale: 30,
  tileScale: 4,
  geometries: true
});

// Add coordinates to sample data
var sampleDataWithCoords = sampleData.map(function(f) {
  var coords = f.geometry().coordinates();
  return f.set({
    'longitude': coords.get(0),
    'latitude': coords.get(1)
  });
});

print('═══════════════════════════════════════════');
print('📊 SAMPLED DATA:');
print('Samples with valid raster values:', sampleData.size());
print('═══════════════════════════════════════════');


// =====================================================
// 14. TRAIN / TEST SPLIT
// =====================================================

var withRandom = sampleData.randomColumn('rand', 42);
var train = withRandom.filter(ee.Filter.lt('rand', 0.7));
var test = withRandom.filter(ee.Filter.gte('rand', 0.7));

print('═══════════════════════════════════════════');
print('📊 TRAIN/TEST SPLIT:');
print('═══════════════════════════════════════════');
print('Training samples (70%):', train.size());
print('Testing samples (30%):', test.size());

// Class distribution
var trainHigh = train.filter(ee.Filter.eq('class', 1)).size();
var trainLow = train.filter(ee.Filter.eq('class', 0)).size();
var testHigh = test.filter(ee.Filter.eq('class', 1)).size();
var testLow = test.filter(ee.Filter.eq('class', 0)).size();

print('Training - High Potential:', trainHigh);
print('Training - Low Potential:', trainLow);
print('Testing - High Potential:', testHigh);
print('Testing - Low Potential:', testLow);
print('═══════════════════════════════════════════');


// =====================================================
// 15. RANDOM FOREST TRAINING
// =====================================================

print('═══════════════════════════════════════════');
print('🌲 TRAINING RANDOM FOREST MODEL...');
print('═══════════════════════════════════════════');

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
// 16. ACCURACY ASSESSMENT
// =====================================================

var validated = test.classify(rf);
var cm = validated.errorMatrix('class', 'classification');

print('═══════════════════════════════════════════');
print('🎯 MODEL ACCURACY:');
print('═══════════════════════════════════════════');
print('Confusion Matrix:', cm);
print('Overall Accuracy:', cm.accuracy());
print('Kappa Coefficient:', cm.kappa());
print('Producers Accuracy:', cm.producersAccuracy());
print('Consumers Accuracy:', cm.consumersAccuracy());
print('═══════════════════════════════════════════');


// =====================================================
// 17. VARIABLE IMPORTANCE
// =====================================================

var importance = ee.Dictionary(rf.explain()).get('importance');

print('═══════════════════════════════════════════');
print('📊 VARIABLE IMPORTANCE:');
print('═══════════════════════════════════════════');
print(importance);

// Create importance chart
var importanceFC = ee.FeatureCollection(
  ee.Dictionary(importance).keys().map(function(key) {
    return ee.Feature(null, {
      'variable': key,
      'importance': ee.Dictionary(importance).get(key)
    });
  })
);

var importanceChart = ui.Chart.feature.byFeature({
  features: importanceFC,
  xProperty: 'variable',
  yProperties: ['importance']
}).setChartType('ColumnChart')
  .setOptions({
    title: 'Variable Importance - Random Forest',
    hAxis: {title: 'Variable', slantedText: true, slantedTextAngle: 45},
    vAxis: {title: 'Importance'},
    colors: ['#1a9850'],
    legend: {position: 'none'},
    chartArea: {width: '70%', height: '65%'}
  });

print('📊 VARIABLE IMPORTANCE CHART:');
print(importanceChart);


// =====================================================
// 18. PROBABILITY OUTPUT
// =====================================================

var rfProb = rf.setOutputMode('PROBABILITY');

var gwPotentialProb = predictors
  .classify(rfProb)
  .clip(studyArea)
  .rename('GW_Probability');

Map.addLayer(gwPotentialProb, {
  min: 0, max: 1,
  palette: ['#d73027', '#fc8d59', '#fee08b', '#91cf60', '#1a9850']
}, 'GW Potential (RF Probability)');


// =====================================================
// 19. CLASSIFY INTO 5 ZONES
// =====================================================

var gwClasses = ee.Image(1)
  .where(gwPotentialProb.gt(0.2), 2)
  .where(gwPotentialProb.gt(0.4), 3)
  .where(gwPotentialProb.gt(0.6), 4)
  .where(gwPotentialProb.gt(0.8), 5)
  .clip(studyArea)
  .rename('GW_Class');

var classVis = {
  min: 1, max: 5,
  palette: ['#d73027', '#fc8d59', '#fee08b', '#91cf60', '#1a9850']
};

Map.addLayer(gwClasses, classVis, 'GW Potential Classes (RF)');


// =====================================================
// 20. ROC CURVE AND AUC
// =====================================================

var testWithProb = test.classify(rfProb, 'probability');

function calculateROC(testData, numThresholds) {
  var thresholds = ee.List.sequence(0, 1, null, numThresholds);
  
  var rocPoints = thresholds.map(function(threshold) {
    threshold = ee.Number(threshold);
    
    var classified = testData.map(function(f) {
      var prob = ee.Number(f.get('probability'));
      var predicted = prob.gte(threshold);
      return f.set('predicted', predicted);
    });
    
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
    
    var TPR = ee.Number(TP).divide(ee.Number(TP).add(FN).max(1));
    var FPR = ee.Number(FP).divide(ee.Number(FP).add(TN).max(1));
    
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

var rocData = calculateROC(testWithProb, 50);


// =====================================================
// 21. AUC CALCULATION
// =====================================================

function calculateAUC(rocFC) {
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
print('AUC Interpretation:');
print('  0.5 = Random (no discrimination)');
print('  0.5-0.7 = Poor');
print('  0.7-0.8 = Acceptable');
print('  0.8-0.9 = Excellent');
print('  0.9-1.0 = Outstanding');
print('═══════════════════════════════════════════');


// ROC Chart
var rocChart = ui.Chart.feature.byFeature({
  features: rocData,
  xProperty: 'FPR',
  yProperties: ['TPR']
}).setOptions({
  title: 'ROC Curve - Groundwater Potential Model',
  hAxis: {title: 'False Positive Rate (1 - Specificity)', viewWindow: {min: 0, max: 1}},
  vAxis: {title: 'True Positive Rate (Sensitivity)', viewWindow: {min: 0, max: 1}},
  lineWidth: 2,
  pointSize: 3,
  series: {0: {color: 'blue'}},
  legend: {position: 'none'},
  chartArea: {width: '70%', height: '70%'}
});

print('═══════════════════════════════════════════');
print('📊 ROC CURVE:');
print(rocChart);


// =====================================================
// 22. OPTIMAL THRESHOLD (YOUDEN'S J)
// =====================================================

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
print('═══════════════════════════════════════════');


// =====================================================
// 23. AREA STATISTICS
// =====================================================

var scale = 30;

var classNames = ['Very Low', 'Low', 'Moderate', 'High', 'Very High'];

var areaStats = ee.FeatureCollection([1, 2, 3, 4, 5].map(function(classVal) {
  var classArea = gwClasses.eq(classVal).multiply(pixelArea);
  var areaSum = classArea.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: studyArea,
    scale: scale,
    maxPixels: 1e13,
    bestEffort: true
  });
  var areaSqKm = ee.Number(areaSum.get('GW_Class')).divide(1e6);
  
  return ee.Feature(null, {
    'ClassValue': classVal,
    'ClassName': ee.List(classNames).get(classVal - 1),
    'Area_sqkm': areaSqKm
  });
}));

// Calculate total and percentages
var totalArea = ee.Number(pixelArea.reduceRegion({
  reducer: ee.Reducer.sum(),
  geometry: studyArea,
  scale: scale,
  maxPixels: 1e13,
  bestEffort: true
}).get('area')).divide(1e6);

areaStats = areaStats.map(function(f) {
  var pct = ee.Number(f.get('Area_sqkm')).divide(totalArea).multiply(100);
  return f.set('Percentage', pct);
});

print('═══════════════════════════════════════════');
print('📐 GROUNDWATER POTENTIAL - AREA STATISTICS:');
print('═══════════════════════════════════════════');
print('Total Study Area (sq km):', totalArea);
print(areaStats);


// =====================================================
// 24. AREA CHARTS
// =====================================================

var areaChart = ui.Chart.feature.byFeature({
  features: areaStats,
  xProperty: 'ClassName',
  yProperties: ['Area_sqkm']
}).setChartType('ColumnChart')
  .setOptions({
    title: 'Groundwater Potential - Area Distribution (sq km)',
    hAxis: {title: 'Potential Class'},
    vAxis: {title: 'Area (sq km)', minValue: 0},
    colors: ['#1a9850'],
    legend: {position: 'none'},
    chartArea: {width: '70%', height: '65%'}
  });

print('📊 AREA DISTRIBUTION CHART:');
print(areaChart);

var percentChart = ui.Chart.feature.byFeature({
  features: areaStats,
  xProperty: 'ClassName',
  yProperties: ['Percentage']
}).setChartType('ColumnChart')
  .setOptions({
    title: 'Groundwater Potential - Percentage Distribution (%)',
    hAxis: {title: 'Potential Class'},
    vAxis: {title: 'Percentage (%)', minValue: 0, maxValue: 100},
    colors: ['#2166ac'],
    legend: {position: 'none'},
    chartArea: {width: '70%', height: '65%'}
  });

print('📊 PERCENTAGE DISTRIBUTION CHART:');
print(percentChart);

var pieChart = ui.Chart.feature.byFeature({
  features: areaStats,
  xProperty: 'ClassName',
  yProperties: ['Percentage']
}).setChartType('PieChart')
  .setOptions({
    title: 'Groundwater Potential Zone Distribution',
    slices: {
      0: {color: '#d73027'},
      1: {color: '#fc8d59'},
      2: {color: '#fee08b'},
      3: {color: '#91cf60'},
      4: {color: '#1a9850'}
    },
    pieSliceText: 'percentage',
    legend: {position: 'right'},
    chartArea: {width: '90%', height: '80%'}
  });

print('🥧 PIE CHART:');
print(pieChart);


// =====================================================
// 25. SUMMARY TABLE (FORMATTED OUTPUT) - FIXED
// =====================================================

print('═══════════════════════════════════════════');
print('📋 AREA SUMMARY TABLE:');
print('═══════════════════════════════════════════');

// Evaluate and print formatted table - FIXED VERSION
areaStats.evaluate(function(fc) {
  if (fc && fc.features) {
    print('┌─────────────┬──────────────┬────────────┐');
    print('│ Class       │ Area (sq km) │ Percentage │');
    print('├─────────────┼──────────────┼────────────┤');
    
    fc.features.forEach(function(f) {
      var name = String(f.properties.ClassName);
      var area = f.properties.Area_sqkm.toFixed(2);
      var pct = f.properties.Percentage.toFixed(2);
      
      // Simple formatting without padEnd/padStart
      print('│ ' + name + ' │ ' + area + ' sq km │ ' + pct + '% │');
    });
    
    print('└─────────────┴──────────────┴────────────┘');
  }
});

totalArea.evaluate(function(total) {
  print('Total Study Area: ' + total.toFixed(2) + ' sq km');
});


// =====================================================
// 26. LEGEND
// =====================================================

var legend = ui.Panel({
  style: {position: 'bottom-left', padding: '8px 15px'}
});

legend.add(ui.Label({
  value: 'Groundwater Potential (RF)',
  style: {fontWeight: 'bold', fontSize: '16px', margin: '0 0 10px 0'}
}));

var legendColors = ['#d73027', '#fc8d59', '#fee08b', '#91cf60', '#1a9850'];

for (var i = 0; i < 5; i++) {
  legend.add(ui.Panel({
    widgets: [
      ui.Label({style: {
        backgroundColor: legendColors[i],
        padding: '10px',
        margin: '0 8px 4px 0'
      }}),
      ui.Label({value: classNames[i]})
    ],
    layout: ui.Panel.Layout.Flow('horizontal')
  }));
}

Map.add(legend);


// =====================================================
// 27. EXPORTS
// =====================================================

// Export probability map
Export.image.toDrive({
  image: gwPotentialProb,
  description: 'GW_Potential_RF_Probability',
  folder: 'GEE_Exports',
  region: studyArea,
  scale: 30,
  maxPixels: 1e13
});

// Export classified map
Export.image.toDrive({
  image: gwClasses,
  description: 'GW_Potential_RF_Classes',
  folder: 'GEE_Exports',
  region: studyArea,
  scale: 30,
  maxPixels: 1e13
});

// Export favorability score
Export.image.toDrive({
  image: favorabilityScore,
  description: 'GW_Favorability_Score',
  folder: 'GEE_Exports',
  region: studyArea,
  scale: 30,
  maxPixels: 1e13
});

// Export area statistics
Export.table.toDrive({
  collection: areaStats,
  description: 'GW_Potential_Area_Statistics',
  folder: 'GEE_Exports',
  fileFormat: 'CSV',
  selectors: ['ClassValue', 'ClassName', 'Area_sqkm', 'Percentage']
});

// Export ROC data
Export.table.toDrive({
  collection: rocData,
  description: 'GW_ROC_Curve_Data',
  folder: 'GEE_Exports',
  fileFormat: 'CSV',
  selectors: ['threshold', 'FPR', 'TPR', 'TP', 'FP', 'TN', 'FN']
});

// Export training sample points
Export.table.toDrive({
  collection: sampleDataWithCoords,
  description: 'GW_Training_Sample_Points',
  folder: 'GEE_Exports',
  fileFormat: 'CSV'
});

// Export variable importance
Export.table.toDrive({
  collection: importanceFC,
  description: 'GW_Variable_Importance',
  folder: 'GEE_Exports',
  fileFormat: 'CSV',
  selectors: ['variable', 'importance']
});

print('═══════════════════════════════════════════');
print('✅ ALL EXPORTS READY - Check Tasks Tab ▶️');
print('═══════════════════════════════════════════');
print('Exports available:');
print('  1. GW_Potential_RF_Probability (GeoTIFF)');
print('  2. GW_Potential_RF_Classes (GeoTIFF)');
print('  3. GW_Favorability_Score (GeoTIFF)');
print('  4. GW_Potential_Area_Statistics (CSV)');
print('  5. GW_ROC_Curve_Data (CSV)');
print('  6. GW_Training_Sample_Points (CSV)');
print('  7. GW_Variable_Importance (CSV)');
print('═══════════════════════════════════════════');
