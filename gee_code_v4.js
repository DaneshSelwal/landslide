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
// 12. ROC CURVE AND AUC CALCULATION
// =====================================================

var rfProb = rf.setOutputMode('PROBABILITY');
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

var rocData = calculateROC(testWithProb, 50);

print('═══════════════════════════════════════════');
print('ROC DATA (sample points):', rocData.limit(10));


// =====================================================
// 13. AUC CALCULATION
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


// =====================================================
// 14. ROC CURVE VISUALIZATION
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
  series: {0: {color: 'blue'}},
  legend: {position: 'none'}
});

print('═══════════════════════════════════════════');
print('📊 ROC CURVE:');
print(rocChart);


// =====================================================
// 15. OPTIMAL THRESHOLD
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


// =====================================================
// 16. AUC INTERPRETATION
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
// 21. AREA CALCULATION FOR EACH CLASS ✅ (NEW)
// =====================================================

// ---- Define scale (30m resolution) ----
var scale = 30;

// ---- Calculate pixel area image ----
var pixelArea = ee.Image.pixelArea();

// ---- Calculate area for each susceptibility class ----

// LOW Susceptibility (Class 1)
var lowArea = susceptibility3.eq(1).multiply(pixelArea);
var lowAreaSum = lowArea.reduceRegion({
  reducer: ee.Reducer.sum(),
  geometry: studyArea,
  scale: scale,
  maxPixels: 1e13
});
var lowAreaSqKm = ee.Number(lowAreaSum.get('LSM_Class')).divide(1e6);

// MEDIUM Susceptibility (Class 2)
var mediumArea = susceptibility3.eq(2).multiply(pixelArea);
var mediumAreaSum = mediumArea.reduceRegion({
  reducer: ee.Reducer.sum(),
  geometry: studyArea,
  scale: scale,
  maxPixels: 1e13
});
var mediumAreaSqKm = ee.Number(mediumAreaSum.get('LSM_Class')).divide(1e6);

// HIGH Susceptibility (Class 3)
var highArea = susceptibility3.eq(3).multiply(pixelArea);
var highAreaSum = highArea.reduceRegion({
  reducer: ee.Reducer.sum(),
  geometry: studyArea,
  scale: scale,
  maxPixels: 1e13
});
var highAreaSqKm = ee.Number(highAreaSum.get('LSM_Class')).divide(1e6);

// TOTAL Study Area
var totalArea = pixelArea.reduceRegion({
  reducer: ee.Reducer.sum(),
  geometry: studyArea,
  scale: scale,
  maxPixels: 1e13
});
var totalAreaSqKm = ee.Number(totalArea.get('area')).divide(1e6);


// =====================================================
// 22. CALCULATE PERCENTAGES ✅ (NEW)
// =====================================================

var lowPercent = lowAreaSqKm.divide(totalAreaSqKm).multiply(100);
var mediumPercent = mediumAreaSqKm.divide(totalAreaSqKm).multiply(100);
var highPercent = highAreaSqKm.divide(totalAreaSqKm).multiply(100);


// =====================================================
// 23. PRINT AREA STATISTICS ✅ (NEW)
// =====================================================

print('═══════════════════════════════════════════');
print('📐 AREA STATISTICS:');
print('═══════════════════════════════════════════');
print('Total Study Area (sq km):', totalAreaSqKm);
print('───────────────────────────────────────────');
print('Low Susceptibility Area (sq km):', lowAreaSqKm);
print('Medium Susceptibility Area (sq km):', mediumAreaSqKm);
print('High Susceptibility Area (sq km):', highAreaSqKm);
print('───────────────────────────────────────────');
print('Low Susceptibility (%):', lowPercent);
print('Medium Susceptibility (%):', mediumPercent);
print('High Susceptibility (%):', highPercent);
print('═══════════════════════════════════════════');


// =====================================================
// 24. CREATE FEATURE COLLECTION FOR CHARTS ✅ (NEW)
// =====================================================

var areaStats = ee.FeatureCollection([
  ee.Feature(null, {
    'Class': 'Low',
    'Area_sqkm': lowAreaSqKm,
    'Percentage': lowPercent,
    'ClassOrder': 1
  }),
  ee.Feature(null, {
    'Class': 'Medium',
    'Area_sqkm': mediumAreaSqKm,
    'Percentage': mediumPercent,
    'ClassOrder': 2
  }),
  ee.Feature(null, {
    'Class': 'High',
    'Area_sqkm': highAreaSqKm,
    'Percentage': highPercent,
    'ClassOrder': 3
  })
]);


// =====================================================
// 25. BAR CHART - ABSOLUTE AREA (sq km) ✅ (NEW)
// =====================================================

var areaChart = ui.Chart.feature.byFeature({
  features: areaStats,
  xProperty: 'Class',
  yProperties: ['Area_sqkm']
}).setChartType('ColumnChart')
  .setOptions({
    title: 'Landslide Susceptibility - Area Distribution (sq km)',
    hAxis: {
      title: 'Susceptibility Class',
      slantedText: false
    },
    vAxis: {
      title: 'Area (sq km)',
      minValue: 0
    },
    colors: ['#2ca25f', '#ffeb3b', '#de2d26'],
    legend: {position: 'none'},
    bar: {groupWidth: '70%'},
    chartArea: {width: '70%', height: '70%'}
  });

print('═══════════════════════════════════════════');
print('📊 AREA DISTRIBUTION CHART (sq km):');
print(areaChart);


// =====================================================
// 26. BAR CHART - PERCENTAGE AREA ✅ (NEW)
// =====================================================

var percentChart = ui.Chart.feature.byFeature({
  features: areaStats,
  xProperty: 'Class',
  yProperties: ['Percentage']
}).setChartType('ColumnChart')
  .setOptions({
    title: 'Landslide Susceptibility - Percentage Distribution (%)',
    hAxis: {
      title: 'Susceptibility Class',
      slantedText: false
    },
    vAxis: {
      title: 'Percentage (%)',
      minValue: 0,
      maxValue: 100
    },
    colors: ['#2ca25f', '#ffeb3b', '#de2d26'],
    legend: {position: 'none'},
    bar: {groupWidth: '70%'},
    chartArea: {width: '70%', height: '70%'}
  });

print('═══════════════════════════════════════════');
print('📊 PERCENTAGE DISTRIBUTION CHART (%):');
print(percentChart);


// =====================================================
// 27. PIE CHART - PERCENTAGE DISTRIBUTION ✅ (NEW)
// =====================================================

var pieChart = ui.Chart.feature.byFeature({
  features: areaStats,
  xProperty: 'Class',
  yProperties: ['Percentage']
}).setChartType('PieChart')
  .setOptions({
    title: 'Landslide Susceptibility - Area Proportion',
    slices: {
      0: {color: '#2ca25f'},  // Low - Green
      1: {color: '#ffeb3b'},  // Medium - Yellow
      2: {color: '#de2d26'}   // High - Red
    },
    pieSliceText: 'percentage',
    legend: {position: 'right'},
    chartArea: {width: '90%', height: '80%'}
  });

print('═══════════════════════════════════════════');
print('🥧 PIE CHART - AREA PROPORTION:');
print(pieChart);


// =====================================================
// 28. COMBINED CHART (AREA + PERCENTAGE) ✅ (NEW)
// =====================================================

var combinedChart = ui.Chart.feature.byFeature({
  features: areaStats,
  xProperty: 'Class',
  yProperties: ['Area_sqkm', 'Percentage']
}).setChartType('ColumnChart')
  .setOptions({
    title: 'Landslide Susceptibility - Combined Statistics',
    hAxis: {
      title: 'Susceptibility Class'
    },
    vAxes: {
      0: {title: 'Area (sq km)', minValue: 0},
      1: {title: 'Percentage (%)', minValue: 0, maxValue: 100}
    },
    series: {
      0: {targetAxisIndex: 0, color: '#1f77b4', type: 'bars'},
      1: {targetAxisIndex: 1, color: '#ff7f0e', type: 'bars'}
    },
    bar: {groupWidth: '70%'},
    legend: {position: 'top'},
    chartArea: {width: '70%', height: '65%'}
  });

print('═══════════════════════════════════════════');
print('📊 COMBINED CHART (Area & Percentage):');
print(combinedChart);


// =====================================================
// 29. SUMMARY TABLE ✅ (NEW)
// =====================================================

print('═══════════════════════════════════════════');
print('📋 SUMMARY TABLE:');
print('═══════════════════════════════════════════');
print('┌─────────────┬──────────────┬────────────┐');
print('│ Class       │ Area (sq km) │ Percentage │');
print('├─────────────┼──────────────┼────────────┤');

// Format the values for printing
lowAreaSqKm.evaluate(function(low) {
  mediumAreaSqKm.evaluate(function(med) {
    highAreaSqKm.evaluate(function(high) {
      totalAreaSqKm.evaluate(function(total) {
        lowPercent.evaluate(function(lowP) {
          mediumPercent.evaluate(function(medP) {
            highPercent.evaluate(function(highP) {
              print('│ Low         │ ' + low.toFixed(2) + '       │ ' + lowP.toFixed(2) + '%     │');
              print('│ Medium      │ ' + med.toFixed(2) + '       │ ' + medP.toFixed(2) + '%     │');
              print('│ High        │ ' + high.toFixed(2) + '       │ ' + highP.toFixed(2) + '%     │');
              print('├─────────────┼──────────────┼────────────┤');
              print('│ TOTAL       │ ' + total.toFixed(2) + '       │ 100.00%    │');
              print('└─────────────┴──────────────┴────────────┘');
            });
          });
        });
      });
    });
  });
});


// =====================================================
// 30. EXPORT AREA STATISTICS TO CSV ✅ (NEW)
// =====================================================

Export.table.toDrive({
  collection: areaStats,
  description: 'Susceptibility_Area_Statistics',
  folder: 'GEE_Exports',
  fileNamePrefix: 'susceptibility_area_stats',
  fileFormat: 'CSV',
  selectors: ['Class', 'Area_sqkm', 'Percentage']
});


// =====================================================
// 31. EXPORT FINAL MAP
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
// 32. EXPORT ROC DATA TO CSV
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
// 33. EXPORT SAMPLE DATA TO CSV
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
