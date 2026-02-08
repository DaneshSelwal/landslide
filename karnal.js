var dem = image;
var lulc = image2;
var ndvi = image3;
var rainfall = image4;
var slope = image5;
var soil = image6;
var drainageDensity = image7;
var boundary = table;
var geomorphVector = table2;
var lithologyVector = table3;
var highGWPoints = table4;

var studyArea = boundary.geometry();
Map.centerObject(studyArea, 10);
Map.addLayer(studyArea, {color: 'black'}, 'Study Area (Karnal)', false);

dem = dem.clip(studyArea);
lulc = lulc.clip(studyArea);
ndvi = ndvi.clip(studyArea);
rainfall = rainfall.clip(studyArea);
slope = slope.clip(studyArea);
soil = soil.clip(studyArea);
drainageDensity = drainageDensity.clip(studyArea);

print('═══════════════════════════════════════════');
print('🌊 GROUNDWATER POTENTIAL MAPPING - KARNAL');
print('═══════════════════════════════════════════');

var geomorphClasses = geomorphVector.aggregate_array('DESCRIPTIO').distinct();
var geomorphWithId = geomorphVector.map(function(f) {
  var className = f.get('DESCRIPTIO');
  var id = geomorphClasses.indexOf(className);
  return f.set('geo_id', ee.Algorithms.If(id.eq(-1), 0, id));
});
var geomorphRaster = geomorphWithId.reduceToImage({
  properties: ['geo_id'],
  reducer: ee.Reducer.first()
}).rename('geomorphology').clip(studyArea).unmask(0);

var lithoClasses = lithologyVector.aggregate_array('LITHOLOGIC').distinct();
var lithologyWithId = lithologyVector.map(function(f) {
  var className = f.get('LITHOLOGIC');
  var id = lithoClasses.indexOf(className);
  return f.set('litho_id', ee.Algorithms.If(id.eq(-1), 0, id));
});
var lithologyRaster = lithologyWithId.reduceToImage({
  properties: ['litho_id'],
  reducer: ee.Reducer.first()
}).rename('lithology').clip(studyArea).unmask(0);

var predictors = ee.Image.cat([
  dem.rename('dem'),
  slope.rename('slope'),
  ndvi.rename('ndvi'),
  rainfall.rename('rainfall'),
  lulc.rename('lulc'),
  soil.rename('soil'),
  drainageDensity.rename('drainage_density'),
  geomorphRaster,
  lithologyRaster
]).float();

print('Predictor Bands:', predictors.bandNames());

var highPts = highGWPoints.map(function(f) {
  return f.set({'class': 1, 'potential': 'High'});
});
highPts = highPts.filterBounds(studyArea);
var numHighPts = highPts.size();

print('═══════════════════════════════════════════');
print('📍 TRAINING DATA:');
print('High GW Points (from CSV):', numHighPts);

var highBuffer = highPts.map(function(f) {
  return f.buffer(500);
});
var excludeArea = highBuffer.geometry();

var lowPts = ee.FeatureCollection.randomPoints({
  region: studyArea.difference(excludeArea),
  points: numHighPts,
  seed: 42
}).map(function(f) {
  return f.set({'class': 0, 'potential': 'Low'});
});

print('Low GW Points (random):', lowPts.size());

var allSamples = highPts.merge(lowPts);
print('Total Samples:', allSamples.size());
print('═══════════════════════════════════════════');

Map.addLayer(highPts, {color: 'blue'}, 'High GW Points');
Map.addLayer(lowPts, {color: 'red'}, 'Low GW Points (Random)');

var sampleData = predictors.sampleRegions({
  collection: allSamples,
  properties: ['class', 'potential'],
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

print('Samples with valid raster values:', sampleData.size());

var withRandom = sampleData.randomColumn('rand', 42);
var train = withRandom.filter(ee.Filter.lt('rand', 0.7));
var test = withRandom.filter(ee.Filter.gte('rand', 0.7));

print('═══════════════════════════════════════════');
print('📊 TRAIN/TEST SPLIT:');
print('Training (70%):', train.size());
print('Testing (30%):', test.size());
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

var validated = test.classify(rf);
var cm = validated.errorMatrix('class', 'classification');

print('═══════════════════════════════════════════');
print('🎯 MODEL ACCURACY:');
print('Confusion Matrix:', cm);
print('Overall Accuracy:', cm.accuracy());
print('Kappa:', cm.kappa());
print('═══════════════════════════════════════════');

var importance = ee.Dictionary(rf.explain()).get('importance');
print('📊 Variable Importance:', importance);

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
    title: 'Variable Importance',
    hAxis: {title: 'Variable', slantedText: true, slantedTextAngle: 45},
    vAxis: {title: 'Importance'},
    colors: ['#1a9850'],
    legend: {position: 'none'}
  });
print(importanceChart);

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
      ee.Filter.eq('class', 1), ee.Filter.eq('predicted', 1))).size();
    var FP = classified.filter(ee.Filter.and(
      ee.Filter.eq('class', 0), ee.Filter.eq('predicted', 1))).size();
    var TN = classified.filter(ee.Filter.and(
      ee.Filter.eq('class', 0), ee.Filter.eq('predicted', 0))).size();
    var FN = classified.filter(ee.Filter.and(
      ee.Filter.eq('class', 1), ee.Filter.eq('predicted', 0))).size();
    var TPR = ee.Number(TP).divide(ee.Number(TP).add(FN).max(1));
    var FPR = ee.Number(FP).divide(ee.Number(FP).add(TN).max(1));
    return ee.Feature(null, {
      'threshold': threshold, 'TPR': TPR, 'FPR': FPR,
      'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN
    });
  });
  return ee.FeatureCollection(rocPoints);
}

var rocData = calculateROC(testWithProb, 50);

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
print('🎯 AUC:', AUC);
print('═══════════════════════════════════════════');

var rocChart = ui.Chart.feature.byFeature({
  features: rocData,
  xProperty: 'FPR',
  yProperties: ['TPR']
}).setOptions({
  title: 'ROC Curve',
  hAxis: {title: 'False Positive Rate', viewWindow: {min: 0, max: 1}},
  vAxis: {title: 'True Positive Rate', viewWindow: {min: 0, max: 1}},
  lineWidth: 2,
  pointSize: 3,
  series: {0: {color: 'blue'}},
  legend: {position: 'none'}
});
print(rocChart);

var rocWithJ = rocData.map(function(f) {
  var tpr = ee.Number(f.get('TPR'));
  var fpr = ee.Number(f.get('FPR'));
  return f.set('J', tpr.subtract(fpr));
});
var optimalPoint = rocWithJ.sort('J', false).first();
print('Optimal Threshold:', optimalPoint.get('threshold'));

var gwPotentialProb = predictors.classify(rfProb).clip(studyArea).rename('GW_Probability');

Map.addLayer(gwPotentialProb, {
  min: 0, max: 1,
  palette: ['#d73027', '#fc8d59', '#fee08b', '#91cf60', '#1a9850']
}, 'GW Potential (Probability)');

var pixelArea = ee.Image.pixelArea();

var gwClasses = ee.Image(1)
  .where(gwPotentialProb.gt(0.2), 2)
  .where(gwPotentialProb.gt(0.4), 3)
  .where(gwPotentialProb.gt(0.6), 4)
  .where(gwPotentialProb.gt(0.8), 5)
  .clip(studyArea)
  .rename('GW_Class');

Map.addLayer(gwClasses, {
  min: 1, max: 5,
  palette: ['#d73027', '#fc8d59', '#fee08b', '#91cf60', '#1a9850']
}, 'GW Potential (5 Classes)');

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
print('📐 AREA STATISTICS:');
print('Total Area (sq km):', totalArea);
print(areaStats);
print('═══════════════════════════════════════════');

var areaChart = ui.Chart.feature.byFeature({
  features: areaStats,
  xProperty: 'ClassName',
  yProperties: ['Area_sqkm']
}).setChartType('ColumnChart')
  .setOptions({
    title: 'Area Distribution (sq km)',
    hAxis: {title: 'Class'},
    vAxis: {title: 'Area (sq km)'},
    colors: ['#1a9850'],
    legend: {position: 'none'}
  });
print(areaChart);

var percentChart = ui.Chart.feature.byFeature({
  features: areaStats,
  xProperty: 'ClassName',
  yProperties: ['Percentage']
}).setChartType('ColumnChart')
  .setOptions({
    title: 'Percentage Distribution',
    hAxis: {title: 'Class'},
    vAxis: {title: 'Percentage (%)'},
    colors: ['#2166ac'],
    legend: {position: 'none'}
  });
print(percentChart);

var pieChart = ui.Chart.feature.byFeature({
  features: areaStats,
  xProperty: 'ClassName',
  yProperties: ['Percentage']
}).setChartType('PieChart')
  .setOptions({
    title: 'GW Potential Distribution',
    slices: {
      0: {color: '#d73027'},
      1: {color: '#fc8d59'},
      2: {color: '#fee08b'},
      3: {color: '#91cf60'},
      4: {color: '#1a9850'}
    },
    pieSliceText: 'percentage'
  });
print(pieChart);

areaStats.evaluate(function(fc) {
  if (fc && fc.features) {
    print('┌─────────────┬──────────────┬────────────┐');
    print('│ Class       │ Area (sq km) │ Percentage │');
    print('├─────────────┼──────────────┼────────────┤');
    fc.features.forEach(function(f) {
      var name = String(f.properties.ClassName);
      var area = f.properties.Area_sqkm.toFixed(2);
      var pct = f.properties.Percentage.toFixed(2);
      print('│ ' + name + ' │ ' + area + ' │ ' + pct + '% │');
    });
    print('└─────────────┴──────────────┴────────────┘');
  }
});

var legend = ui.Panel({style: {position: 'bottom-left', padding: '8px 15px'}});
legend.add(ui.Label({
  value: 'GW Potential',
  style: {fontWeight: 'bold', fontSize: '14px', margin: '0 0 6px 0'}
}));

var legendColors = ['#d73027', '#fc8d59', '#fee08b', '#91cf60', '#1a9850'];
for (var i = 0; i < 5; i++) {
  legend.add(ui.Panel({
    widgets: [
      ui.Label({style: {backgroundColor: legendColors[i], padding: '8px', margin: '0 6px 4px 0'}}),
      ui.Label({value: classNames[i]})
    ],
    layout: ui.Panel.Layout.Flow('horizontal')
  }));
}
Map.add(legend);

Export.image.toDrive({
  image: gwPotentialProb,
  description: 'Karnal_GW_Probability',
  folder: 'GEE_Exports',
  region: studyArea,
  scale: 30,
  maxPixels: 1e13
});

Export.image.toDrive({
  image: gwClasses,
  description: 'Karnal_GW_Classes',
  folder: 'GEE_Exports',
  region: studyArea,
  scale: 30,
  maxPixels: 1e13
});

Export.table.toDrive({
  collection: areaStats,
  description: 'Karnal_GW_Area_Statistics',
  folder: 'GEE_Exports',
  fileFormat: 'CSV',
  selectors: ['ClassValue', 'ClassName', 'Area_sqkm', 'Percentage']
});

Export.table.toDrive({
  collection: rocData,
  description: 'Karnal_GW_ROC_Data',
  folder: 'GEE_Exports',
  fileFormat: 'CSV',
  selectors: ['threshold', 'FPR', 'TPR', 'TP', 'FP', 'TN', 'FN']
});

Export.table.toDrive({
  collection: sampleDataWithCoords,
  description: 'Karnal_GW_Training_Points',
  folder: 'GEE_Exports',
  fileFormat: 'CSV'
});

Export.table.toDrive({
  collection: importanceFC,
  description: 'Karnal_GW_Variable_Importance',
  folder: 'GEE_Exports',
  fileFormat: 'CSV',
  selectors: ['variable', 'importance']
});

print('═══════════════════════════════════════════');
print('✅ EXPORTS READY - Check Tasks Tab');
print('═══════════════════════════════════════════');
