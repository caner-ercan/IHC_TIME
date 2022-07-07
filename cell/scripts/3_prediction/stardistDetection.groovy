import qupath.tensorflow.stardist.StarDist2D


// Specify the model directory (you will need to change this!)
def modelDir = "/scicore/home/pissal00/ercan0000/220310_training/cellClassification/211107"
def pathModel = buildFilePath(modelDir)

double pixelSize = getCurrentServer().getPixelCalibration().getAveragedPixelSize()
def stardist = StarDist2D.builder(pathModel)
      //.channels(ColorTransforms.createColorDeconvolvedChannel(getCurrentImageData().getColorDeconvolutionStains(), 1)) 
      .threshold(0.05)              // Prediction threshold
      .normalizePercentiles(1, 99) // Percentile normalization
      .pixelSize(pixelSize)              // Resolution for detection
       //.pixelSize(0.25) 
       .cellExpansion(5.0) 
        .measureShape()              // Add shape measurements
        //.measureIntensity()          // Add cell measurements (in all compartments)
        .constrainToParent(false)
      .includeProbability(false)
      //.classify("Tumor")  
      .build()

// Run detection for the selected objects
def imageData = getCurrentImageData()
selectObjectsByClassification("analyse")
//selectAnnotations();
def pathObjects = getSelectedObjects()
if (pathObjects.isEmpty()) {
    Dialogs.showErrorMessage("StarDist", "Please select a parent object!")
    return
}
stardist.detectObjects(imageData, pathObjects)


//selectAnnotations()
//resolveHierarchy()

println getProjectEntry().getImageName()+' Done!'



selectDetections();
runPlugin('qupath.lib.algorithms.IntensityFeaturesPlugin', '{"pixelSizeMicrons": 0.24,  "region": "ROI",  "tileSizeMicrons": 25.0,  "colorOD": true,  "colorStain1": true,  "colorStain2": true,  "colorStain3": false,  "colorRed": false,  "colorGreen": false,  "colorBlue": false,  "colorHue": false,  "colorSaturation": false,  "colorBrightness": false,  "doMean": true,  "doStdDev": true,  "doMinMax": true,  "doMedian": true,  "doHaralick": true,  "haralickDistance": 1,  "haralickBins": 32}');
//addShapeMeasurements("AREA", "LENGTH", "CIRCULARITY", "SOLIDITY", "MAX_DIAMETER", "MIN_DIAMETER", "NUCLEUS_CELL_RATIO")

getProjectEntry().saveImageData(getCurrentImageData())