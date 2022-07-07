def modelName = "77_2"
resolveHierarchy()
selectObjectsByClassification("analyse");
runObjectClassifier(modelName);


def imageName = getProjectEntry().getImageName()
def folder = 'validation'
def exportFolder = buildFilePath(PROJECT_BASE_DIR, folder, modelName)
def path = buildFilePath(exportFolder, imageName+'.txt')

def testFolder = new File(exportFolder)
if(!testFolder.exists()){
testFolder.mkdir()
}
saveDetectionMeasurements(path)
println "Done " + imageName + " " + modelName