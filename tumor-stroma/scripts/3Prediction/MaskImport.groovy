// Script written for QuPath v0.2.3
// Minimal working script to import labelled images 
// (from the TileExporter) back into QuPath as annotations.

import qupath.lib.objects.PathObjects
import qupath.lib.regions.ImagePlane
import static qupath.lib.gui.scripting.QPEx.*
import ij.IJ
import ij.process.ColorProcessor
import ij.process.ImageProcessor
import qupath.imagej.processing.RoiLabeling
import qupath.imagej.tools.IJTools
import java.util.regex.Matcher
import java.util.regex.Pattern
import ij.process.ByteProcessor
import ij.plugin.filter.ThresholdToSelection

import ij.measure.Calibration
import ij.plugin.filter.ThresholdToSelection
import ij.process.ByteProcessor
import ij.process.ImageProcessor
import qupath.imagej.tools.IJTools
import qupath.lib.objects.PathAnnotationObject
import qupath.lib.objects.classes.PathClassFactory
import static qupath.lib.gui.scripting.QPEx.*

import javax.imageio.ImageIO
import qupath.lib.regions.ImagePlane
import qupath.lib.roi.ROIs
import qupath.lib.objects.PathObjects

//def px = 512
def ds = 4

//def px_ds = ds+'_'+px

clearAnnotations()
def name = getProjectEntry().getImageName()
def stemName = GeneralTools.getNameWithoutExtension(name)
//def dsstr = name.substring(0,1)
//def directoryPath = "C:/Users/Caner/Downloads/dpx_trial/"+name // TO CHANGE
def predPath= '/scicore/home/pissal00/ercan0000/QuPath/preCometeePOC/preds'


def directoryPath = buildFilePath(predPath, stemName)
print(directoryPath)
File folder = new File(directoryPath);
File[] listOfFiles = folder.listFiles();

listOfFiles.each { file ->
    def path = file.getPath()
    def imp = IJ.openImage(path)



    print "Now processing: " + path

    // Parse filename to understand where the tile was located
    def parsedXY = parseFilename(GeneralTools.getNameWithoutExtension(path))

    double downsample = ds // TO CHANGE (if needed)
    ImagePlane plane = ImagePlane.getDefaultPlane()


    // TO CHANGE ACCORDING TO PIXEL VALUES AND CLASSES
    def classMap = [0:'Background', 1 : 'Tumor', 2: 'Stroma',3  : 'Debris' ]

    classMap.each{
        def ip = imp.getProcessor()
        ip.setThreshold(it.key,it.key, ImageProcessor.NO_LUT_UPDATE)
        def roiIJ = new ThresholdToSelection().convert(ip)
        if (roiIJ != null){
            def roi = IJTools.convertToROI(roiIJ, -parsedXY[0]/downsample, -parsedXY[1]/downsample, downsample, plane)
            def pathObjects = PathObjects.createAnnotationObject(roi, getPathClass(it.value))
            addObjects(pathObjects)
        }
        
    }

}

int[] parseFilename(String filename) {
    def p = Pattern.compile("\\[x=(.+?),y=(.+?),")
    parsedXY = []
    Matcher m = p.matcher(filename)
    if (!m.find())
        throw new IOException("Filename does not contain tile position")

    parsedXY << (m.group(1) as double)
    parsedXY << (m.group(2) as double)

    return parsedXY
}




for (cls in ['Tumor','Stroma', 'Debris','Background']) {
    selectObjectsByClassification(cls)
    mergeSelectedAnnotations()
}



getProjectEntry().saveImageData(getCurrentImageData())

