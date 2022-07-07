
setBatchMode(true);
directory = "C:/Users/Caner/Documents/OneDrive/OneDrive - Universität Basel/PhD_Project_220202/QuPath/stroma/GT/";

imageSuffix = "-labelled.png"
deleteCount = 0;
renameCount=0
processFolder(directory)

//search subfolders and fetch images
function processFolder(input) {
    list = getFileList(input);
    for (i = 0; i < list.length; i++) {
        /*
        if(File.isDirectory(input + list[i]))
            processFolder("" + input + list[i]);
        */
        if(endsWith(list[i], imageSuffix))
            deleteZeroMask(input + list[i]);
    }
}



// function to delete extra files
function deleteZeroMask(mask) { 
    //print("Processing: " + mask);
    tile= substring(mask,0,mask.length()-13);
    tile=tile+".png"; 
    open(mask);
    getStatistics(min, max); 

    close();

    if (max == 0){File.delete(mask);File.delete(tile);deleteCount++;
        print(File.getNameWithoutExtension(tile));
}else if (min == 0 ) {File.rename(mask, mask+'with0');File.rename(tile, tile+'with0');renameCount++;
}


    /*
    print("Deleting: " + File.getNameWithoutExtension(tile));
     	}else {
     		print("Keeping: "+File.getNameWithoutExtension(tile));
     	}
    */
}

print("Done. Total deleted= "+deleteCount+" with0="+renameCount);