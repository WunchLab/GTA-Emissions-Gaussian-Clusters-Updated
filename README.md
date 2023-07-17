## README
author: Juliette Lavoie  
last modified: August 16th, 2019

#What is this repository for?

This repository creates the Gaussian Cluster map. It takes in the raw data from the airmar and lgr and finds the peaks in methane concentration. Then, it uses a Gaussian Mixture algorithm to do a cluster analysis on the position on the peaks. The final html file shows the different clusters, their potential source and known facilities that emit methane.

# Dependencies

I would recommend creating a virtual environment to install all the dependencies and run the notebook.

Packages to install:
folium, sklearn, pandas, numpy, matplotlib, scipy

# Content

-AddDataToGaussian.ipynb : Most important part. Adjust files, find peaks and run gaussian mixture. See instructions below.

-helperFunctions.py: functions that I used a lot. They are in a different file to make the notebook more readable.

-SURF_peakfinder_program_Emily_Ju.py : Functions used to find the peaks. This is Emily Knuckey's file (2018) modified by Juliette Lavoie(2019).

-facilities_lim_C.csv : csv file of known facilities

-dataBikeOldDelayed: Directory of original bike files. 

-dataBike: Directory of bike files adjusted for the time delay.

-dataTruck: Directory of original truck files. ( use directly, no need to adjust for delay)

-areaPeaks: directory csv file of peaks found by SURF_peakfinder_program_Emily_Ju.py 

# To do first

++Get data from dataverse (https://dataverse.scholarsportal.info/dataverse/wunchlab):

-Put bike data in dataBikeOldDelayed with format sync_data_YYYY-MM-DD.csv  
We need latitude, longitude, UTC time, CH_4 concentration in ppm, wind speed and wind direction.  

-Put truck data in dataTruck with format sync_data_YYYY-MM-DD_Truck.csv  

-The 2 other directories will be filled by the notebook




#AddDataToGaussian.ipynb 

There is 2 ways to use AddDataToGaussianCluster.ipynb:  

1. First time built. Take all data, make adjusted, find peaks and build map. Use when dataBike and areaPeaks directories are empty.  

2. Adding one or a small number of days to the already built map. Use to be quick when most of the dates have already been through 1. dataBike and areaPeaks directory have most data and you just want to add one file.  

For both ways, you only need to modify the 4th cell of the notebook. Then, Kernel--Restart & Run All.  

1. 
filesToBeAdjusted= [f  for f in listdir(delayedPath) if isfile(join(delayedPath, f)) and f[-3:]=="csv" ] # every day  
files = [f  for f in listdir(mypath) if isfile(join(mypath, f)) and f[-3:]=="csv" ]  

2.
filesToBeAdjusted=['sync_data_2019-07-31.csv']  
files =['sync_data_2019-07-31.csv']  


You can also change the type of transport( bike or truck).
All new bike data must adjustDelay to account for the 30 seconds delay of the instrument.

The last cell is to be used after having added a lot of new data. If you want the best gaussian cluster, run the last section to optimize the parameters and then change the parameter in the code above.




