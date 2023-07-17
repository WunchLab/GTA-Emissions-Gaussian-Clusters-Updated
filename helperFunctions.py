import folium
import pandas as pd
import math
import numpy as np 
import random 
from matplotlib import path
from folium.plugins import FloatImage
import sklearn

def clusterMap(x,label,n,xDates,aldates, showFacilities=True,save=False,name='test',showMeans=True,means=[],covariances=[],alone=[]):
    """
    Create HTML file of peaks colored by cluster.
    INPUT
    x: list of coordinate of peaks 
    label: list of label for peaks in x. each number represents a different cluster
    n: number of cluster
    showFacilities: plot known facilities
    save: save the html file under name
    showMeans: plot mean of each gaussian cluster
    means: list of means to plot
    covariances: list of covariance for each mean ( for information in popup of means)
    alone: list of coordinates of peaks that are isolate and not use in the gaussian mixture ( not in a cluster)
    OUTPUT
    m: folium map
    """
    # map around u of t
    m = folium.Map(location=[43.656997372, -79.390331772],zoom_start=12,tiles='Stamen Toner')
    # get n random color for cluster
    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(n)]
    #plot known sources
    if showFacilities:
        sources=pd.read_csv('facilities_lim_C.csv')
        for i in range(sources.shape[0]):
            folium.Marker([sources["Latitude"][i],sources["Longitude"][i]], popup=sources["TreatmentPlant"][i] + "\nType: " + sources["Type"][i],icon=folium.Icon(icon='ok-sign')).add_to(m)
    #plot gaussian means ( potential sources?)
    if showMeans:
        totcovarea = 0
        for cov,mea in zip(covariances,means):
            #mistake 3
            folium.Marker(location=list(mea[::-1]), popup='Potential source with cov: '+str(cov),
                          icon=folium.Icon(color='red', icon='fa-question',prefix='fa')).add_to(m)
            totcovarea += (abs(cov[0][0]*cov[1][1] - cov[0][1]*cov[1][0]))**(1/2.0)*math.pi*111**2
        print(str(n) + ": " + str(totcovarea) + " km^2")
    #plot peak with color of label
    for i,pair in enumerate(x):
        folium.Circle(radius=100,location=pair[::-1],color=colors[label[i]],fill=True,popup=xDates[i]).add_to(m)
    #plot alone peaks
    if len(alone)>0:
        for i,pair in enumerate(alone):
            for o in range(len(pair)):
                if math.isnan(pair[o]):
                    pair[o] = 0
                #print(pair[o])
            folium.Circle(radius=100,location=pair[::-1],color="blue",fill=False,popup=aldates[i]).add_to(m)
    #save as html
    if save:
        m.save(name+".html")
    return m

def separate(x,xDate, limit=0.008,sepMonth=False,month=[]):
    """ 
    separate the set of points x in alone points and grouped points to put in gaussian mixture
    INPUT
    x:list of coordinates of peaks 
    limit: minimum radius with no neighboors to be considered alone
    sepMonth: if separate per month
    month: month of each peak in x
    OUTPUT
    alone: list of coordinates of alone peaks
    group: list of coordinates of grouped peaks
    am: month of each peak in alone
    xm:month of each peak in group
    """
    alone=[]
    group=[]
    alonedate=[]
    groupdate=[]
    am=[]
    xm=[]

    if not sepMonth:
        month=x
    
    for pair,m,date in zip(x,month,xDate):
        far=True
        for other in x:
            dist = math.sqrt( (pair[0] - other[0])**2 + (pair[1] - other[1])**2 )
            #print(dist)
            if dist < limit and dist!=0:
                far=False
        #print(far)
        if far:
            alone.append(pair)
            alonedate.append(date)
            if sepMonth:
                am.append(m)
        else:
            group.append(pair)
            groupdate.append(date)
            if sepMonth:
                xm.append(m)
    return alone, group,am,xm,groupdate,alonedate


def optimizeMult(n,random_state,allPeaks,verbose=True, listMult=[0,0.00001,0.000025,0.00005,0.000075,0.0001,0.00025,0.0005,0.00075]):
    """
    Find the multiplicative factor for wind correction (mult) that gives the best(highest) silhouette score
    INPUT
    n:number of cluster
    random_state: random number to initialize gaussian mixture
    allPeaks: list of dataframe of each peak files
    verbose: True to print progress
    listMult: list of multiplicative factor to try
    OUTPUT:
    bestSil:best silhouette score
    bestMultS: best mult
     """
    bestSil=-2
    bestMult=0
    for mult in listMult:
        if verbose:
            print("mult",mult)
        #get corrected x
        x=[]
        for peaks in allPeaks:
            for i in range(peaks.shape[0]):
                long, lat = peaks["Longitude"][i],peaks['Latitude'][i]
                wd=math.radians(peaks['wd_corr'][i])
                ws=peaks['ws_corr'][i]
                corrLong= long- (mult*ws*math.cos(wd))
                corrLat= lat- (mult*ws*math.sin(wd))
                if peaks["Level"][i]==2:
                    x.append([corrLong,corrLat])
                if peaks["Level"][i]==3:
                    for count in range(3):
                        x.append([corrLong,corrLat])
        alone,x,_,_=separate(x, limit=0.01)
        if verbose:
            print("alone,x",len(alone),len(x))
        if len(alone)<75:
            #learn cluster
            gm =sklearn.mixture.GaussianMixture(n_components=n, covariance_type='full',random_state=random_state).fit(x)
            label=gm.predict(x)
            
            #silhouette score
            sil=sklearn.metrics.silhouette_score(x, label)
            if sil>bestSil:
                bestSil=sil
                bestMultS=mult
    return bestSil,bestMultS

def bestParam(cleanedFiles,nRan=range(20,28), rRan=range(25),verbose=True):
    """
    Find best parameters for gaussian mixture
    INPUT
    cleanedFiles: List of files of peaks  to use in gaussian mixture
    nRan: List of n to try
    rRan: Range of how many random state to try
    verbose: Print progress
    OUTPUT
    maxMult: best multiplicative factor for wind correction
    maxSil: best silhouette score
    maxRan: best random state
    maxN:best number of cluster 
    """

    maxSil=0
    maxMult=None
    maxRan=None
    maxN=None
    i=0
    allPeaks=[pd.read_csv(f) for f in cleanedFiles]
    #iterate through possibilities of different n and random state
    for n in nRan :
        for r in rRan:
            ran=np.random.randint(0,2**31 - 1)
            #find best mult
            newSil,newMult=optimizeMult(n,ran,allPeaks,verbose=False)
            if newSil>maxSil:
                maxSil=newSil
                maxMult=newMult
                maxRan=ran
                maxN=n

            if verbose:
                print(i," number of cluster:", n," random_state ",ran," mult: ",newMult," silhouette: ",newSil)
            i+=1
    return maxMult, maxSil, maxRan, maxN 

    
def adjustDelay(files,delayedPath='dataBikeOldDelayed' , adjustedPath='dataBike'):
    """
    Adjust Delay ( for bike data)¶
    Modify file 'sync_data_YYYY-MM-DD.csv' from 'dataBikeOldDelayed' directory (from Sébastien or dataverse) to account for the 30 second delay to measure concentration and put adjusted file in 'dataBike' directory
    INPUT:
    files: files to be adjusted
    delayedPath: directory of files to be adjusted
    adjustedPath: directory to put the adjusted file
    """

    for f in files:
        data=pd.read_csv(delayedPath+'/'+f,skipinitialspace=True)
        #get right name for the keys
        for k in data.keys():
                if k=="ch4d":
                    nameCh4='ch4d'
                    nameCO='cod'
                    nameCO2='co2d'
                    nameh2o='h2o'
                elif k =="ch4_d":
                    nameCh4="ch4_d"
                    nameCO='co_d'
                    nameCO2='co2_d'
                    nameh2o='h2o'
        names=[nameCh4,nameCO,nameCO2,nameh2o]
        #put right concentration in front of right time
        for name in names:
            newCol=list(data[name].copy())
            newCol.extend([np.nan]*30)
            newCol=newCol[30:]
            data[name]=newCol
        data=data[:-30]
        #save new file in right directory
        data.to_csv(adjustedPath +'/'+f)


