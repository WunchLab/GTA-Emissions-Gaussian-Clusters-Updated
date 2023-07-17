#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 16:36:38 2018
@author: emilyknuckey

Modified June 20, 2019 by Juliette Lavoie
"""
###imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import time
import csv
import math
import scipy
from itertools import groupby
from scipy.ndimage import gaussian_filter1d
from collections import OrderedDict
import string



####helper functions
def format_date_axis(ax, time):
    import matplotlib.dates as mdates
    
    minutes = mdates.MinuteLocator(interval=30)   # every minute
    yearsFmt = mdates.DateFormatter('%H:%M')
    
    # format the ticks
    ax.xaxis.set_major_locator(minutes)
    ax.xaxis.set_major_formatter(yearsFmt)
    #ax.xaxis.set_minor_locator(minutes)
    
    # round to nearest years...
    datemin = np.datetime64(time.min(), 'm')
    datemax = np.datetime64(time.max(), 'm') + np.timedelta64(1, 'm')
    ax.set_xlim(datemin, datemax)
    


# Convert the gps_time strings into real date&times
def convert_to_time(x):
    return datetime.datetime.strptime(x[:19],"%Y-%m-%d %H:%M:%S")#.%f")



def encode(l):
    return [(len(list(group)),name) for name, group in groupby(l)]



def classify_peak(peak, sm, med, lar):
    """
    (float, int, int, int) -> (str, int, int, int)
    Taking the float peak and determining whether it is small, medium, or large
    and then adding one to the count identifier of either sm, med, or lar.
    """  
    if peak > 1:
        size="Large"
        lar += 1
    elif peak > 0.4:
        size="Medium"
        med += 1
    elif peak > 0.04:
        size="Small"
        sm += 1
    elif peak < 0.04:
        size="Negilgable"
        
    return size, sm, med, lar



def checkbefore(time_data, index):
    """
    (array, int) -> bool
    Checking that the twenty values in time_data before the index exists. 
    Return True if exist. Return False otherwise.
    This is to ensure that we're not including peaks where there is a gap
    in the data.
    Precondition:
    date of form '2017-07-26 01:13:38.500' for example
    """
    t = time_data[index][11:]
    x = time.strptime(t.split('.')[0],'%H:%M:%S')
    sec = datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds()
    i = 1
    while i in range(21):
        #converting the time we want to compare
        t2 = time_data[index - i][11:]
        x2 = time.strptime(t2.split('.')[0],'%H:%M:%S')
        sec2 = datetime.timedelta(hours=x2.tm_hour,minutes=x2.tm_min,seconds=x2.tm_sec).total_seconds()
        if sec - sec2 > 60:
            #more than one minute of data not taken
            return False                
        sec -= 1
        i += 1
    return True



def checkafter(time_data, st_ind, end_ind):
    """
    (array, int, int) -> bool
    Checking that values in time_data from st_ind to 20 values after end_ind 
    index exists. 
    Return True if exist. Return False otherwise.
    This is to ensure that we're not including peaks where there is a gap
    in the data.
    Precondition:
    date of form '2017-07-26 01:13:38.500' for example
    """
    t = time_data[st_ind][11:]
    x = time.strptime(t.split('.')[0],'%H:%M:%S')
    sec = datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds()
    peak_len = end_ind - st_ind
    i = 1
    while i in range(peak_len + 20):
        #converting the time we want to compare
        t2 = time_data[st_ind + i][11:]
        x2 = time.strptime(t2.split('.')[0],'%H:%M:%S')
        sec2 = datetime.timedelta(hours=x2.tm_hour,minutes=x2.tm_min,seconds=x2.tm_sec).total_seconds()
        if sec2 - sec > 60: 
            #more than one minute of data not taken
            return False                
        sec += 1
        i += 1
    return True



def checkpeak(time_data, st_ind, end_ind):
    """
    (array, int, int) -> bool
    Checking if before, in between, and after peak data has gaps.
    Return True if there are no gaps.
    Return False if there are gaps.
    """
    
    if not checkafter(time_data, st_ind, end_ind) or not checkbefore(time_data, st_ind):
        return False
    else:
        return True



def distance_between(lat1, lon1, lat2, lon2):
    """
    (float, float, float, float) -> float
    Findng the distance in metres between the location given by lat1, lon1
    and the location given by lat2, lon2.
    """
    #using haversine formula from 'https://www.movable-type.co.uk/scripts/latlong.html'
    r =  6371e3 #earth's radius (m)
    lat1, lat2, lon1, lon2 = map(np.radians, [lat1, lat2, lon1, lon2])
    ch_lat = lat2-lat1 
    ch_lon = lon2-lon1
    
    a = (np.sin(ch_lat/2)**2) + (np.cos(lat1) * np.cos(lat2) * np.sin(ch_lon/2)**2)
        
    c = 2 * math.atan2(np.sqrt(a), np.sqrt(1-a)) #the angular distance in radians
    
    d = r * c
    
    return d



def sml_dis(file):
    """
    (file) -> ([float, str, str],[float, str, str])
    Look through a csv file containing rows [loc, discription, lat, lon] 
    for all of our recorded possible emitters and find the two smallest distances
    between locations.
    Precondtion:
    Lat and Lon are assumed to be zero if there is no location known.
    """
    f = open(file, 'r', encoding="utf-8")
    f.readline() #reading the first line with titles
    
    loc, lat, lon = [], [], []
    smol = 10000000 #big number placeholder
    loc1a, loc1b, loc2a, loc2b = '', '', '', ''
    
    for line in f:
        var = line.split(',')
        loc.append(var[0])
        lat.append(float(var[2]))
        lon.append(float(var[3]))
    
    i = 0
    while i in range(len(lat)):
        j = i
        while j in range(len(lat)):
            dlat = abs(lat[i] - lat[j])
            dlon = abs(lon[i] - lon[j])
            if not (dlat < 0.00002) or not (dlon < 0.00002): #making sure not the same loc twice
                d = distance_between(lat[i], lon[i], lat[j], lon[j])
                if d < smol:
                    smol2, loc2a, loc2b = smol, loc1a, loc1b
                    smol, loc1a, loc1b = d, loc[i], loc[j]
            j += 1
        i += 1
    
    return ([smol, loc1a, loc1b], [smol2, loc2a, loc2b])

 

       
def find_peak_loc(ch4, lat, lon):
    """
    Takes the array of raw data of the gas, ch4, and finds the index of the
    maximum. This index is then used to return the lat and lon position, 
    respectively, at this index.
    ch4: array of gas data
    lat: array of latitude data
    lon: array of longitude data
    """
    #finding index of the max value for ch4
    max_index = np.argmax(ch4)
    
    #using the found index to find the lat and lon for that time
    loc_lat = lat[max_index]
    loc_lon = lon[max_index]
    
    return loc_lat, loc_lon 

def adjustBackground(background,window=1000,limit=0.006):
    """
    calculate std of background in a rolling window
    INPUT: 
    background: array of 1st calculation of background
    window: size of rolling window
    limit: limit of std. above limit, backgorund must be corrected
    OUTPUT
    std:list of std found by rolling
    limit: same as input
    index:list of index of where to change, where std is higher than limit
    newBg: list of corrected background
    """
    std=[]
    for i in range(len(background)):
        halfWin=int(window/2)
        if np.isnan(background[i]):
            std.append(np.nan)
        elif i<window:
            std.append(np.std(background[:i+halfWin]))
        elif i>(len(background)-window):
            std.append(np.std(background[i-halfWin:]))
        else:
            std.append(np.std(background[i-halfWin:i+halfWin]))
    #index of where to change
    index=[i for i in range(len(std)) if std[i]>limit]

    #new background, extrapolate for index
    newBg=background.copy()
    start=[index[i] for i in range(len(index))if i==0 or index[i]-index[i-1]>1 ]
    stop=[index[i] for i in range(len(index))if i==len(index)-1 or index[i+1]-index[i]>1 ]
    
    for a,o in zip(start,stop):
        #handle edge case
        if a==0 or np.isnan(newBg[a-1]):
            valStop=newBg[o+1]
            valStart=valStop
        elif o==len(newBg)-1 or np.isnan(newBg[o+1]):
            valStart=newBg[a-1]
            valStop=valStart
        else:
            valStart=newBg[a-1]
            valStop=newBg[o+1]
            
        diff=valStop-valStart
        length=o-a +2
        for i in range(1,length):
            newBg[a+i-1]=valStart+(i*(diff/length))
    return std,limit,index,newBg

####main functions####
class data_analy():
    
    
    def filter_data(file,filter_len,threshold, date, show=True,savePdf=True, transport='bike'):
        """
        (file, int, int, str, bool) -> 
                            (datetime64[ns], float64, float64, float64) 
        Take LGR data file containing raw data and filter the background. Show
        a graph of the original data, the background, and filtered data over time.
        file: name of LGR data file
        filter_len: length of the filter window
        threshold: float that is 0 <= x <= 1
        """
        
        df = pd.read_csv(file,skipinitialspace=True)
        timeStr=df["gps_time"]
        df["gps_time"] = df["gps_time"].apply(convert_to_time)
        for k in df.keys():
            if k=="ch4d":
                nameCh4='ch4d'
            elif k =="ch4_d":
                nameCh4="ch4_d"
            elif k ==" ch4_d":
                nameCh4=" ch4_d"

        #first estimation of background with small window( initial code by emily )
        time, data, pressure = df["gps_time"], df[nameCh4], df["pressure"]
        background = data.rolling(window=filter_len,center=True).quantile(threshold)
        anomaly = data-background
        #fix background
        if transport=='bike':
            window=500#1000
            limit=0.006#0.006
        elif transport=='truck':
            window=400
            limit=0.008
        #correction of background (Juliette)
        std,limit,index,newBg=adjustBackground(background,window=window,limit=limit)
        newAnomaly=data-newBg
        time2change=[time[i] for i in index]
        bg2change=[background[i] for i in index]
        
        if show==True:
            ##ploting graphs
            fig, [ax1,ax2,ax3,ax4] = plt.subplots(4,1,figsize=(8,10),sharex=True)
            #plt.suptitle("Background on {}".format(date))
            
            
            #raw data
            ax1.plot(time,data, '.', ms=2,color='#979A9A')
            format_date_axis(ax1, time)
            n = 5  # Keeps every 5th label
            [l.set_visible(False) for (i,l) in enumerate(ax1.xaxis.get_ticklabels()) if i % n != 0]
            #ax1.set_xlabel("Time", fontsize=20)
            ax1.set_ylabel("Raw $CH_4$\n (ppm)", fontsize=16)
            ax1.tick_params(axis='y', which='major', labelsize=12)
            ax1.text(0.01, 0.80, string.ascii_uppercase[0], transform=ax1.transAxes, size=20, weight='bold')	# A

            #raw data with background
            ax2.plot(time,data,'.',ms=2,color='#979A9A',label="$CH_4$ data")
            ax2.plot(time,background,'.',ms=2,color='green',label="Background")
            ax2.plot(time2change,bg2change,'.',ms=2,color='red',label="To change")
            ax2.set_ylabel("1st bckgd\n estimate (ppm)", fontsize=16)
            ax2.tick_params(axis='y', which='major', labelsize=12)
            ax2.legend(markerscale=6)
            ax2.text(0.01, 0.80, string.ascii_uppercase[1], transform=ax2.transAxes, size=20, weight='bold')	# B

            #std of background
            ax3.plot(time,std, '.', ms=2)
            ax3.axhline(limit,linestyle=':', color='red',label="limit")
            ax3.set_ylabel("Bckgd STD\n (ppm)", fontsize=16)
            ax3.tick_params(axis='y', which='major', labelsize=12)
            ax3.legend()  
            ax3.text(0.01, 0.80, string.ascii_uppercase[2], transform=ax3.transAxes, size=20, weight='bold')	# C
            
            #newbackground
            ax4.plot(time,data,'.',ms=2,color='#979A9A',label="$CH_4$ data")
            ax4.plot(time,newBg, '.', ms=2,color='green',label="Adjusted background")
            ax4.set_ylabel("Adjusted bckgd\n (ppm)", fontsize=16)
            #x4.set_ylim((1.8,3))
            ax4.tick_params(axis='y', which='major', labelsize=12)
            ax4.legend(markerscale=6)
            ax4.text(0.01, 0.80, string.ascii_uppercase[3], transform=ax4.transAxes, size=20, weight='bold')	# D
            
            format_date_axis(ax4, time)
            [l.set_visible(False) for (i,l) in enumerate(ax4.xaxis.get_ticklabels()) if i % n != 0]
            ax4.tick_params(axis='both', which='major', labelsize=15)
            ax4.set_xlabel("Time", fontsize=20)
            
            if savePdf:
                name = "peaks/{}_filtlen{}_th{}.filtered".format(date,filter_len, threshold)
                plt.savefig(name + ".png")
    
        return (time, data, newBg, newAnomaly, background)

   

    def show_peaks(file, filter_len, threshold, date, mode=1, show=True,savePdf=True,lim=[0.04,0.2,1]):
        """
        (file, int, int, str, bool) -> 
                            (datetime64[ns], float64, float64, float64, array)
        Identifying peaks in the data from the LGR data file. Show graph of the
        identified peaks and and print their size, peak, latitude, longitude, 
        and time taken if show is True.
        file: name of LGR data file
        filter_len: length of the filter window
        threshold: float that is 0 <= x <= 1
        """
        
        df = pd.read_csv(file,skipinitialspace=True)
        
        time_data, data, background, anomaly,_ = data_analy.filter_data(file, filter_len, threshold, date, show=False)
        
        above=(anomaly > mode*np.std(background))
        #print(np.std(background))
        below=~above
        ls = np.array(encode(above)) #group the peaks
        
        ls[:,0] = ls[:,0].cumsum().astype(int) #find the indices
        # construct the blocks as a list of [start,stop,type]
        blocks = np.vstack([np.hstack([0,ls[:-1,0]]),
                            np.hstack([ls[:-1,0],above.size]),
                            ls[:,1]]).astype(int)
        
        l,h = [0,0],[anomaly.max(),anomaly.max()]
        
        if show==True:
            plt.figure(figsize=(12,5))
            plt.title("{}: Filter Len = {}, Threshold = {}".format(date, filter_len, threshold))
          
        width_threshold = 15
        #counting how many of each peaks
        sm, med, lar = 0, 0, 0
        for block in blocks.T:
            if block[2]==0 or block[1]-block[0] < width_threshold:
                continue #skip non-anomalies
            #checking for gaps before and after block 
            if not checkpeak(df['gps_time'], block[0], block[1]):
                continue #ignoring a peak if the data before before or after is missing
            tloc = np.array([block[0],block[1]])
            t = time_data[tloc].values
            if show==True:
                plt.fill_between(t,l,h,color='black',alpha=0.1)
            peak=anomaly[block[0]:block[1]].mean()
            #next line identifies the size of peak and adds 1 to the count of its type
            #size, sm, med, lar = classify_peak(peak, sm, med, lar)              
            lon, lat, ti = df["lon"].iloc[block[0]:block[1]].mean(), df["lat"].iloc[block[0]:block[1]].mean(),df["gps_time"].iloc[block[0]:block[1]].min()
            #if show==True:
                #print("{} peak of {:.2f} ppm at {:.6f}E {:.6f}N at time {}".format(size,peak,lon,lat,ti))
            
            #color peak
            if show==True:
                peakM=max(anomaly[block[0]:block[1]])
                r=4 # if no r
                if peakM > lim[2]:
                    r=3
                elif peakM > lim[1]:
                    r=2
                elif peakM > lim[0]:
                    r=1
                elif peakM < lim[0]:
                    r=0
                colo=['blue','red','green','orange','black']
                lab=["Level 0",'Level 1',"Level 2", "Level 3", "Inconclusive"]
                ran=[x>= block[0] and x< block[1] for x in range(len(time_data))]
                plt.plot(time_data[ran], anomaly[ran],'.',ms=2,color=colo[r], label=lab[r])

        
        if show==True:
            plt.plot(time_data[below], anomaly[below],'.',ms=2,color='grey', label="below "+str(mode)+ " std")
            format_date_axis(plt.gca(), time_data)
            plt.xticks(size=8, rotation=70)
            plt.set_ylabel("Methane anomaly (ppm)")
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            if savePdf:
                plt.savefig("peaks/{}_filtlen{}_th{}.peaks.pdf".format(date,filter_len, threshold))
        
        #print("# small peaks = {}, # medium peaks = {}, # large peaks = {}".format(sm, med, lar))
        return (time_data, data, background, anomaly, blocks)



    def save_peaks(file, filter_len, threshold, date, mode=1,lim=[0.04,0.2,1]):
        """
        (file, int, int, str) -> None
        Write and save a csv file with rows: Date, Enhancement, Latitude,
        Longitude, Level.
        file: name of LGR data file
        filter_len: length of the filter window
        threshold: float that is 0 <= x <= 1
        """
        
        df = pd.read_csv(file,skipinitialspace=True)

        
        time, data, background, anomaly, blocks = data_analy.show_peaks(file, filter_len, threshold, date, mode=mode, show=False)
        
        width_threshold = 15
        list_peaks = []
        
        for block in blocks.T:
            if block[2]==0 or block[1]-block[0] < width_threshold:
                continue #skip non-anomalies
            peak=max(anomaly[block[0]:block[1]])
            indx = np.argmax(anomaly[block[0]:block[1]])
            
            if peak > lim[2]:
                r=3
            elif peak > lim[1]:
                r=2
            elif peak > lim[0]:
                r=1
            elif peak < lim[0]:
                r=0
            lon, lat, ti, wd_corr, ws_corr = df["lon"].iloc[block[0]:block[1]][indx], df["lat"].iloc[block[0]:block[1]][indx],df["gps_time"].iloc[block[0]:block[1]][indx],df["wd_corr"].iloc[block[0]:block[1]][indx],df["ws_corr"].iloc[block[0]:block[1]][indx]
            peak = float("{0:.2f}".format(peak)) #rounding peak for cvs file
            list_data = [ti, peak, lat, lon, r, wd_corr, ws_corr]
            list_peaks.append(list_data)


        
        file_name = "peaks/{}.peaks.csv".format(date)
        with open(file_name, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(["Date", "Enhancement_ppm", "Latitude", "Longitude", "Level", "wd_corr", "ws_corr"])
            writer.writerows(list_peaks)
            csvfile.close()
            #print("File written as:{}".format(file_name))



    def areaPeaks(file, filter_len, threshold, date, mode=1, show=True,limA=[1,4,60], limH=[0.04,0.2,1],transport='bike'):
    	#def areaPeaks(file, filter_len, threshold, date, mode=1, show=True,limA=[4,16,240], limH=[0.04,0.2,1],transport='bike'):
        """
		Use area  and height to find peaks location and level
		INPUT
		file:sync_data_YYYY-MM-DD.csv. lgr and amr data with concentration to extract peak from.
		filter_len:length of the filter window for background
		threshold: threshold for background
		date: date for filename and plot
		mode: number std over to be considered a peak
		show=to plot background and peaks
		limA=limit for level based on area
		limH=limit for level based on height
		transport='bike' or 'truck'
        """
        #print(file)
        df = pd.read_csv(file,skipinitialspace=True)
        time_data, data, background, anomaly, _=data_analy.filter_data(file,filter_len,threshold,
                                                                      date, show=show,savePdf=True, transport=transport)
        longitude, latitude = df["lon"], df["lat"] ####test

        above=(anomaly > mode*np.std(background))
        below=~above
        ls = np.array(encode(above)) #group the peaks
        ls[:,0] = ls[:,0].cumsum().astype(int) #find the indices
        # construct the blocks as a list of [start,stop,type]
        blocks = np.vstack([np.hstack([0,ls[:-1,0]]),np.hstack([ls[:-1,0],above.size]),ls[:,1]]).astype(int)
        l,h = [0,0],[anomaly.max(),anomaly.max()]

        if show:
            plt.figure(figsize=(12,6))
            #plt.title("Peaks on "+ date)

        width_threshold = 15
        list_peaks = []

        for block in blocks.T:
            if block[2]==0 or block[1]-block[0] < width_threshold:
                continue #skip non-anomalies
            #checking for gaps before and after block 
            #if not checkpeak(df['gps_time'], block[0], block[1]):		# Ne pas utiliser checkpeak pour les campagne voiture avec AIO2
                #continue #ignoring a peak if the data before or after is missing
            tloc = np.array([block[0],block[1]])
            t = time_data[tloc].values
            #if show:
            #    plt.fill_between(t,l,h,color='black',alpha=0.1)




            
            #area under the curve of peak
            largeur = 0
            length = []
            for ii in range(block[0],block[1]):
            	length.append(largeur)
            	largeur = largeur + distance_between(latitude[ii], longitude[ii], latitude[ii+1], longitude[ii+1])
            area = abs(scipy.integrate.trapz(anomaly[block[0]:block[1]],x=length))


            #position of peak at max amplitude
            peakM=max(anomaly[block[0]:block[1]])
            #evaluate the level
            if area > limA[2] or peakM>limH[2]:
                r=3
                #print('Large:',area,length[-1])
            elif area > limA[1] or peakM> limH[1]:
                r=2
                #print('Medium:',area,length[-1])
            elif area > limA[0] or peakM>limH[0]:
                r=1
                #print('Small:',area,length[-1])
            elif area < limA[0] or peakM<limH[0]:
                r=0
                #print('Negligible:',area,length[-1])
            if show==True:#color peak
                colo=['blue','yellow','orange','red','black']
                #lab=["Level 0",'Level 1',"Level 2", "Level 3", "Inconclusive"]
                lab=["Negligible",'Small',"Medium", "Large", "Inconclusive"]
                ran=[x>= block[0] and x< block[1] for x in range(len(time_data))]
                plt.plot(time_data[ran], anomaly[ran],'.',ms=2,color=colo[r], label=lab[r])    
            
            indx = anomaly[block[0]:block[1]].idxmax()
            lon, lat, ti, wd_corr, ws_corr = df["lon"].iloc[block[0]:block[1]][indx], df["lat"].iloc[block[0]:block[1]][indx],df["gps_time"].iloc[block[0]:block[1]][indx],df["wd_corr"].iloc[block[0]:block[1]][indx],df["ws_corr"].iloc[block[0]:block[1]][indx]
            peakM = float("{0:.2f}".format(peakM)) #rounding peak for cvs file
            list_data = [ti, peakM, area, lat, lon, r, wd_corr, ws_corr,largeur]
            list_peaks.append(list_data)

        file_name = "areaPeaks/area{}.peaks.csv".format(date)
        with open(file_name, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(["Date", "Enhancement_ppm", "Area_ppm.m", "Latitude", "Longitude", "Level", "wd_corr", "ws_corr","width_m"])
            writer.writerows(list_peaks)
            csvfile.close()
        if show:
            plt.plot(time_data[below], anomaly[below],'.',ms=2,color='grey', label="below "+str(mode)+ " std")
            format_date_axis(plt.gca(), time_data)
            plt.xticks(size=4)#, rotation=70)
            plt.ylabel("Methane anomaly (ppm)", fontsize=20)
            plt.xlabel("Time", fontsize=20)
            plt.tick_params(axis='both', which='major', labelsize=15)
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(),fontsize=14, markerscale=6)

            name = "peaks/{}_filtlen{}_th{}.peaks".format(date,filter_len, threshold)
            plt.savefig(name + ".png")



    def analyze_data(file, date, mode=1,savePeak=True,lim=[0.04,0.4,1],transport='bike'):
        """
        (file, int, int, str) -> None
        Use created functions filter_data, show_peak, and save_peaks at once.
        file: name of LGR data file
        filter_len: length of the filter window
        threshold: float that is 0 <= x <= 1
        mode: defining peaks as 1, 2, or 3 std above the background
        """
        if transport =='bike':
            filter_len=500
            threshold=0.05
        elif transport =='truck':
            filter_len=300
            threshold=0.05
        
        time, data, background, anomaly,_ = data_analy.filter_data(file, filter_len, threshold, date, show=True, savePdf=True,transport=transport)
        
        time, data, background, anomaly, blocks = data_analy.show_peaks(file, filter_len, threshold, date, mode=mode, show=True, savePdf=True,lim=lim)
        
        if savePeak:
            data_analy.save_peaks(file, filter_len, threshold, date, mode=mode,lim=lim)
   
   
   
    def clean_file(file):
        """
        (file) -> file
        Take the csv file created in function data_analy.save_peaks and look
        to see if any of the sites are taken from the same location. If the
        peaks are within dis of one another and taken within one hour then 
        don't write the smaller peak to the new csv file.
        """
        
        f = open(file, 'r', encoding="utf-8")
        f.readline() #reading the first line with titles
        
        data = [] #will be a list of lists
        
        for line in f:
            var = line.split(',')
            if len(var) > 1:
                data.append([var[0], float(var[1]), float(var[2]), float(var[3]), float(var[4]), int(var[5]), str(var[6]), str(var[7]), float(var[8])])
            #this is a list of [date, enh, area, lat, lon, level, wd_corr, ws_corr,width]
           
        for peak in data:
            t = peak[0][11:]
            x = time.strptime(t.split('.')[0],'%H:%M:%S')
            sec = datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds()
            peak.append(sec)
     
        #creating new csv file
        file_name = file[:-4] + "_cleaned.csv"
        with open(file_name, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(["Date", "Enhancement_ppm", "Area_ppm.m", "Latitude", "Longitude", "Level", "wd_corr", "ws_corr","width_m"])
            i = 0
            for peak in data:
                same_emit = [peak]
                enhanc = [peak[1]] #putting the enhancements into another list in order to make it easier to find the max later
                i += 1
                for peak2 in data[i:]:
                    dis = distance_between(peak[3], peak[4], peak2[3], peak2[4])
                    t_diff = peak2[-1] - peak[-1]
                    if dis < 12 and t_diff < 3600: #maybe should increase from 12 to account for locations such as Keeting and Disco
                        #if distance between peaks is less than 12m and taken less
                        #than an hour apart
                        same_emit.append(peak2)
                        enhanc.append(peak2[1])
                for item in same_emit[1:]: #don't want to get rid of the first term or messes up the for loop
                    data.remove(item)
                #finding max enhancement and writing it to the new csv file
                index_max = np.argmax(enhanc)#index of the largest enhancement
                lar_peak = same_emit[index_max]
                #writing row of date, enhancement, latitude, longitude
                writer.writerow(lar_peak[:-1])
            csvfile.close()
            #print("File Written as: {}".format(file_name))
