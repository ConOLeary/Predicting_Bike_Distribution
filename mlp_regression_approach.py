#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv, sys
import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np; np.set_printoptions(threshold=sys.maxsize)
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

DUD_VALUE= 0
TOTAL_ROWS= 2228278 # This number is greater than the total amount of rows circa epoch change to 27th, but it still works
INPUT_ROWS_LIMIT= TOTAL_ROWS
FILENAME= 'dublinbikes_2020_Q1.csv'
MAX_STATION_ID= 117
SECS_IN_5MIN= 300
DATAPOINTS_PER_DAY= 288
DAYS_OF_WEEK= ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'] # yes, I consider Monday to be the '0'/start of the week
STARTING_DAY= 0 # aka Monday. Because the 27th of Jan 2020 is a Monday
MISSING_STATIONS= [117, 116, 70, 60, 46, 35, 20, 14, 1]
TOTAL_DAYS= 66 # from 27 / 1 / 2020 to (and including) 1 / 4 / 2020
HOURS= 24
EPOCH= datetime.datetime(2020, 1, 27, 0, 0)
MAX_TIME= int((datetime.datetime(2020,4,2,0,0) - EPOCH).total_seconds() / SECS_IN_5MIN)
DECREMENTS= 10
K= 5

class DataDay: # ideally this would be nested in the Station class
    def __init__(self, index):
        self.index= index
        self.times_populated= 0
        self.day_of_week= DUD_VALUE
        self.day_since_epoch= DUD_VALUE
        self.day_of_week= ((STARTING_DAY + index - 1) % len(DAYS_OF_WEEK))
        
        self.daily_epoch_time= np.full(DATAPOINTS_PER_DAY, DUD_VALUE, dtype=np.int)
        self.epoch_time= np.full(DATAPOINTS_PER_DAY, DUD_VALUE, dtype=np.int)
        self.bikes= np.full(DATAPOINTS_PER_DAY, DUD_VALUE, dtype=np.int)
        self.percent_bikes= np.full(DATAPOINTS_PER_DAY, DUD_VALUE, dtype=np.float)

    def populate(self, daily_epoch_time, epoch_time, bikes, percent_bikes):
        self.daily_epoch_time[daily_epoch_time]= daily_epoch_time
        self.epoch_time[daily_epoch_time]= epoch_time
        self.bikes[daily_epoch_time]= bikes
        self.percent_bikes[daily_epoch_time]= percent_bikes
        self.times_populated+= 1

class Station:
    def __init__(self, index):
        self.index= index
        self.name= DUD_VALUE
        self.bike_capacity= DUD_VALUE
        self.address= DUD_VALUE
        self.latitude= DUD_VALUE
        self.longitude= DUD_VALUE
        self.data_days= [DataDay(i) for i in range(1, TOTAL_DAYS + 2)] # + 1 because 0 is excluded and + 1 because we want to include the last day, not have it as the rim
    
    def populate_consts(self, name, bike_capacity, address, latitude, longitude):
        self.name= name
        self.bike_capacity= bike_capacity
        self.address= address
        self.latitude= latitude
        self.longitude= longitude

def get_station_id(name):
    try:
        index= [x.name for x in stations].index(name)
    except ValueError:
        index= -1
    return index


# In[2]:


total_capacity= 0 # not in use currently
index= []; daily_epoch_time= []; epoch_time= []; percent_bikes= [];
stations= [Station(i) for i in range(1, MAX_STATION_ID + 1)] # note: MAX_STATION_ID + 1 is not included in the range
indices_to_populate= list(range(1, MAX_STATION_ID + 1))
for index in MISSING_STATIONS:
    indices_to_populate.remove(index)

with open(FILENAME, newline='') as f:
    reader = csv.reader(f); next(reader) # skip data header
    current_index= 0
    try:
        while len(indices_to_populate) != 0:
            row= next(reader)
            if int(row[0]) == current_index: # this 'if' is just for performance
                continue
            current_index= int(row[0])
            if current_index in indices_to_populate:
                stations[current_index - 2].populate_consts(row[3], row[4], row[8], row[9], row[10])
                indices_to_populate.remove(current_index)
                total_capacity+= int(row[4])
        
        f.seek(0)
        reader= csv.reader(f); row= next(reader) # skip data header
        for row_i, row in enumerate(reader):
            if row_i >= INPUT_ROWS_LIMIT:
                break
            if int((datetime.datetime(int(row[1][0:4]), int(row[1][5:7]), int(row[1][8:10]), int(row[1][11: 13]), int(row[1][14: 16])) - EPOCH).total_seconds()) < 0:
                continue
            try:
                epoch_time= int((datetime.datetime(int(row[1][0:4]), int(row[1][5:7]), int(row[1][8:10]), int(row[1][11: 13]), int(row[1][14: 16])) - EPOCH).total_seconds() / SECS_IN_5MIN)
                stations[int(row[0]) - 1].data_days[int(epoch_time / (DATAPOINTS_PER_DAY - 1))].populate(                     int((datetime.datetime(int(row[1][0:4]), int(row[1][5:7]), int(row[1][8:10]), int(row[1][11: 13]), int(row[1][14: 16])) - datetime.datetime(int(row[1][0:4]), int(row[1][5:7]), int(row[1][8:10]), 0, 0)).total_seconds() / (SECS_IN_5MIN + 1)),                     epoch_time,                     int(row[6]),                     float("{:.3f}".format(float(row[6]) / float(row[4]))))
            except IndexError as e:
                print("\nTRIED: ", epoch_time, ' / ', DATAPOINTS_PER_DAY, ' = ', int(epoch_time / (DATAPOINTS_PER_DAY - 1)))
                print(row[1])
    except csv.Error as e:
        sys.exit('file {}, line {}: {}'.format(filename, reader.line_num, e))


# In[3]:


# counter= 0
# limit= 50
# for row_i in range(limit):
#     print("################# row_i: ",row_i)
#     for s_i, station in enumerate(stations):
        
#         print(stations[s_i].data_days[row_i].day_of_week)


# In[4]:


# Approach 01 - Data Prep

fullness= np.full((MAX_TIME, MAX_STATION_ID - len(MISSING_STATIONS)), 0, dtype=np.int)
fullness_in10= np.full((MAX_TIME, MAX_STATION_ID - len(MISSING_STATIONS)), 0, dtype=np.int)
fullness_in30= np.full((MAX_TIME, MAX_STATION_ID - len(MISSING_STATIONS)), 0, dtype=np.int)
fullness_in60= np.full((MAX_TIME, MAX_STATION_ID - len(MISSING_STATIONS)), 0, dtype=np.int)
fullness_percent= np.full((MAX_TIME, MAX_STATION_ID - len(MISSING_STATIONS)), 0, dtype=np.float)
bikes_changes_past5= np.full((MAX_TIME, MAX_STATION_ID - len(MISSING_STATIONS)), 0, dtype=np.int)
bikes_changes_past15= np.full((MAX_TIME, MAX_STATION_ID - len(MISSING_STATIONS)), 0, dtype=np.int)
bikes_changes_past45= np.full((MAX_TIME, MAX_STATION_ID - len(MISSING_STATIONS)), 0, dtype=np.int)
day_of_week= np.full((MAX_TIME, len(DAYS_OF_WEEK)), 0, dtype=np.int)
hour_of_day= np.full((MAX_TIME, HOURS), 0, dtype=np.float)

station_index_decrement= 1 # this is a varying offset for the indexing of stations that accounts for missing stations that are being ignored
for epoch_day_i in range(TOTAL_DAYS):
    #print("########### epoch_day_i: ", epoch_day_i)
    x_offset= epoch_day_i * DATAPOINTS_PER_DAY
    y_offset= 0
    
    block= np.zeros((DATAPOINTS_PER_DAY, HOURS), dtype=np.float)
    daily_epoch_time= list(range(DATAPOINTS_PER_DAY))
    for time_i in daily_epoch_time:
        hour= float("{:.3f}".format(time_i / 12))
        block[time_i][(int(hour) + 1) % 24]= hour % 1
        block[time_i][int(hour)]= 1 - (hour % 1)
    hour_of_day[x_offset:x_offset + block.shape[0], y_offset:y_offset + block.shape[1]]= block
    
    day= stations[2].data_days[epoch_day_i].day_of_week
    block= np.zeros((DATAPOINTS_PER_DAY, len(DAYS_OF_WEEK)), dtype=np.int)
    for block_i, sub_arr in enumerate(block):
        block[block_i][day]= 1
    day_of_week[x_offset:x_offset + block.shape[0], y_offset:y_offset + block.shape[1]]= block
    
    for station in stations:
        #print("###### station.index: ", station.index)
        if station.index == 1:
            station_index_decrement= 1
        if station.index in MISSING_STATIONS:
            station_index_decrement+= 1
            continue
        y_offset= station.index - station_index_decrement
        
        block= station.data_days[epoch_day_i].percent_bikes
        block= np.reshape(block, (DATAPOINTS_PER_DAY, 1))
        fullness_percent[x_offset:x_offset + block.shape[0], y_offset:y_offset + block.shape[1]]= block
        
        block= station.data_days[epoch_day_i].bikes
        block= np.reshape(block, (DATAPOINTS_PER_DAY, 1))
        fullness[x_offset:x_offset + block.shape[0], y_offset:y_offset + block.shape[1]]= block
        
        bikes= station.data_days[epoch_day_i].bikes
        block= np.reshape(bikes[2:], (bikes.shape[0] - 2, 1))
        fullness_in10[x_offset:x_offset + block.shape[0], y_offset:y_offset + block.shape[1]]= block
        block= np.reshape(bikes[6:], (bikes.shape[0] - 6, 1))
        fullness_in30[x_offset:x_offset + block.shape[0], y_offset:y_offset + block.shape[1]]= block
        block= np.reshape(bikes[12:], (bikes.shape[0] - 12, 1))
        fullness_in60[x_offset:x_offset + block.shape[0], y_offset:y_offset + block.shape[1]]= block
        
        block= station.data_days[epoch_day_i].bikes
        block_5minchange= np.zeros(DATAPOINTS_PER_DAY, dtype=np.int)
        block_15minchange= np.zeros(DATAPOINTS_PER_DAY, dtype=np.int)
        block_45minchange= np.zeros(DATAPOINTS_PER_DAY, dtype=np.int)
        for block_i in range(len(block)):
            five_ago= block[block_i]; fifteen_ago= block[block_i]; fourtyfive_ago= block[block_i];
            for decrement in reversed(range(DECREMENTS)): # i.e. [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
                try:
                    if block[block_i - min(1, decrement)] == DUD_VALUE or block[block_i - min(3, decrement)] == DUD_VALUE or block[block_i - min(9, decrement)] == DUD_VALUE:
                        continue
                    fourtyfive_ago= block[block_i - min(9, decrement)]
                    fifteen_ago= block[block_i - min(3, decrement)]
                    five_ago= block[block_i - min(1, decrement)]
                    break
                except IndexError:
                    print("IndexError with ", decrement)
                    continue
            current= block[block_i]
            decrement= 1
            while current == DUD_VALUE and decrement < 10:
                current= block[block_i - decrement]
                decrement+= 1
            block_5minchange[block_i]= current - five_ago
            block_15minchange[block_i]= current - fifteen_ago
            block_45minchange[block_i]= current - fourtyfive_ago
        block= np.reshape(block, (DATAPOINTS_PER_DAY, 1))
        block_5minchange= np.reshape(block_5minchange, (DATAPOINTS_PER_DAY, 1))
        block_15minchange= np.reshape(block_15minchange, (DATAPOINTS_PER_DAY, 1))
        block_45minchange= np.reshape(block_45minchange, (DATAPOINTS_PER_DAY, 1))
        bikes_changes_past5[x_offset:x_offset + block.shape[0], y_offset:y_offset + block.shape[1]]= block_5minchange
        bikes_changes_past15[x_offset:x_offset + block.shape[0], y_offset:y_offset + block.shape[1]]= block_15minchange
        bikes_changes_past45[x_offset:x_offset + block.shape[0], y_offset:y_offset + block.shape[1]]= block_45minchange


# In[5]:


# print(fullness_percent.shape)
# print(fullness_in10.shape)
# print(bikes_changes_past5.shape)
# print(bikes_changes_past15.shape)
# print(bikes_changes_past45.shape)
# print(day_of_week.shape)
# print(hour_of_day.shape)

# counter= 0
# limit= 22051
#for row_i in range(limit):
#     print("\n####################################")
#     print(day_of_week[row_i])
#     print(hour_of_day[row_i])
#     print(fullness_percent[row_i])
#     print(fullness[row_i])
#     print("---")
#     print(fullness_in10[row_i])
#     print(fullness_in30[row_i])
#     print(fullness_in60[row_i])
#     print(bikes_changes_past5[row_i])
#     print(bikes_changes_past15[row_i])
#     print(bikes_changes_past45[row_i])
#     print("####################################")


# In[6]:


X= np.full((MAX_TIME, hour_of_day.shape[1] + day_of_week.shape[1] + bikes_changes_past5.shape[1] * 4), 0, dtype=np.int)
y= np.full((MAX_TIME, 6), 0, dtype=np.int)

y[0:MAX_TIME, 0:1]= np.reshape(fullness_in10[:,get_station_id("PORTOBELLO ROAD")], (MAX_TIME, 1))
y[0:MAX_TIME, 1:2]= np.reshape(fullness_in10[:,get_station_id("CUSTOM HOUSE QUAY")], (MAX_TIME, 1))
y[0:MAX_TIME, 2:3]= np.reshape(fullness_in30[:,get_station_id("PORTOBELLO ROAD")], (MAX_TIME, 1))
y[0:MAX_TIME, 3:4]= np.reshape(fullness_in30[:,get_station_id("CUSTOM HOUSE QUAY")], (MAX_TIME, 1))
y[0:MAX_TIME, 4:5]= np.reshape(fullness_in60[:,get_station_id("PORTOBELLO ROAD")], (MAX_TIME, 1))
y[0:MAX_TIME, 5:6]= np.reshape(fullness_in60[:,get_station_id("CUSTOM HOUSE QUAY")], (MAX_TIME, 1))
X[0:MAX_TIME, 0:7]= day_of_week
X[0:MAX_TIME, 7:31]= hour_of_day
X[0:MAX_TIME, 31:139]= fullness_percent
X[0:MAX_TIME, 139:247]= fullness_in10
X[0:MAX_TIME, 247:355]= fullness_in30
X[0:MAX_TIME, 355:463]= fullness_in60

# X_train, X_test, y_train, y_test= train_test_split(X, y, random_state=1)
# regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
# y_pred= regr.predict(X_test)

# print(regr.score(X_test, y_test))


# In[8]:


kf= KFold(n_splits= K)
kf.get_n_splits(X)
score_sum= 0.0
i= 1
for train_index, test_index in kf.split(X):
    X_train, X_test= X[train_index], X[test_index]
    y_train, y_test= y[train_index], y[test_index]
    regr= MLPRegressor(random_state= 1, max_iter= 500).fit(X_train, y_train)
    y_pred= regr.predict(X_test)
    score_sum+= regr.score(X_test, y_test)
    print("Data split ", i, " accuracy: ", regr.score(X_test, y_test) * 100, " %")
    i+= 1

print("\nAverage accuracy of model: ", (score_sum / K) * 100, " %")

