#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv, sys
import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np; np.set_printoptions(threshold=sys.maxsize)

DUD_VALUE= 123
TOTAL_ROWS= 2228278
INPUT_ROWS_LIMIT= TOTAL_ROWS # TOTAL_ROWS
FILENAME= 'dublinbikes_2020_Q1.csv'
MAX_STATION_ID= 117
SECS_IN_5MIN= 300
DATAPOINTS_PER_DAY= 288
DATAPOINTS_PER_HOUR= 12
DAYS_OF_WEEK= ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'] # yes, I consider Monday to be the '0'/start of the week
STARTING_DAY= 2
MISSING_STATIONS= [117, 116, 70, 60, 46, 35, 20, 14, 1]
TOTAL_DAYS= 91
OPEN_HOURS= 20 # want to change this to 18 i.e. excluding 1, 2, 3, and 4 am
MAX_DISPLAYED_STATIONS= 1
MAX_DISPLAYED_DAYS= 10
MAX_TIME= int((datetime.datetime(2020,4,2,0,0) - datetime.datetime(2020,1,1,0,0)).total_seconds() / SECS_IN_5MIN)
START_TIME= int((datetime.datetime(2020, 1, 1, 4, 0) - datetime.datetime(2020, 1, 1, 0, 0)).total_seconds() / SECS_IN_5MIN) # bike sharing starts at 4am at earliest
MAX_MISSING_STATIONS= 10

class DataDay: # ideally this would be nested in the Station class
    def __init__(self, index):
        self.index= index
        self.times_populated= 0
        self.day_of_week= DUD_VALUE
        self.day_since_epoch= DUD_VALUE
        self.day_of_week= ((STARTING_DAY - 1 + index) % len(DAYS_OF_WEEK))
        
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
        self.data_days= [DataDay(i) for i in range(1, TOTAL_DAYS + 3)] # Admitedly, I don't even know exactly why this has to be +3
    
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


# In[ ]:


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
            try:
                epoch_time= int((datetime.datetime(int(row[1][0:4]), int(row[1][5:7]), int(row[1][8:10]), int(row[1][11: 13]), int(row[1][14: 16])) - datetime.datetime(2020,1,1,0,0)).total_seconds() / (SECS_IN_5MIN + 1)) # the unit of epoch_time is 5-minutes # the + 1 is a means to decrement the results
                stations[int(row[0]) - 1].data_days[int(epoch_time / (DATAPOINTS_PER_DAY - 1))].populate(                     int((datetime.datetime(int(row[1][0:4]), int(row[1][5:7]), int(row[1][8:10]), int(row[1][11: 13]), int(row[1][14: 16])) - datetime.datetime(int(row[1][0:4]), int(row[1][5:7]), int(row[1][8:10]), 0, 0)).total_seconds() / (SECS_IN_5MIN + 1)),                     epoch_time,                     int(row[6]),                     float("{:.3f}".format(float(row[6]) / float(row[4]))))
            except IndexError as e:
                print("\nTRIED: ", epoch_time, ' / ', DATAPOINTS_PER_DAY, ' = ', int(epoch_time / (DATAPOINTS_PER_DAY - 1)))
                print(row[1])
    except csv.Error as e:
        sys.exit('file {}, line {}: {}'.format(filename, reader.line_num, e))


# In[ ]:


# counter= 0
# limit= 50
# for row_i in range(limit):
#     print("################# row_i: ",row_i)
#     for s_i, station in enumerate(stations):
        
#         print(stations[s_i].data_days[row_i].day_of_week)


# In[ ]:


# Approach 01 - Data Prep

time= int((datetime.datetime(2020, 1, 1, 6, 25) - datetime.datetime(2020, 1, 1, 0, 0)).total_seconds() / SECS_IN_5MIN)
station_index_decrement= 1 # this is a varying offset for the indexing of stations that accounts for missing stations that are being ignored
row_vol= MAX_TIME - time - START_TIME * TOTAL_DAYS
fullness= np.full((row_vol, MAX_STATION_ID - len(MISSING_STATIONS)), DUD_VALUE, dtype=np.float)
bikes_changes_past5= np.full((row_vol, MAX_STATION_ID - len(MISSING_STATIONS)), DUD_VALUE, dtype=np.int)
bikes_changes_past15= np.full((row_vol, MAX_STATION_ID - len(MISSING_STATIONS)), DUD_VALUE, dtype=np.int)
bikes_changes_past45= np.full((row_vol, MAX_STATION_ID - len(MISSING_STATIONS)), DUD_VALUE, dtype=np.int)
day_of_week= np.full((row_vol, len(DAYS_OF_WEEK)), DUD_VALUE, dtype=np.int)
hour_of_day= np.full((row_vol, OPEN_HOURS + 1), DUD_VALUE, dtype=np.float) # + 1 because we need the 0th/24th hour to be represented despite its exclusion

rejected_days= 0
for epoch_day_i in range(TOTAL_DAYS):
    print("########### epoch_day_i: ", epoch_day_i)
    x= epoch_day_i * (DATAPOINTS_PER_DAY - START_TIME)
    y= 0
    day= stations[2].data_days[epoch_day_i].day_of_week
    
    block= np.zeros((DATAPOINTS_PER_DAY - START_TIME, OPEN_HOURS + 1), dtype=np.float)
    daily_epoch_time= stations[2].data_days[epoch_day_i].daily_epoch_time[START_TIME:]
    for time_i in daily_epoch_time:
        if time_i != DUD_VALUE:
            hour= float("{:.3f}".format(time_i / 12))
            hour-= START_TIME / DATAPOINTS_PER_HOUR
            block[time_i - START_TIME][int(hour) + 1]= hour % 1
            block[time_i - START_TIME][int(hour)]= 1 - (hour % 1)
    hour_of_day[x:x + block.shape[0], y:y + block.shape[1]]= block
    
    block= np.zeros((DATAPOINTS_PER_DAY - START_TIME, len(DAYS_OF_WEEK)), dtype=np.int)
    for block_i, sub_arr in enumerate(block):
        block[block_i][day]= 1
    day_of_week[x:x + block.shape[0], y:y + block.shape[1]]= block
    
    for station in stations:
        print("###### station.index: ", station.index)
        if station.index == 1:
            station_index_decrement= 1
        if station.index in MISSING_STATIONS:
            #print("Rejected.")
            station_index_decrement+= 1
            continue
        y= station.index - station_index_decrement
        
        block= station.data_days[epoch_day_i].percent_bikes[START_TIME:DATAPOINTS_PER_DAY]
        block= np.reshape(block, (DATAPOINTS_PER_DAY - START_TIME, 1))
        fullness[x:x + block.shape[0], y:y + block.shape[1]]= block
        #if num in arr:
        
        block= station.data_days[epoch_day_i].bikes[START_TIME:DATAPOINTS_PER_DAY]
        block_5minchange= np.zeros(DATAPOINTS_PER_DAY - START_TIME, dtype=np.int)
        block_15minchange= np.zeros(DATAPOINTS_PER_DAY - START_TIME, dtype=np.int)
        block_45minchange= np.zeros(DATAPOINTS_PER_DAY - START_TIME, dtype=np.int)
        for block_i in range(len(block)):
            five_ago= block[block_i]; fifteen_ago= block[block_i]; fourtyfive_ago= block[block_i];
            accept= False
            decrements= 10
            for decrement in reversed(range(decrements)): # i.e. [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
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
            #print("current: ", current)
            block_5minchange[block_i]= current - five_ago
            block_15minchange[block_i]= current - fifteen_ago
            block_45minchange[block_i]= current - fourtyfive_ago
        block= np.reshape(block, (DATAPOINTS_PER_DAY - START_TIME, 1))
        block_5minchange= np.reshape(block_5minchange, (DATAPOINTS_PER_DAY - START_TIME, 1))
        block_15minchange= np.reshape(block_15minchange, (DATAPOINTS_PER_DAY - START_TIME, 1))
        block_45minchange= np.reshape(block_45minchange, (DATAPOINTS_PER_DAY - START_TIME, 1))
        bikes_changes_past5[x:x + block.shape[0], y:y + block.shape[1]]= block_5minchange
        bikes_changes_past15[x:x + block.shape[0], y:y + block.shape[1]]= block_15minchange
        bikes_changes_past45[x:x + block.shape[0], y:y + block.shape[1]]= block_45minchange


# In[ ]:


print(fullness.shape)
print(bikes_changes_past5.shape)
print(bikes_changes_past15.shape)
print(bikes_changes_past45.shape)
print(day_of_week.shape)
print(hour_of_day.shape)

counter= 0
limit= 22051
for row_i in range(limit):
    print("\n####################################")
    print(day_of_week[row_i])
    print(hour_of_day[row_i])
    print(fullness[row_i])
    print(bikes_changes_past5[row_i])
    print(bikes_changes_past15[row_i])
    print(bikes_changes_past45[row_i])
    print("####################################")


# def station_graph(index):
#     if MAX_DISPLAYED_DAYS < TOTAL_DAYS:
#         colors= cm.rainbow(np.linspace(0, 1, MAX_DISPLAYED_DAYS))
#     else:
#         colors= cm.rainbow(np.linspace(0, 1, TOTAL_DAYS))
# 
#     for (day_i, day), c in zip(enumerate(stations[index].data_days), colors): # use day_i + 1 to access days because day 0 is empty but kept so that the day's date works as an array index too
#         if day_i >= MAX_DISPLAYED_DAYS:
#             break
#         #plt.scatter(station.daily_epoch_time, station.percent_bikes, 1, marker="*", color= c, linewidth=0.0001)
#         plt.plot(day.daily_epoch_time, day.percent_bikes, 1, color= c, linewidth=1)
#     plt.ylabel('% of station\'s bikes available')
#     plt.xlabel('time')
#     plt.xticks(())
#     plt.yticks(())
# 
#     plt.show()
# 
#     #plt.scatter(daily_epoch_time, percent_bikes,  color='black')
# 
# station_graph(get_station_id('PORTOBELLO ROAD'))
