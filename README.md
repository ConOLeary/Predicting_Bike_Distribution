# Predicting_bike_distribution

See report: PredictDistribDublinBike.pdf

## Development Log

09 / 08 / 2021

* Initial effort at a 'quick' baseline model is a complete flop. I certainly have ammendments in mind, but I am evaluating the added complexity of such and questioning if making them would render the baseline modal not baseline.
I think the way I have invisioned y is not relevant to the baseline model. if I simply made y= time, then I could make predictions on the test y values, and technically I could be enquiring from the perspective of a time 10, 30 or 60 minutes beforehand, but then it wouldn't be a proper test scenario, as I would have the data points surrounding the 'future' time we are predicting at.
I think what I will have to do is find a way to do the train test split so that the test datapoints are contiguous in terms of time.
* Current approach for the baseline of using the first 4/5 of the data as training for a linear regression with poly features is a complete flop. Next 'easy' thing to do would be making a poly linear regression for each day
* Yep.. baseline model is certainly not a baseline model at this point - might as well finish it and have it an additional model

08 / 08 / 2021

* Just realising now that I should be producing graphs for the tuning process, like optimisating alpha
* I now understand the purpose of a baseline modal (thanks to this blog https://blog.insightdatascience.com/always-start-with-a-stupid-model-no-exceptions-3a22314b9aaa); linear regression baseline modal could easily be done. I see how building a baseline first would have lent insight to my more complex approaches

07 / 08 / 2021

* bikes_changes_pastx finally ironed out to perfection, but models will not return positive results ..
* A ~10% leap in accuracy for approach 1! However, going to see what features from bikes_changes_pastx I can thin out without significant accuracy loss as the load time for approach 1 is way up. Would also like to see if giving approach 1 some data from stations that aren't in question (but not as much as that about the station in question) has benefit
* So I forgot to increase the feature matrix row size when I added the new bikes_changes_pastx features on the last run, which led to past5, past10, & past15 being used instead of past5, past15, & past45, and that is what caused the 10% accuracy increase. Accident is the mother of creation
* 90 & 80 % on approach 1 (portobello road & custom house quay) with past 5 .. past 15
  90 & 79 % on approach 1 (portobello road & custom house quay) with past 5 .. past 25 + past 35 + past 45
  90 " .. " with past 5 .. past 25
  90 " .. " with past 5 + past 10
  82 " .. " with past 5 + past 10 for all stations
  80 " .. " with past 5 + past 10, and past 15 + past 20 for all stations (the all stations here could be subed for stations that are near in terms of latitude and longitude)
  77 " .. " with past 5 + past 10, and percent_fullness for all stations
  88 " .. " with nothing
  88 " .. " with past 5 + past 10, and extra hidden layers! (maybe the extra hidden layers should be coupled with extra pastxs)
  90 " .. " with past 5 + past 10, and x10 default alpha


06 / 08 / 2021

* Well on the way to fixing fullness_xago once and for all. However, every data_day has a row of 0s at their end, which I will solve at the root earlier, rather then compensating for it when generating features
* Things i've learned from this project that have little to nothing to do with machine learning:
    Don't start structuring your data until you know what you want from it
    If you write code with questionable use of indices, everything built on top of that has to compensate for the irregularity
    When divising algorithms, visualise it with a pen and paper wherever applicable
* I should evaluate to see how often the models are completely accurate, as well as how often they are wrong by 1 or 2 bikes

05 / 08 / 2021

* Have bike fullness changes for a range of times available now from the FEATURE DATA PREPERATION cell, so going to play around with different combinations of them as features for both approaches

04 / 08 / 2021

* On Portobello road with version of apporach 1 generalised for all stations

    Data split  1  accuracy:  91.5901298779  %
    Data split  2  accuracy:  90.84333255725707  %
    Data split  3  accuracy:  92.11429567826995  %
    Data split  4  accuracy:  76.6831594124561  %
    Data split  5  accuracy:  -34.84047791784899  %

    Average accuracy of model:  63.27808792160682  %

* Same as above but for Custom House Quay

    Data split  1  accuracy:  59.086351542752126  %
    Data split  2  accuracy:  61.66025837446901  %
    Data split  3  accuracy:  57.31415044971864  %
    Data split  4  accuracy:  57.844545451913476  %
    Data split  5  accuracy:  4.349001760599261  %

    Average accuracy of model:  48.050861515890496  %

* On Custom House Quay with version of apporach 1 specified to station

    Data split  1  accuracy:  83.84539054161961  %
    Data split  2  accuracy:  86.32709352743758  %
    Data split  3  accuracy:  86.22389292961451  %
    Data split  4  accuracy:  82.57340319003345  %
    Data split  5  accuracy:  61.32177880424313  %

    Average accuracy of model:  80.05831179858966  %

* Same as above but for Portobello Road

    Data split  1  accuracy:  96.40710865746992  %
    Data split  2  accuracy:  97.45073515003462  %
    Data split  3  accuracy:  98.44706688236448  %
    Data split  4  accuracy:  96.08811423178055  %
    Data split  5  accuracy:  67.28540332565863  %

    Average accuracy of model:  91.13568564946164  %

* On Portobello Road with version of apporach 2 specified to station

[0.94888804 0.95045243 0.96736847 0.94180191 0.23107503]
cv_scores mean:0.8079171763199342

* Same as above but for Custom House Quay

[0.76235509 0.80046548 0.79188385 0.77405647 0.63267826]
cv_scores mean:0.7522878279125137

* On Portobello road with version of apporach 2 generalised for all stations

[ 0.1180936   0.10373099 -0.01712411 -0.14956879 -0.55059175]
cv_scores mean:-0.09909201204348392

* Same as above but for Custom House Quay

[ 0.29359842  0.22464943  0.26320513  0.20161131 -3.76352284]
cv_scores mean:-0.5560917105315376

03 / 08 / 2021

* KNN having bikes_changes_past45 makes it less accurate - probably because datapoints that are neighbours insofar as they are similar in their changes in the past 45 minutes are not necessarly similar in the future fullness change of their respective stations
* Tried evaluating model 1 differntly - same results - error must be in model
* X input array had datatype set to int, so all float feature into was being rounded

02 / 08 / 2021

* reason accuracy was so high is because of the following:

X[0:MAX_TIME, 139:247]= fullness_in10
X[0:MAX_TIME, 247:355]= fullness_in30
X[0:MAX_TIME, 355:463]= fullness_in60

which should have been

X[0:MAX_TIME, 139:247]= bikes_changes_past5
X[0:MAX_TIME, 247:355]= bikes_changes_past15
X[0:MAX_TIME, 355:463]= bikes_changes_past45

01 / 08 / 2021

* Retrospectively, avoiding pandas probably made my life harder
* Big brain moment: going to represent the time of day as an x and y value, where a 24hr day is a circle. This way 15:00 and 16:00 are as similar to each other as 23:00 and 00:00

31 / 07 / 2021

* 1st model seems solid. Going to try a KNRegressor for the 2nd. Neighbours for a given time to predict will be the bike_fullness of that same station at that same prediction time at previous dates when the day_of_week was the same. Closer (and greater weighted) neighbours will be those where the bike_occupancy of the given station is more similar at the current time.
* Yep, so there is more to a knn when x and y aren't 1 dimensional. Trying to mentally conceive how a nearest neighbour is calculated when there are multiple features in X.

30 / 07 / 2021

* ~= 0.99938 accuracy 😳 Something must be wrong

29 / 07 / 2021

* Data from 27 / 1 / 2020 onwards is solid
* hindsight is 20,20; data processing feeling more streamlined. just got to exclude gaps in data now
* the edge cases of the block_xminchange arrays still have issues, but will sort those after the model is fit
* going to take a gamble and instead of rooting out what poor data is between 27 / 1 and 1 / 4 I am going to change the DUD_VALUE to 0 and train the model

28 / 07 / 2021

* Start-edge bikes_changes_pastx issue looking better
    bikes_changes_pastx completely ironed out in processor effective way
* Looks like it could just be exclusing poor days of data that is left in the feature data generation

27 / 07 / 2021

* There are so many kinks to iron out in the data and how I am processing it - it's starting to feel like I am writing code into a blackhole and that the ways I am structuring the data are arbitrary and redundant
An example of such a kink that I have just ironed out is changing
self.day_of_week= (STARTING_DAY + int(epoch_time / DATAPOINTS_PER_DAY ) % len(DAYS_OF_WEEK))
to
self.day_of_week= (STARTING_DAY + int(epoch_time / (DATAPOINTS_PER_DAY - 1)) % len(DAYS_OF_WEEK))
Although perhaps I have learned something here and not just applied a patchwork solution insofar as I imagine if there is a length of an array, I should universaly divide by the length of the array - 1
* .ipynb often unviewable. Perhaps I should export a .py before every commit
* Checklist at this point seems to be:
Fix problem with start-edge bikes_changes_pastx
Find way to exclude days with insufficent data
Fix hour_of_day

26 / 07 / 2021

* Some values in bikes_changes_past45 are greater than 100 - something must be wrong
* 3d confusion matrix could be used to gauge results: e.g. negative change in bikes, no change, and positive. Perhaps a better way to visualise though

25 / 07 / 2021

* 5, 15, and 45 minutes seem to be more appropriate time spans to look at previous bike station fullness changes in
* I want to change OPEN_HOURS to 18 but the data processing system ain't ready yet

24 / 07 / 2021

* For purposes of MLPreg approach, DataDay information is now as sparse arrays. This is causing graphs to look funny though
* I suspect long logs from print statements causing trouble for j notebook and github
* I need to make a change to the code in the .ipynb file so that the logs from the cells change
* There are many holes in 'fullness' due to gaps in the data. Since features for a given time are dependent on there being data before it, I will only generate sets of training features where there has been data in all stations for the past hour and where there is data in all stations for the coming hour

23 / 07 / 2021

* Might try an MLPRegressor approach, with input nodes being:
the 7 days of the week +
the 18 active hours of the day +
the fullness of the 117 stations +
changes in capacity for the 117 stations in the past 10, 30, and 60 minutes,
where the 7 are either 1 or 0 (it is that day or it's not),
the 18 are from 0 to 1 (eg if it's half 6 when 6 and 7 are 0.5)
and the 351 (117 * 3) are from 0 to N

22 / 07 / 2021

* There's a bug when you set INPUT_ROWS_LIMIT a good bit over 1,000,000
* Feature brainstorm: Getting rainfall data; lat & long where bikes leave, and then assuming they'll turn up at a distance in some time; increase / decrease in bikes at a station rather than bikes at a station; overall ratio of bikes that are out of stations vs bikes that are in stations
* yep, there are certain days that are just missing but that shouldn't matter
* TODO: in graphing cell, empty days that are having a colour allocated to them mean there are colours being wasted

21 / 07 / 2021

* TODO: Sort station_var values by epoch_time
* Not popping missing stations from arrays anymore because that leads to station ids not being usable as indices for these arrays

20 / 07 / 2021

* Assuming for now I should train on data for all bike stations but test on select ones
* Removing INPUT_ROWS_LIMIT can crash jupyter notebook
* Some days in the 1st Jan to 24th Feb 2020 range the dataset is supposed to cover are simply absent. The order of rows is unclear, often with various patches of records covering a few hours of a stations status.
* Only reading in these station_id= []; date= []; time= []; available_bikes= []; because non-variable info can be associated to a single object
* Going to ignore "STATUS" because none of the stations are anything but "Open" in the dataset. Going to ignore "AVAILABLE BIKE STANDS" because this is probably redundant for my ends when I have "AVAILABLE BIKES"
* Apparently there is no station where id = 1
