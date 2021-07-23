# Predicting_bike_distribution

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