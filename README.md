# Predicting_bike_distribution

22 / 07 / 2021

* There's a bug when you set INPUT_ROWS_LIMIT a good bit over 1,000,000

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