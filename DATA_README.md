# Data Requirements

This project requires Yelp restaurant data for the Las Vegas area.

## Required Files

Place the following CSV files in the root directory of the project:

- `yelp242a_train.csv` - Training dataset
- `yelp242a_test.csv` - Test dataset

## Data Format

Each CSV file should contain the following columns:

### Target Variable
- **stars**: Average star rating (1.0 to 5.0)

### Features
- **review_count**: Number of reviews (integer)
- **GoodForKids**: Whether the business is good for kids (TRUE/FALSE/(Missing))
- **Alcohol**: Type of alcohol service ('none', "'beer_and_wine'", "'full_bar'"/(Missing))
- **BusinessAcceptsCreditCards**: Accepts credit cards (TRUE/FALSE/(Missing))
- **WiFi**: WiFi availability ('no', "'free'", "'paid'"/(Missing))
- **BikeParking**: Bike parking available (TRUE/FALSE/(Missing))
- **ByAppointmentOnly**: By appointment only (TRUE/FALSE/(Missing))
- **WheelechairAccessible**: Wheelchair accessible (TRUE/FALSE/(Missing))
- **OutdoorSeating**: Outdoor seating available (TRUE/FALSE/(Missing))
- **RestaurantsReservations**: Accepts reservations (TRUE/FALSE/(Missing))
- **DogsAllowed**: Dogs allowed (TRUE/FALSE/(Missing))
- **Caters**: Provides catering (TRUE/FALSE/(Missing))

## Data Source

This project uses data from the [Yelp Dataset](https://www.yelp.com/dataset), which is freely available for academic and educational purposes.

### How to Obtain the Data

1. Visit [Yelp Dataset](https://www.yelp.com/dataset)
2. Download the dataset (requires agreement to terms of use)
3. Extract the business and review data
4. Filter for restaurants in Las Vegas, Nevada
5. Process and split the data into training/test sets

## Data Privacy

- The dataset contains only public business information from Yelp
- No personal user information is included
- All data is aggregated and anonymized

## Notes on Missing Values

- Missing values are represented as `(Missing)` in the categorical columns
- The model treats missing values as a separate category rather than imputing them
- This approach preserves the information that a restaurant didn't provide certain details
