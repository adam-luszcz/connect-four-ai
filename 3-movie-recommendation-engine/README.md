# Movie recommendation engine

The program is a movie recommendation engine.  
It uses the Surprise library to train a recommendation model and provide movie recommendations for a specific user.  
The program reads from an [Excel file](parsed_data.xlsx), handles missing values, and transforms the data into a format where each row represents a user, a movie, and a rating.  
It uses the Pearson and Cosine metrics to calculate the distance between users.  
The output is top 5 movie recommendations and 5 movies not recommended for a specific user.

## How to set up:
Install the packages from the requirements.txt with the following command `pip3 install -r requirements.txt`

## How to run:
Run the program with the following command `python3 movie_recommendation_engine.py`  
You will be asked to provide a user for whom the recommendations are to be generated.  
Please type in user name exactly as it is in the [Excel file](parsed_data.xlsx).  
The program will print top 5 movie recommendations and 5 movies not recommended with both the Pearson and Cosine metrics.  

## Usage example:

**Example 1**:

https://github.com/annarogala/ai-projects/assets/13242654/e97bc26c-2ad6-4027-8f47-b48eb5d92fa3

**Example 2**:

https://github.com/annarogala/ai-projects/assets/13242654/39fb1ab3-44ca-41b2-9848-2a32950be543

**Example 3**:

https://github.com/annarogala/ai-projects/assets/13242654/96e29c57-7094-4323-a220-98c1acce0e25

**Example 4**:

https://github.com/annarogala/ai-projects/assets/13242654/4f1f26d1-95fd-40a0-9c88-86f760119c92


## Authors:
Adam Łuszcz s22994  
Anna Rogala s21487

## Sources:
- https://surprise.readthedocs.io/
- https://pandas.pydata.org/docs/
