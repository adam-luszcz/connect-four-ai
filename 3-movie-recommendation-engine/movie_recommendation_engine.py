import pandas as pd
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split
from surprise import KNNBasic
from surprise import accuracy


df = pd.read_excel('parsed_data.xlsx')
reader = Reader(rating_scale=(1, 10))

# Przetworzenie danych do formatu zgodnego z biblioteką Surprise
data_list = []
for index, row in df.iterrows():
    user = str(row['Osoba'])
    for i in range(1, len(row), 2):
        movie = row.iloc[i]
        rating = row.iloc[i + 1]
        # Wartosci puste w arkuszu sa z jakiegos powodu interpretowane jako float
        if isinstance(movie, float) or isinstance(rating, float):
            continue
        data_list.append((user, movie, rating))


# Utworzenie DataFrame z przetworzonych danych
processed_data = pd.DataFrame(data_list, columns=['Osoba', 'Nazwa', 'Ocena'])

data = Dataset.load_from_df(processed_data[['Osoba', 'Nazwa', 'Ocena']], reader)


# Podział danych na zestawy treningowe i testowe
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

sim_options = {
    'name': 'pearson',
    'user_based': True
}
model = KNNBasic(sim_options=sim_options)
model.fit(trainset)

# Ocena modelu na danych testowych
predictions = model.test(testset)
accuracy.rmse(predictions)

selected_user = 'Paweł Czapiewski'

# Utwórzenie listę filmów, które użytkownik już ocenił
rated_movies = processed_data[processed_data['Osoba'] == selected_user]['Nazwa'].tolist()

# Wygenerowanie rekomendacji dla wybranego użytkownika
recommendations = []
for movie_id in processed_data['Nazwa'].unique():
    if movie_id not in rated_movies:
        predicted_rating = model.predict(selected_user, movie_id).est
        recommendations.append((movie_id, predicted_rating))

top_recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:5]

print(f"Top 5 rekomendacji dla użytkownika {selected_user}:")
for movie, rating in top_recommendations:
    print(f"{movie}: {rating}")
