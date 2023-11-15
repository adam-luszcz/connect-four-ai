import pandas as pd
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split
from surprise import KNNBasic
from surprise import accuracy
import requests


def process_data(filename):
    df = pd.read_excel(filename)
    df.fillna(0, inplace=True)

    # Przetworzenie danych do formatu zgodnego z biblioteką Surprise
    data_list = []
    for index, row in df.iterrows():
        user = str(row['Osoba'])
        for i in range(1, len(row), 2):
            movie = row.iloc[i]
            rating = row.iloc[i + 1]
            if movie and rating:
                data_list.append((user, movie, rating))

    # Utworzenie DataFrame z przetworzonych danych
    return pd.DataFrame(data_list, columns=['Osoba', 'Nazwa', 'Ocena'])


def get_movie_recommendations(model, trainset, testset, user):
    model.fit(trainset)
    predictions = model.test(testset)
    accuracy.rmse(predictions)
    # Utworzenie listy filmów, które użytkownik już ocenił
    rated_movies = processed_data[processed_data['Osoba'] == user]['Nazwa'].tolist()

    # Wygenerowanie rekomendacji dla wybranego użytkownika
    recommendations = []
    for movie_id in processed_data['Nazwa'].unique():
        if movie_id not in rated_movies:
            predicted_rating = model.predict(selected_user, movie_id).est
            recommendations.append((movie_id, predicted_rating))

    top_recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:5]
    do_not_watch = sorted(recommendations, key=lambda x: x[1])[:5]
    return top_recommendations, do_not_watch


def print_movie_recommendations(recommendations):
    for movie, rating in recommendations:
        response = requests.get('https://search.imdbot.workers.dev/', params={'q': movie})
        year = response.json()['description'][0]['#YEAR']
        actors = response.json()['description'][0]['#ACTORS']
        imdb_url = response.json()['description'][0]['#IMDB_URL']
        print(f'''
        =============================================
        {movie}: {rating}
        Year: {year}
        Actors: {actors}
        IMDB URL: {imdb_url}
        =============================================
        ''')

processed_data = process_data('parsed_data.xlsx')
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(processed_data[['Osoba', 'Nazwa', 'Ocena']], reader)

# Podział danych na zestawy treningowe i testowe
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

sim_options_pearson = {
    'name': 'pearson',
    'user_based': True
}
sim_options_cosine = {
    'name': 'cosine',
    'user_based': True
}

model_pearson = KNNBasic(sim_options=sim_options_pearson)
model_cosine = KNNBasic(sim_options=sim_options_cosine)

selected_user = input('Podaj użytkownika dla którego chcesz otrzymać rekomendacje: ')
while processed_data[processed_data['Osoba'] == selected_user].empty:
    print('Podany użytkownik nie istnieje w bazie!')
    selected_user = input('\nPodaj użytkownika dla którego chcesz otrzymać rekomendacje: ')

top_recommendations_pearson, do_not_watch_pearson = get_movie_recommendations(model_pearson, trainset, testset, selected_user)
print('Metryka liczenia odległości: pearson')
print(f'Top 5 rekomendacji dla użytkownika {selected_user}:')
print_movie_recommendations(top_recommendations_pearson)
print(f'\nUżytkownik {selected_user} nie powinien oglądać:')
print_movie_recommendations(do_not_watch_pearson)

top_recommendations_cosine, do_not_watch_cosine = get_movie_recommendations(model_cosine, trainset, testset, selected_user)
print('Metryka liczenia odległości: cosine')
print(f'Top 5 rekomendacji dla użytkownika {selected_user}:')
print_movie_recommendations(top_recommendations_cosine)
print(f'\nUżytkownik {selected_user} nie powinien oglądać:')
print_movie_recommendations(do_not_watch_cosine)
