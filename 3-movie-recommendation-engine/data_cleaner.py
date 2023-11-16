import pandas as pd
import requests


def fetch_movie_title(movie_name):
    try:
        response = requests.get('https://search.imdbot.workers.dev/', params={'q': movie_name})
        response.raise_for_status()
        data = response.json()
        if '#TITLE' in data['description'][0]:
            return data['description'][0]['#TITLE']
        else:
            raise ValueError(f"Brak danych '#TITLE' dla filmu: {movie_name}")
    except requests.RequestException as e:
        print(f"Błąd podczas zapytania do API dla filmu '{movie_name}': {e}")
    except ValueError as e:
        print(e)


def process_dataframe(df):
    processed_titles = {}
    for column in df.columns:
        if column.startswith('Nazwa'):
            for idx, value in enumerate(df[column]):
                if isinstance(value, str) and value not in processed_titles:
                    title = fetch_movie_title(value)
                    if title:
                        processed_titles[value] = title
                        df.at[idx, column] = title
                elif value in processed_titles:
                    df.at[idx, column] = processed_titles[value]


if __name__ == '__main__':
    df = pd.read_excel(r'data.xlsx')
    process_dataframe(df)
    df.to_excel('parsed_data.xlsx')
