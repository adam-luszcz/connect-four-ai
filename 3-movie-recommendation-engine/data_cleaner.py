import pandas as pd
import requests

if __name__ == '__main__':
    df = pd.read_excel(r'data.xlsx')
    for column in df.columns:
        if column.startswith('Nazwa'):
            for idx, value in enumerate(df[column]):
                if isinstance(value, str):
                    response = requests.get('https://search.imdbot.workers.dev/', params={'q': value})
                    df.at[idx, column] = response.json()['description'][0]['#TITLE']
                    print(df.at[idx, column])

    df.to_excel('parsed_data.xlsx')
