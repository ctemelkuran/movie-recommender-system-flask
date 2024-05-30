import requests

def get_poster_url(title):
    api_key = 'YOUR_API_KEY'

    search_url = f'https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={title}'
    response = requests.get(search_url)
    data = response.json()

    if 'results' in data and len(data['results']) > 0:
        result = data['results'][0]
        poster_path = result['poster_path']

        base_url = 'https://image.tmdb.org/t/p/w500'
        poster_url = f'{base_url}{poster_path}' if poster_path else None

        return poster_url

    return None
