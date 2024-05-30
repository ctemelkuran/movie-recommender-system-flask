# Movie Recommender System

This repository contains a Movie Recommender System built with Flask, utilizing both content-based filtering and hybrid recommendation techniques. The system allows users to get movie recommendations based on a given movie title or a combination of user ID and movie title.

## Features

- **Content-Based Filtering**: Recommends movies similar to a given movie based on its description.
- **Hybrid Recommendation**: Combines content-based filtering with collaborative filtering to provide personalized movie recommendations.

## Requirements

- Python 3.x
- Flask
- pandas
- scikit-learn
- surprise

## Usage

1. Start the Flask application:
    ```bash
    python app.py
    ```

2. Open your web browser and go to `http://127.0.0.1:5000/`.

3. Enter a movie name to get recommendations or provide a user ID and movie title for a hybrid search.

## File Structure

- `app.py`: The main Flask application file that handles routes and user inputs.
- `recommender.py`: Contains functions for content-based and hybrid recommendation.
- `poster_service.py`: Retrieves movie poster URLs from The Movie Database (TMDb) API.
- `templates/`: Contains HTML templates for the web interface.
  - `base.html`: Base template with common layout.
  - `index.html`: Main page for user input.
  - `recommendations.html`: Displays the recommended movies.

## Example

### Content-Based Recommendation

1. Enter a movie name (e.g., "Inception") on the main page.
2. The system will display a list of movies similar to "Inception" along with their posters.

### Hybrid Recommendation

1. Enter a user ID and a movie title (e.g., User ID: 1, Movie Title: "Toy Story").
2. The system will display personalized movie recommendations based on the user's preferences and the given movie title.

## API Key

To use the poster service, you need an API key from The Movie Database (TMDb). Replace the placeholder `api_key` in `poster_service.py` with your actual TMDb API key.

```python
api_key = 'your_tmdb_api_key'
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

The Movie Database (TMDb) for providing the API to fetch movie posters.
MovieLens for providing the movie rating datasets.
