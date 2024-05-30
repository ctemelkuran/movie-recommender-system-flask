import pandas as pd
import random
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

# Content based filtering
# First section the filtering is made by TD-IDF analysis according to movie descriptions

# Read movies metadata from CSV file
movies_metadata_df = pd.read_csv('movie-lens-dataset/movies_metadata.csv', low_memory=False)

# Extract genres from the 'genres' column
movies_metadata_df['genres'] = movies_metadata_df['genres'].apply(
    lambda x: [genre['name'] for genre in literal_eval(x)] if isinstance(x, str) and x != '' else [])

# Calculate vote counts and vote averages
vote_counts = movies_metadata_df['vote_count'].dropna().astype(int)
vote_averages = movies_metadata_df['vote_average'].dropna().astype(int)
vote_avg_mean = vote_averages.mean()

# Calculate 95th percentile of vote counts
vote_counts_95 = vote_counts.quantile(0.95)

# Extract year from release_date and convert to datetime
movies_metadata_df['year'] = pd.to_datetime(movies_metadata_df['release_date'], errors='coerce').dt.year

# Read links_small from CSV file
links_small = pd.read_csv('movie-lens-dataset/links_small.csv')
links_small = links_small.loc[links_small['tmdbId'].notnull(), 'tmdbId'].astype(int)
movies_metadata_df = movies_metadata_df.drop([19730, 29503, 35587])  # Remove a bad data which is in wrong format

# Convert 'id' column to integer
movies_metadata_df['id'] = movies_metadata_df['id'].astype(int)

# Select movies from movies_metadata_df that have corresponding IDs in links_small
movies_metadata_s = movies_metadata_df[movies_metadata_df['id'].isin(links_small)].copy()

# Fill missing values in 'tagline' and 'description' columns
movies_metadata_s['tagline'] = movies_metadata_s['tagline'].fillna('')
movies_metadata_s['description'] = movies_metadata_s['overview'].fillna('') + movies_metadata_s['tagline'].fillna('')

# Create TF-IDF matrix
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(movies_metadata_s['description'])

# Calculate cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Reset index of movies_metadata_s DataFrame
movies_metadata_s = movies_metadata_s.reset_index()

# Create a Series for movie titles and indices
titles = movies_metadata_s['title']
indices = pd.Series(movies_metadata_s.index, index=movies_metadata_s['title'])


def content_based_recommender(title, top_n=10):
    if title in indices:
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        if len(sim_scores) > 1:
            sim_scores = sim_scores[1:top_n + 1]
            movie_indices = [i[0] for i in sim_scores]
            return titles.iloc[movie_indices].tolist()  # Return movie titles

    return []


# SVD Predictions

reader = Reader()
# Read the ratings from CSV file
ratings = pd.read_csv('movie-lens-dataset/ratings_small.csv')
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

svd = SVD()

# Perform cross-validation
results = cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Print the average RMSE and MAE scores
print("Average RMSE:", round(results['test_rmse'].mean(), 3))
print("Average MAE:", round(results['test_mae'].mean(), 3))

# Get a random movie and retrieve users who rated it
movie_ids = pd.Series(ratings['movieId'].unique())
valid_movie_ids = movie_ids

if len(valid_movie_ids) == 0:
    print("No valid movie IDs found.")
else:
    movie_id = random.choice(valid_movie_ids)
    user_ids = ratings[ratings['movieId'] == movie_id]['userId'].sample(n=5, replace=True).tolist()

    for user_id in user_ids:
        prediction = svd.predict(user_id, movie_id)
        predicted_rating = prediction.est
        print("User ID:", user_id)
        print("Predicted rating:", predicted_rating)

        user_ratings = ratings[(ratings['userId'] == user_id) & (ratings['movieId'] == movie_id)]

        if not user_ratings.empty:
            actual_rating = user_ratings['rating'].values[0]
            print("Actual rating: ", actual_rating)
        else:
            print("No rating available for the specified user")
        print()

# Hybrid Approach

# Read the ID mapping data from links_small.csv
id_mapping = pd.read_csv('movie-lens-dataset/links_small.csv')[['movieId', 'tmdbId']]
id_mapping['tmdbId'] = pd.to_numeric(id_mapping['tmdbId'], errors='coerce')
id_mapping.columns = ['movieId', 'id']
id_mapping = id_mapping.merge(movies_metadata_s[['title', 'id']], on='id').set_index('title')
indices_mapping = id_mapping.set_index('id')


def hybrid_recommender(user_id, title, top_n=10):
    if title not in indices:
        return []  # Return an empty list if the movie title is not found

    idx = indices[title]
    movie_id = id_mapping.loc[title]['id']
    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]

    recommended_movies = movies_metadata_s.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
    recommended_movies['est'] = recommended_movies['id'].apply(
        lambda x: svd.predict(user_id, indices_mapping.loc[x]['movieId']).est) # Perform predictions
    recommended_movies = recommended_movies.sort_values('est', ascending=False)
    recommended_movies = recommended_movies.head(top_n)['title'].tolist()  # Extract movie titles as a list
    return recommended_movies
