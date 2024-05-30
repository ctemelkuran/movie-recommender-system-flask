from flask import Flask, render_template, request

from poster_service import get_poster_url
from recommender import content_based_recommender, hybrid_recommender

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'movie_name' in request.form:
            return handle_movie_recommendations(request.form['movie_name'])
        elif 'user_id' in request.form and 'movie_title' in request.form:
            return handle_hybrid_search(request.form['user_id'], request.form['movie_title'])
    return render_template('index.html')

def handle_movie_recommendations(movie_name):
    recommendations = content_based_recommender(movie_name)
    print("Recommended films:")
    print(recommendations)
    if not recommendations:  # Empty recommendation list
        message = "Upps, the movie you have searched not found: {}".format(movie_name)
        return render_template('index.html', message=message)
    posters = [get_poster_url(movie) for movie in recommendations]
    return render_template('recommendations.html', movie_name=movie_name, recommendations=recommendations,
                           posters=posters)

def handle_hybrid_search(user_id, movie_title):
    recommendations = hybrid_recommender(user_id, movie_title)
    print("Hybrid search results:")
    print(recommendations)
    if not recommendations:  # Empty recommendation list
        message = "Upps, the movie you have searched not found: {}".format(movie_title)
        return render_template('index.html', message=message)
    posters = [get_poster_url(movie) for movie in recommendations]
    return render_template('recommendations.html', user_id=user_id, recommendations=recommendations,
                           posters=posters, movie_name=movie_title)

if __name__ == '__main__':
    app.run(debug=True)
