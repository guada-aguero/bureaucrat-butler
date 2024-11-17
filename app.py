import os
import io
import re
import json
import logging
import requests
import pandas as pd
import numpy as np
import openai
import base64  # Added for Idealista encoding
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template
from sklearn.metrics.pairwise import cosine_similarity
from google.cloud import translate_v2 as translate
from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# API Credentials
openai.api_key = os.getenv('OPENAI_API_KEY')
IDEALISTA_API_KEY = os.getenv('IDEALISTA_API_KEY')
IDEALISTA_API_SECRET = os.getenv('IDEALISTA_API_SECRET')
COST_OF_LIVING_API_KEY = os.getenv('COST_OF_LIVING_API_KEY')

# Load data
city_df = pd.read_csv('worldcities_portugal.csv')
with open('KB_data.json', 'r', encoding='utf-8') as f:
    knowledge_base = json.load(f)

# Define the base directory for static files
static_data_dir = os.path.join(os.getcwd(), 'static', 'data')

# Load the JSON files from the static/data directory
with open(os.path.join(static_data_dir, 'finance_offices.json'), 'r', encoding='utf-8') as f:
    finance_offices = json.load(f)

with open(os.path.join(static_data_dir, 'ss_offices_data.json'), 'r', encoding='utf-8') as f:
    ss_offices = json.load(f)

with open(os.path.join(static_data_dir, 'aima_offices.json'), 'r', encoding='utf-8') as f:
    aima_offices = json.load(f)

with open(os.path.join(static_data_dir, 'embassies_data.json'), 'r', encoding='utf-8') as f:
    embassies_in_portugal = json.load(f)

with open(os.path.join(static_data_dir, 'portuguese_embassies.json'), 'r', encoding='utf-8') as f:
    portuguese_embassies_abroad = json.load(f)


# Google Translate client
translate_client = translate.Client()

### Utility Functions ###

# Categorize cities into urban, suburban, and rural
def categorize_cities():
    urban_cities = ['Porto', 'Lisbon', 'Coimbra', 'Faro', 'Portimao', 'Funchal', 'Braga', 'Setubal', 'Vila Nova de Gaia', 'Aveiro', 'Leiria']
    urban, suburban, rural = [], [], []
    
    for _, row in city_df.iterrows():
        city_name = row['city']
        if city_name in urban_cities:
            urban.append(city_name)
        elif row['population'] > 100000:
            suburban.append(city_name)
        else:
            rural.append(city_name)
    
    return {"urban": urban, "suburban": suburban, "rural": rural}

# Categorize the cities
city_data = categorize_cities()

# Fetch cost of living data from API
def fetch_cost_of_living(city, country):
    url = "https://cost-of-living-and-prices.p.rapidapi.com/prices"
    querystring = {"city_name": city, "country_name": country}
    headers = {
        "x-rapidapi-key": COST_OF_LIVING_API_KEY,
        "x-rapidapi-host": "cost-of-living-and-prices.p.rapidapi.com"
    }

    try:
        response = requests.get(url, headers=headers, params=querystring)
        if response.status_code == 200:
            return response.json()
        else:
            logging.error(f"Failed to fetch cost of living data. Status code: {response.status_code}")
            return {}
    except requests.exceptions.RequestException as e:
        logging.error(f"An error occurred: {e}")
        return {}


# Fetch rent data from Idealista API
def get_idealista_token():
    credentials = f"{IDEALISTA_API_KEY}:{IDEALISTA_API_SECRET}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()
    
    headers = {
        "Authorization": f"Basic {encoded_credentials}",
        "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"
    }
    data = {"grant_type": "client_credentials", "scope": "read"}

    response = requests.post("https://api.idealista.com/oauth/token", headers=headers, data=data)
    if response.status_code == 200:
        return response.json().get('access_token')
    return None

#def search_properties(token, city_lat, city_lng, distance, total_budget, property_type):
#    max_price = float(total_budget) * 0.5  # 50% of total budget
#    valid_property_types = ['bedrooms', 'garages', 'homes', 'offices', 'premises', 'transfers', 'buildings', 'storageRooms', 'newDevelopments']
#    
#    if property_type not in valid_property_types:
#        return []
#
#    url = "https://api.idealista.com/3.5/pt/search"
#    headers = {
#        "Authorization": f"Bearer {token}",
#        "Content-Type": "application/x-www-form-urlencoded"
#    }
#    data = {
#        "operation": "rent",
#        "propertyType": property_type,
#        "center": f"{city_lat},{city_lng}",
#        "distance": distance,
#        "maxPrice": max_price,
#        "locale": "pt",
#        "numPage": 1,  # Pagination
#        "maxItems": 20
#    }
#
#    # Logging the outgoing request
#    logging.info(f"Searching properties with params: {data}")
#
#    response = requests.post(url, headers=headers, data=data)
#    
#    if response.status_code == 200:
#        property_list = response.json().get('elementList', [])
#        
#        # Logging the returned properties
#        logging.info(f"Found {len(property_list)} properties.")
#
#        # Format properties for frontend
#        return [{
#            'name': prop.get('suggestedTexts', {}).get('title', 'No title'),
#            'price': prop.get('price', 'N/A'),
#            'image_url': prop.get('thumbnail'),  # Extracting image URL from the 'thumbnail' key
#            'link': prop.get('url')  # Extracting property URL from the 'url' key
#        } for prop in property_list]
#
#    else:
#        logging.error(f"Idealista API Error: {response.status_code} - {response.text}")
#        return []
#    
#def format_recommendations(rent_data, city):
#    recommendations = f"<h3>Available Properties in {city}, Portugal</h3>"
#    
#    recommendations += '<div class="property-cards">'
#    for prop in rent_data:
#        name = prop.get('suggestedTexts', {}).get('title', 'No title')
#        price = prop.get('price', 'N/A')
#        image_url = prop.get('thumbnail') or '/static/default_property_image.jpg'  # Set default image if None
#        link = prop.get('url') or '#'  # Set a default link if None
#
#        recommendations += f"""
#            <div class="property-card">
#                <img src="{image_url}" alt="Property Image" width="150" height="150">
#                <div class="property-info">
#                    <h4>{name}</h4>
#                    <p>Price: €{price}</p>
#                    <a href="{link}" target="_blank" class="view-property-btn">View Property</a>
#                </div>
#            </div>
#        """
#    
#    recommendations += "</div>"
#    
#    return recommendations

# Search properties from Idealista
def search_properties(token, city_lat, city_lng, distance, total_budget, property_type):
    max_price = float(total_budget) * 0.5  # 50% of total budget
    valid_property_types = ['bedrooms', 'garages', 'homes', 'offices', 'premises', 'transfers', 'buildings', 'storageRooms', 'newDevelopments']

    if property_type not in valid_property_types:
        return []

    url = "https://api.idealista.com/3.5/pt/search"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "operation": "rent",
        "propertyType": property_type,
        "center": f"{city_lat},{city_lng}",
        "distance": distance,
        "maxPrice": max_price,
        "locale": "pt",
        "numPage": 1,  # Pagination
        "maxItems": 20
    }

    logging.info(f"Searching properties with params: {data}")

    response = requests.post(url, headers=headers, data=data)

    if response.status_code == 200:
        property_list = response.json().get('elementList', [])
        logging.info(f"Found {len(property_list)} properties.")
        return property_list
    else:
        logging.error(f"Idealista API Error: {response.status_code} - {response.text}")
        return []

# Format recommendations for frontend
def format_recommendations(rent_data, city):
    if not rent_data:
        return "<p>No properties found for the selected city and budget.</p>"

    recommendations = f"<h3>Available Properties in {city}, Portugal</h3>"
    recommendations += '<div class="property-cards">'

    for prop in rent_data:
        name = prop.get('suggestedTexts', {}).get('title', 'No title')
        price = prop.get('price', 'N/A')
        image_url = prop.get('thumbnail') or '/static/default_property_image.jpg'
        link = prop.get('url') or '#'

        recommendations += f"""
            <div class="property-card">
                <img src="{image_url}" alt="Property Image" width="150" height="150">
                <div class="property-info">
                    <h4>{name}</h4>
                    <p>Price: €{price}</p>
                    <a href="{link}" target="_blank" class="view-property-btn">View Property</a>
                </div>
            </div>
        """

    recommendations += "</div>"
    return recommendations

#BOT    
# Stopwords and important stopwords
STOPWORDS = {"in", "how", "to", "can", "do", "i", "what", "are", "the", "a", "for", "portugal", "im", "am", "and", "at"}
IMPORTANT_STOPWORDS = {"requirements", "documents", "visa", "nif", "niss", "work", "permit"}

# Helper function to get embeddings from OpenAI API
def get_embedding(text):
    response = openai.Embedding.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return np.array(response['data'][0]['embedding'])

# Precompute embeddings for KB entries
def precompute_kb_embeddings(kb):
    kb_embeddings = []
    for category in kb['categories']:
        for entry in category['entries']:
            text = entry['question'] + ' ' + entry['answer']['summary']  # Concatenate question and summary
            embedding = get_embedding(text)
            kb_embeddings.append((entry, embedding))
    return kb_embeddings

# Precompute embeddings for the knowledge base
kb_embeddings = precompute_kb_embeddings(knowledge_base)

# Clean and filter the query
def clean_query(query):
    query_cleaned = re.sub(r'[^\w\s]', '', query.lower()) 
    words = query_cleaned.split()
    filtered_words = [word for word in words if word not in STOPWORDS or word in IMPORTANT_STOPWORDS]
    cleaned_query = ' '.join(filtered_words)
    logging.info(f"Cleaned query: {cleaned_query}")
    return cleaned_query

# Restore Google Translate API functions
def detect_language(text):
    result = translate_client.detect_language(text)
    return result['language']

def translate_to_english(text):
    result = translate_client.translate(text, target_language='en')
    return result['translatedText']

def translate_from_english(text, target_language):
    result = translate_client.translate(text, target_language=target_language)
    return result['translatedText']

# Hybrid search combining keyword and embedding
def filter_by_keyword(query, kb_embeddings):
    query_keywords = query.lower().split()
    filtered = []
    for entry, embedding in kb_embeddings:
        entry_keywords = entry['keywords']
        # Check if any of the keywords in the query match the entry keywords
        if any(keyword in query_keywords for keyword in entry_keywords):
            filtered.append((entry, embedding))
    
    # If no keyword match found, return the whole KB for embedding comparison
    return filtered if filtered else kb_embeddings

# Weighted similarity score for intent boosting
def weighted_similarity_score(query, entry, kb_embedding, query_embedding):
    # Assign higher weights to key visa-related terms
    keyword_weight = 1
    if 'd7' in query.lower() and 'd7' in entry['keywords']:
        keyword_weight = 1.5  # Boost D7 entries by 50%
    elif 'financial' in query.lower() and 'financial' in entry['keywords']:
        keyword_weight = 1.4  # Boost financial-related entries by 40%

    similarity = cosine_similarity([query_embedding], [kb_embedding])[0][0]
    return similarity * keyword_weight

# Find the best match for a query using cosine similarity
def find_best_match(query, kb_embeddings, threshold=0.65):
    filtered_embeddings = filter_by_keyword(query, kb_embeddings)
    query_embedding = get_embedding(query)
    similarities = []

    for entry, kb_embedding in filtered_embeddings:
        similarity = weighted_similarity_score(query, entry, kb_embedding, query_embedding)
        logging.debug(f"Similarity for '{entry['question']}' is {similarity}")
        similarities.append((entry, similarity))
    
    best_match = max(similarities, key=lambda x: x[1])
    
    if best_match[1] >= threshold:
        logging.info(f"Best match found with similarity score: {best_match[1]}")
        return best_match[0], best_match[1]
    else:
        logging.warning(f"No match above the threshold. Best score: {best_match[1]}")
        return None, None

# Use GPT to rephrase the KB-based response naturally
def generate_natural_answer_from_kb(query, best_match):
    prompt = f"""
    The user asked: "{query}".
    The following information from the knowledge base was found:

    Question: {best_match['question']}
    Answer Summary: {best_match['answer']['summary']}
    Answer Details: {', '.join(best_match['answer']['details'])}
    
    Please provide a natural and conversational response based on this information.
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that uses the knowledge base strictly to provide answers. Only reference external content when the KB does not have sufficient data."},
            {"role": "user", "content": prompt}
        ]
    )

    # Extract the natural response
    natural_answer = response['choices'][0]['message']['content'].strip()

    return natural_answer

def generate_natural_answer_with_links(query, best_match):
    # Step 1: Call the original function to get the natural response from GPT
    natural_answer = generate_natural_answer_from_kb(query, best_match)

    # Step 2: Dynamically append links from the knowledge base entry
    if 'reference_link' in best_match['answer']:
        links = "<br>For more information, you can check the following links:<br>"
        for link in best_match['answer']['reference_link']:
            links += f'- <a href="{link["url"]}" target="_blank">{link["text"]}</a><br>'
        # Append the links to the response
        natural_answer += links

    return natural_answer

# Fallback to GPT if no match is found
def generate_fallback_answer(query):
    prompt = f"The user asked: {query}. Please provide a helpful answer using your own knowledge."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that uses the knowledge base strictly to provide answers. Only reference external content when the KB does not have sufficient data."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )     
    return response['choices'][0]['message']['content'].strip()


### Routes ###

# Home Route to serve the main page
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/housing')
def housing():
    return render_template('housing.html')

# Fetch cities based on the environment
@app.route('/get-cities', methods=['POST'])
def get_cities():
    data = request.json
    environment = data.get('environment')
    cities = city_data.get(environment, [])
    return jsonify({"cities": cities})


@app.route('/preferences', methods=['POST'])
def preferences():
    data = request.get_json()
    app.logger.debug(f"Received data: {data}")

    city = data.get('city')
    budget = data.get('budget')
    housing = data.get('housing')

    if not city or not budget or not housing:
        app.logger.error('City, budget, and housing type must be provided.')
        return jsonify({'error': 'City, budget, and housing type must be provided.'}), 400

    try:
        budget = float(budget)
    except ValueError:
        app.logger.error('Invalid budget format.')
        return jsonify({'error': 'Invalid budget format.'}), 400

    try:
        city_info = city_df[city_df['city'] == city].iloc[0]
        city_lat = city_info['lat']
        city_lng = city_info['lng']
    except IndexError:
        app.logger.error(f'City {city} not found in the database.')
        return jsonify({'error': 'City not found.'}), 404

    token = get_idealista_token()
    if not token:
        app.logger.error('Failed to fetch Idealista token.')
        return jsonify({'error': 'Failed to fetch Idealista token.'}), 500

    rent_data = search_properties(token, city_lat, city_lng, 10000, budget, housing)
    if not rent_data:
        app.logger.error('Failed to fetch rent data.')
        return jsonify({'error': 'Failed to fetch rent data.'}), 500

    recommendations = format_recommendations(rent_data, city)
    app.logger.debug(f"Generated Recommendations HTML: {recommendations}")

    return jsonify({'recommendations': recommendations})

### Static Test Route
@app.route('/test-recommendations', methods=['GET'])
def test_recommendations():
    recommendations = """
    <div class="property-card">
        <img src="/static/default_property_image.jpg" alt="Property Image" width="150" height="150">
        <div class="property-info">
            <h4>Test Property</h4>
            <p>Price: €1234</p>
            <a href="#" target="_blank" class="view-property-btn">View Property</a>
        </div>
    </div>
    """
    return jsonify({'recommendations': recommendations})

# Fetch cities specifically from the external API for Portugal
@app.route('/fetch-portugal-cities', methods=['GET'])
def fetch_portugal_cities():
    url = "https://cost-of-living-and-prices.p.rapidapi.com/cities"
    
    headers = {
        "x-rapidapi-key": COST_OF_LIVING_API_KEY,
        "x-rapidapi-host": "cost-of-living-and-prices.p.rapidapi.com"
    }
    
    response = requests.get(url, headers=headers)
    city_data = response.json()

    # Filter cities in Portugal
    portugal_cities = [city for city in city_data['cities'] if city['country_name'] == 'Portugal']

    return jsonify({'cities': portugal_cities})

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/dashboard-data', methods=['POST'])
def dashboard_data():
    data = request.get_json()

    city = data.get('city')

    if not city:
        return jsonify({'error': 'City must be provided.'}), 400

    # Fetch cost of living data
    cost_data = fetch_cost_of_living(city, 'Portugal')

    if not cost_data:
        return jsonify({'error': 'Failed to fetch cost of living data.'}), 500

    # Send the prices data to the frontend
    return jsonify({'prices': cost_data.get('prices', [])})

# Route to serve the directory page
@app.route('/directory')
def directory():
    return render_template('directory.html', finance_offices=finance_offices, ss_offices=ss_offices, 
                           aima_offices=aima_offices, embassies_in_portugal=embassies_in_portugal, 
                           portuguese_embassies_abroad=portuguese_embassies_abroad)

# Endpoint to fetch all directory data
@app.route('/get-directory-data', methods=['GET'])
def get_directory_data():
    # Return all data as a JSON response
    return jsonify({
        'finance_offices': finance_offices,
        'ss_offices': ss_offices,
        'aima_offices': aima_offices,
        'embassies_in_portugal': embassies_in_portugal,
        'portuguese_embassies_abroad': portuguese_embassies_abroad
    })

# Routes for POI map

@app.route('/get-map-cities', methods=['GET'])
def get_map_cities():
    # Extract relevant columns from the DataFrame
    cities = city_df[['city', 'lat', 'lng']].to_dict(orient='records')
    
    # Return the cities data as JSON
    return jsonify(cities)

@app.route('/poi-map')
def poi_map():
    # Log when the route is accessed
    app.logger.debug("POI Map route accessed")

    # Fetch the API key and log whether it was found or not
    google_maps_api_key = os.getenv('GOOGLE_MAPS_API_KEY')
    if google_maps_api_key:
        app.logger.debug("Google Maps API Key found")
        app.logger.debug(f"Google Maps API Key: {google_maps_api_key}")
    else:
        app.logger.error("Google Maps API Key not found")

    # Try rendering the template and log if successful
    try:
        app.logger.debug("Rendering POI map template")
        return render_template('poi_map.html', google_maps_api_key=google_maps_api_key)
    except Exception as e:
        app.logger.error(f"Error rendering poi_map.html: {e}")
        return "Error rendering the POI Map page.", 500

### BOT Routes ###
# Route to handle doc-related queries
@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    query = data.get('query', '')

    # Detect the language of the query
    source_language = detect_language(query)

    # Translate the query to English if needed
    if source_language != 'en':
        query = translate_to_english(query)

    # Clean the query
    cleaned_query = clean_query(query)

    # Compute query embedding and find the best match using hybrid search
    best_entry, similarity_score = find_best_match(cleaned_query, kb_embeddings)

    if best_entry:
        # Use GPT to generate a natural answer from the best KB entry
        answer = generate_natural_answer_with_links(cleaned_query, best_entry)
    else:
        # Fall back to GPT if no good match is found
        answer = generate_fallback_answer(cleaned_query)

    # Translate the answer back to the original language if necessary
    if source_language != 'en':
        answer = translate_from_english(answer, source_language)

    return jsonify({'answer': answer})


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

