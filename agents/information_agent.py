import os
import sys
import json
import hashlib
import requests
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Add the parent directory to sys.path to allow imports from services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.weather_api import WeatherService
from services.car_rental_api import CarRentalService
from services.fuel_price_api import get_gas_price
from services.places_api import EnhancedPlacesService


def format_duration(seconds):
    """Format duration in seconds to a human-readable string (hours and minutes)."""
    if seconds is None:
        return "N/A"
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    duration_str = ""
    if hours > 0:
        duration_str += f"{hours} hour{'s' if hours > 1 else ''} "
    if minutes > 0:
        duration_str += f"{minutes} min{'s' if minutes > 1 else ''}"
    if not duration_str:  # Handle cases less than a minute
        duration_str = f"{sec} sec{'s' if sec > 1 else ''}"
    return duration_str.strip()


# Helper function for formatting distance
def format_distance(meters):
    """Format distance in meters to a string with kilometers and miles."""
    if meters is None:
        return "N/A"
    km = meters / 1000.0
    miles = meters / 1609.34
    return f"{km:.1f} km / {miles:.1f} miles"


class InformationAgent:
    def __init__(
        self,
        car_api_key="101c26fdb2msh34c9d61906a2fd7p17131ajsn68eb8cc9ec7f",
        llm_model_name="gpt-4o",
    ):
        """Initialize the InformationAgent - NO API KEYS NEEDED FOR MAPPING!"""
        self.rapidapi_key = car_api_key or os.getenv("RAPIDAPI_KEY")

        # OpenStreetMap Service - NO API KEY REQUIRED!
        print("🌍 Using OpenStreetMap")
        self.osm_service = self._create_osm_service()
        self.poi_api = self._create_poi_api()
        self.weather_service = WeatherService()
        self.car_rental_service = None
        if self.rapidapi_key and self.rapidapi_key != "YOUR_RAPIDAPI_KEY" and len(self.rapidapi_key) >= 30:
            try:
                self.car_rental_service = CarRentalService(rapidapi_key=self.rapidapi_key)
            except ValueError as e:
                print(f"Error initializing CarRentalService: {e}. Car rental will use mock data.")
                self.car_rental_service = None
        else:
            print("RAPIDAPI_KEY not configured correctly for CarRentalService. Car rental will use mock data.")

        try:
            self.llm = ChatOpenAI(model=llm_model_name, temperature=0.5)
        except Exception as e:
            print(f"Error initializing LLM ({llm_model_name}): {e}. LLM-dependent features might not work.")
            self.llm = None

        self.weather_summary_writer = self.llm
        self.llm_rerank_cache = {}

    def _create_osm_service(self):
        """Create OpenStreetMap service - NO API KEY REQUIRED!"""

        class OSMService:
            def __init__(self):
                self.nominatim_url = "https://nominatim.openstreetmap.org"
                self.overpass_url = "https://overpass-api.de/api/interpreter"
                self.osrm_url = "https://router.project-osrm.org"
                self.session = requests.Session()
                self.session.headers.update({'User-Agent': 'Vaiage-Travel-App/1.0'})

            def geocode(self, address: str):
                try:
                    params = {'q': address, 'format': 'json', 'limit': 5}
                    response = self.session.get(f"{self.nominatim_url}/search", params=params, timeout=10)
                    response.raise_for_status()
                    results = response.json()
                    formatted_results = []
                    for result in results:
                        formatted_result = {
                            'geometry': {'location': {'lat': float(result['lat']), 'lng': float(result['lon'])}},
                            'formatted_address': result.get('display_name', ''),
                        }
                        formatted_results.append(formatted_result)
                    return formatted_results
                except Exception as e:
                    print(f"OSM Geocoding error: {e}")
                    return []

            def places_nearby(self, location, radius=1000, type=None, **kwargs):
                try:
                    if isinstance(location, tuple):
                        lat, lng = location
                    else:
                        lat, lng = location['lat'], location['lng']

                    # Map Google types to OSM tags
                    type_mapping = {
                        'tourist_attraction': '["tourism"~"attraction|museum|gallery|zoo"]',
                        'restaurant': '["amenity"="restaurant"]',
                        'museum': '["tourism"="museum"]',
                        'park': '["leisure"="park"]',
                        'hospital': '["amenity"="hospital"]',
                        'bank': '["amenity"="bank"]',
                        'gas_station': '["amenity"="fuel"]',
                    }

                    osm_filter = type_mapping.get(type, '["tourism"]')

                    query = f"""
                    [out:json][timeout:25];
                    (
                      node{osm_filter}(around:{radius},{lat},{lng});
                      way{osm_filter}(around:{radius},{lat},{lng});
                    );
                    out center;
                    """

                    response = self.session.get(self.overpass_url, params={'data': query}, timeout=30)
                    response.raise_for_status()

                    data = response.json()
                    places = []

                    for element in data.get('elements', []):
                        try:
                            tags = element.get('tags', {})

                            if element['type'] == 'node':
                                lat, lng = element['lat'], element['lon']
                            elif 'center' in element:
                                lat, lng = element['center']['lat'], element['center']['lon']
                            else:
                                continue

                            place = {
                                'place_id': f"osm:{element['type']}/{element['id']}",
                                'name': tags.get('name', 'Unnamed Place'),
                                'geometry': {'location': {'lat': lat, 'lng': lng}},
                                'rating': 4.0,  # Default rating for OSM
                                'price_level': 2,  # Default price level
                                'types': self._extract_types(tags),
                                'vicinity': tags.get('addr:street', ''),
                                'user_ratings_total': 100,
                                'photos': [],
                            }
                            places.append(place)
                        except Exception:
                            continue

                    return {'results': places}
                except Exception as e:
                    print(f"OSM Places search error: {e}")
                    return {'results': []}

            def directions(self, origin: str, destination: str, mode: str = 'driving', **kwargs):
                try:
                    # Geocode origin and destination
                    start_results = self.geocode(origin)
                    end_results = self.geocode(destination)

                    if not start_results or not end_results:
                        return []

                    start = start_results[0]['geometry']['location']
                    end = end_results[0]['geometry']['location']

                    # Map mode to OSRM profile
                    profile_mapping = {
                        'driving': 'driving',
                        'walking': 'foot',
                        'bicycling': 'bike',
                        'transit': 'driving',  # Fallback
                    }
                    profile = profile_mapping.get(mode, 'driving')

                    coords = f"{start['lng']},{start['lat']};{end['lng']},{end['lat']}"
                    url = f"{self.osrm_url}/route/v1/{profile}/{coords}"

                    response = self.session.get(url, timeout=15)
                    response.raise_for_status()

                    data = response.json()
                    routes = []

                    for route in data.get('routes', []):
                        route_info = {
                            'legs': [
                                {
                                    'distance': {
                                        'text': f"{route['distance']/1000:.1f} km",
                                        'value': int(route['distance']),
                                    },
                                    'duration': {
                                        'text': f"{int(route['duration']/60)} mins",
                                        'value': int(route['duration']),
                                    },
                                    'start_address': origin,
                                    'end_address': destination,
                                }
                            ],
                            'overview_polyline': {'points': ''},
                            'summary': f"Route via {mode}",
                        }
                        routes.append(route_info)

                    return routes
                except Exception as e:
                    print(f"OSM Directions error: {e}")
                    return []

            def _extract_types(self, tags):
                """Extract place types from OSM tags"""
                types = []
                for key, value in tags.items():
                    if key in ['tourism', 'amenity', 'shop', 'leisure']:
                        types.append(value)
                return types or ['establishment']

        return OSMService()

    def _create_poi_api(self):
        """Create POI API wrapper for OpenStreetMap"""

        class OSMPoiApi:
            def __init__(self, osm_service):
                self.osm_service = osm_service

            def get_poi_details(self, place_id, **kwargs):
                return {'result': {'name': 'OSM Place', 'rating': 4.0, 'geometry': {'location': {'lat': 0, 'lng': 0}}}}

            def get_nearby_places(self, location, type, radius=1000, **kwargs):
                return self.osm_service.places_nearby(location, radius, type)

        return OSMPoiApi(self.osm_service)

    def _get_rerank_cache_key(self, user_prefs, attractions_ids_tuple, weather_summary):
        """Generate a cache key for LLM re-ranking based on user preferences, attraction IDs, and weather."""
        prefs_str = json.dumps(user_prefs, sort_keys=True)
        ids_str = json.dumps(attractions_ids_tuple, sort_keys=True)
        weather_str = weather_summary if weather_summary else ""
        hash_object = hashlib.sha256(f"{prefs_str}-{ids_str}-{weather_str}".encode())
        return hash_object.hexdigest()

    def _get_general_hobbies_for_city(self, city):
        """Generate general purpose hobbies relevant to a specific city"""
        city_lower = city.lower() if city else "unknown"

        # City-specific general hobbies mapping
        city_hobbies = {
            'paris': 'art, museums, architecture, cafes, history',
            'london': 'history, museums, theater, parks, culture',
            'new york': 'architecture, Broadway shows, museums, food, shopping',
            'tokyo': 'culture, temples, technology, food, gardens',
            'rome': 'history, art, architecture, ancient sites, food',
            'barcelona': 'architecture, art, beaches, food, culture',
            'amsterdam': 'canals, museums, cycling, history, culture',
            'berlin': 'history, museums, art, nightlife, culture',
            'sydney': 'beaches, harbor views, nature, museums, food',
            'san francisco': 'technology, bridges, hills, food, culture',
            'miami': 'beaches, art, nightlife, food, architecture',
            'las vegas': 'entertainment, shows, dining, nightlife, attractions',
            'dubai': 'modern architecture, luxury, shopping, desert, culture',
            'istanbul': 'history, mosques, bazaars, culture, architecture',
            'cairo': 'ancient history, museums, pyramids, culture, archaeology',
            'mumbai': 'culture, food, Bollywood, markets, history',
            'bangkok': 'temples, food, markets, culture, nightlife',
            'singapore': 'food, gardens, modern architecture, culture, shopping',
            'hong kong': 'skyline views, food, shopping, culture, temples',
            'seoul': 'technology, food, culture, shopping, temples',
        }

        # Check for exact matches first
        if city_lower in city_hobbies:
            return city_hobbies[city_lower]

        # Check for partial matches (e.g., "New York City" contains "new york")
        for key, hobbies in city_hobbies.items():
            if key in city_lower or city_lower in key:
                return hobbies

        # Default general hobbies for unknown cities
        return 'sightseeing, culture, history, food, architecture'

    def _create_llm_rerank_prompt(self, user_prefs, attractions_for_llm, weather_summary):
        """Create a prompt for the LLM to re-rank attractions."""
        attractions_str = json.dumps(attractions_for_llm, indent=2, ensure_ascii=False)
        user_prefs_str = json.dumps(user_prefs, indent=2, ensure_ascii=False)
        weather_str = weather_summary if weather_summary else "No specific weather summary provided."

        # Get hobbies, or generate city-specific general hobbies if not provided
        hobbies_raw = user_prefs.get('hobbies', '')

        # Handle both string and list cases for hobbies
        if isinstance(hobbies_raw, list):
            hobbies = ', '.join(str(h) for h in hobbies_raw if h).strip()
        elif isinstance(hobbies_raw, str):
            hobbies = hobbies_raw.strip()
        else:
            hobbies = str(hobbies_raw).strip() if hobbies_raw else ''

        if not hobbies:
            city = user_prefs.get('city', '')
            hobbies = self._get_general_hobbies_for_city(city)

        prompt = f"""
        You are an expert travel recommender. Your task is to rank the provided list of attractions based on the user's preferences, the details of each attraction, and the current weather summary.

        User Preferences:
        {user_prefs_str}

        Weather Summary for the trip period:
        {weather_str}

        Attractions List (with details including their original 'id', 'name', 'category', 'estimated_duration', 'price_level', 'rating', and a brief 'description' if available):
        {attractions_str}

        Please consider the following factors for ranking:
        1.  **User Hobbies & Interests**: Match with user's hobbies (e.g., '{hobbies}').
        2.  **User Health & Accessibility**: Consider user's health (e.g., '{user_prefs.get('health', 'good')}') and attraction accessibility.
        3.  **Suitability for Children**: If traveling with kids (e.g., Kids: '{user_prefs.get('kids', 'no')}'), prioritize child-friendly options.
        4.  **Budget Constraints**: Align with budget (e.g., '{user_prefs.get('budget', 'medium')}').
        5.  **Weather Impact**: Prioritize indoor/outdoor activities based on the weather.
        6.  **Category Balance**: Aim for diversity in top recommendations. Also filter out duplicate attractions that are essentially the same place but listed differently.

        Return a JSON list of attraction IDs, ranked from MOST to LEAST recommended.
        The output MUST be a valid JSON list of strings (attraction IDs). For example:
        ["id1", "id2", "id3"]

        Only return the JSON list of IDs. Do not include any other text or explanation.
        """
        return prompt

    def _rerank_attractions_with_llm(self, attractions_list: list, user_prefs: dict, weather_summary: str = None):
        """Re-rank attractions using an LLM based on user preferences and weather."""
        if not self.llm:
            print("LLM not available for re-ranking. Returning original list.")
            return attractions_list
        if not attractions_list:
            return []
        if not user_prefs:
            print("User preferences not provided for LLM re-ranking. Returning original list.")
            return attractions_list

        attractions_for_llm = []
        for attr in attractions_list:
            attractions_for_llm.append(
                {
                    "id": attr.get("id"),
                    "name": attr.get("name"),
                    "category": attr.get("category"),
                    "description": attr.get("description", attr.get("name", "No description available.")),
                    "estimated_duration": attr.get("estimated_duration"),
                    "price_level": attr.get("price_level"),
                    "rating": attr.get("rating"),
                }
            )

        attraction_ids_tuple = tuple(sorted([attr.get('id', '') for attr in attractions_for_llm]))
        cache_key = self._get_rerank_cache_key(user_prefs, attraction_ids_tuple, weather_summary)

        if cache_key in self.llm_rerank_cache:
            print(f"Returning cached LLM re-ranking for key: {cache_key}")
            ranked_ids = self.llm_rerank_cache[cache_key]
        else:
            prompt_str = self._create_llm_rerank_prompt(user_prefs, attractions_for_llm, weather_summary)
            messages = [
                SystemMessage(
                    content="You are an expert travel recommender. Your goal is to rank attractions based on user preferences, attraction details, and weather conditions. Ensure a good balance of attraction categories if appropriate."
                ),
                HumanMessage(content=prompt_str),
            ]
            try:
                print(
                    f"[INFO_AGENT_LLM] Requesting LLM re-ranking for {len(attractions_for_llm)} items. Cache key: {cache_key}"
                )
                response = self.llm.invoke(messages)
                ranked_ids = []
                try:
                    content_str = response.content
                    if hasattr(response.content, 'strip'):
                        # Handle string content
                        if content_str.strip().startswith("```json"):
                            content_str = content_str.strip()[7:]
                            if content_str.strip().endswith("```"):
                                content_str = content_str.strip()[:-3]
                        ranked_ids_data = json.loads(content_str.strip())
                    else:
                        # Handle non-string content (list or dict)
                        ranked_ids_data = response.content

                    if isinstance(ranked_ids_data, list) and all(isinstance(id_val, str) for id_val in ranked_ids_data):
                        ranked_ids = ranked_ids_data
                    else:
                        print(f"[INFO_AGENT_LLM_ERROR] LLM output was not a list of strings: {ranked_ids_data}")
                        raise ValueError("LLM output not in expected list of strings format.")

                except (json.JSONDecodeError, ValueError) as e:
                    print(f"[INFO_AGENT_LLM_ERROR] Parsing LLM re-ranking response: {e}")
                    return attractions_list

                self.llm_rerank_cache[cache_key] = ranked_ids
                print(f"[INFO_AGENT_LLM] Cached LLM re-ranking for key: {cache_key}")

            except Exception as e:
                print(f"[INFO_AGENT_LLM_ERROR] Calling LLM for re-ranking: {e}")
                return attractions_list

        id_to_attraction_map = {attr['id']: attr for attr in attractions_list}
        ordered_attractions = []
        seen_ids = set()
        for id_ in ranked_ids:
            if id_ in id_to_attraction_map and id_ not in seen_ids:
                ordered_attractions.append(id_to_attraction_map[id_])
                seen_ids.add(id_)

        for attr in attractions_list:
            if attr.get('id') not in seen_ids:  # Add any attractions not in the LLM's ranked list
                ordered_attractions.append(attr)

        print(f"[INFO_AGENT_LLM] Re-ranked list size: {len(ordered_attractions)}")
        return ordered_attractions

    def city2geocode(self, city: str):
        """Convert city name to geographic coordinates (latitude and longitude)."""
        try:
            coordinates = self.osm_service.geocode(city)
            if not coordinates:
                return None
            return coordinates[0]['geometry']['location']
        except Exception as e:
            print(f"Error in city2geocode for '{city}': {e}")
            return None

    def get_attractions(
        self,
        lat: float,
        lng: float,
        user_prefs: dict,
        weather_summary: str = None,
        number: int = 20,
        poi_type: str = "tourist_attraction",
        sort_by: str = "rating",
        radius: int = 10000,
    ):
        """Get a list of attractions for a given location, using enhanced multi-source search with images."""

        # Initialize enhanced places service
        if not hasattr(self, 'enhanced_places_service'):
            self.enhanced_places_service = EnhancedPlacesService()

        city = user_prefs.get('city', 'Unknown City')

        try:
            print(f"[INFO_AGENT] Searching for attractions in {city} using enhanced multi-source search...")

            # Use enhanced places service to get attractions from multiple sources
            enhanced_attractions = self.enhanced_places_service.search_attractions(
                city=city,
                lat=lat,
                lng=lng,
                categories=['tourist_attraction', 'museum', 'park', 'landmark', 'gallery'],
                limit=number * 2,  # Get more for better LLM ranking
            )

            print(f"[INFO_AGENT] Found {len(enhanced_attractions)} attractions from enhanced search.")

            # Convert to format expected by the rest of the system
            formatted_attractions = []
            for attraction in enhanced_attractions:
                try:
                    # Ensure all required fields are present
                    formatted_attraction = {
                        'id': attraction.get('id', f"enhanced_{hash(attraction.get('name', 'unknown'))}"),
                        'name': attraction.get('name', 'Unknown Place'),
                        'rating': float(attraction.get('rating', 3.5)),
                        'user_ratings_total': attraction.get('user_ratings_total', 50),
                        'price_level': attraction.get('price_level', 1),
                        'opening_hours': attraction.get('opening_hours'),
                        'address': attraction.get('address', ''),
                        'location': {
                            'lat': (
                                float(attraction['location']['lat'])
                                if attraction.get('location', {}).get('lat')
                                else lat
                            ),
                            'lng': (
                                float(attraction['location']['lng'])
                                if attraction.get('location', {}).get('lng')
                                else lng
                            ),
                        },
                        'category': attraction.get('category', 'tourist_attraction'),
                        'types': attraction.get('types', ['tourist_attraction']),
                        'estimated_duration': attraction.get('estimated_duration', 2.0),
                        'website': attraction.get('website', ''),
                        'description': attraction.get('description', f'Popular attraction in {city}'),
                        'photo_references': [],
                        'image_url': None,
                        'source': attraction.get('source', 'enhanced'),
                        'photos': attraction.get('photos', []),  # New: Include photos from enhanced search
                    }

                    # Set image_url from photos if available
                    if formatted_attraction['photos']:
                        formatted_attraction['image_url'] = formatted_attraction['photos'][0].get('url')

                    formatted_attractions.append(formatted_attraction)

                except Exception as e:
                    print(f"[ERROR] Error formatting attraction {attraction.get('name', 'Unknown')}: {e}")
                    continue

            print(f"[INFO_AGENT] Formatted {len(formatted_attractions)} attractions for ranking.")

            # Apply LLM ranking if available
            if user_prefs and self.llm and formatted_attractions:
                print(f"[INFO_AGENT] Re-ranking {len(formatted_attractions)} attractions with LLM.")
                try:
                    llm_ranked_pois = self._rerank_attractions_with_llm(
                        formatted_attractions, user_prefs, weather_summary or ""
                    )
                    return llm_ranked_pois[:number]
                except Exception as e:
                    print(f"[ERROR] LLM ranking failed: {e}. Using default sorting.")

            # Fall back to sorting by rating and source quality
            print(f"[INFO_AGENT] Using default sorting by rating and source quality.")

            # Sort by rating, then by source quality
            formatted_attractions.sort(
                key=lambda x: (
                    x.get('rating', 0),
                    1 if x.get('source') == 'foursquare' else 2 if x.get('source') == 'opentripmap' else 3,
                ),
                reverse=True,
            )

            return formatted_attractions[:number]

        except Exception as e:
            print(f"[ERROR] Enhanced attraction search failed: {e}")
            print("[INFO_AGENT] Falling back to basic OSM search...")

            # Fallback to original OSM method if enhanced search fails
            return self._get_attractions_fallback(
                lat, lng, user_prefs, weather_summary, number, poi_type, sort_by, radius
            )

    def _get_attractions_fallback(self, lat, lng, user_prefs, weather_summary, number, poi_type, sort_by, radius):
        """Fallback method using original OSM search"""
        location = (lat, lng)
        initial_fetch_limit = 30

        try:
            results = self.osm_service.places_nearby(
                location=location, radius=radius, type=poi_type, language='en'
            ).get('results', [])
        except Exception as e:
            print(f"Error fetching places_nearby: {e}")
            results = []

        initial_pois = []
        print(f"[INFO_AGENT] Fallback: Fetched {len(results)} raw places.")

        for place in results[:initial_fetch_limit]:
            pid = place.get('place_id')
            if not pid:
                continue
            try:
                place_types_list = place.get('types', ["unknown"])
                primary_category_from_place = place_types_list[0] if place_types_list else "unknown"

                details_response = self.poi_api.get_poi_details(place_id=pid, fields=[])
                details = details_response.get('result', {})
                if not details:
                    continue

                raw_location = details.get('geometry', {}).get('location', {})
                location_data = {'lat': raw_location.get('lat', lat), 'lng': raw_location.get('lng', lng)}

                description = details.get('editorial_summary', {}).get('overview', '')
                if not description:
                    description = details.get('name', 'No description available.')

                initial_pois.append(
                    {
                        'id': pid,
                        'name': details.get('name', 'Unknown Place'),
                        'rating': details.get('rating', 3.5),
                        'user_ratings_total': details.get('user_ratings_total', 25),
                        'price_level': details.get('price_level', 1),
                        'opening_hours': details.get('opening_hours', {}).get('weekday_text'),
                        'address': details.get('formatted_address', ''),
                        'location': location_data,
                        'category': primary_category_from_place,
                        'types': place_types_list,
                        'estimated_duration': self.estimate_duration(primary_category_from_place, details),
                        'website': details.get('website', ''),
                        'description': description,
                        'photo_references': [],
                        'image_url': None,
                    }
                )
            except Exception as e:
                print(f"[ERROR] Exception during fallback processing: {e}")
                continue

        if sort_by == 'rating':
            initial_pois.sort(key=lambda x: x.get('rating', 0), reverse=True)

        return initial_pois[:number]

    def estimate_duration(self, category, details):
        """
        Estimate the duration for a given category and details.
        Returns duration in hours.
        """
        category_duration = {
            'restaurant': 2,
            'museum': 2,
            'park': 2,
            'tourist_attraction': 2,
            'night_club': 3,
            'shopping_mall': 3,
            'zoo': 3,
            'amusement_park': 6,
        }

        # Default duration if category is not found
        default_duration = 2

        # Get duration based on category
        duration = category_duration.get(category, default_duration)

        # Adjust duration based on rating
        rating = details.get('rating', 0)
        if rating > 4.5:
            duration *= 1.5
        elif rating < 3:
            duration *= 0.75

        return duration

    def plan_routes(self, origin: str, destination: str):
        """
        Route Planning (Simple A to B for multiple modes).

        Args:
            origin: Starting point (address, place name, or lat/lng tuple/dict)
            destination: End point (address, place name, or lat/lng tuple/dict)

        Returns:
            List of dictionaries, each representing a travel mode, or an empty list.
            Example format:
            [
                {
                    'mode': str,                # e.g., 'driving', 'transit'
                    'distance': str,            # Formatted distance text (e.g., "10.2 miles")
                    'duration': str,            # Formatted duration text (e.g., "25 mins")
                    'distance_meters': int,     # Raw distance in meters
                    'duration_seconds': int,    # Raw duration in seconds
                    'fare': str | None          # Estimated fare text (mostly for transit)
                },
                ...
            ]
        """
        modes = ['driving', 'walking', 'bicycling', 'transit']
        routes = []
        for mode in modes:
            try:
                # Using 'en' for consistent address resolution and international compatibility
                directions = self.osm_service.directions(origin, destination, mode=mode, language='en')
                if not directions:
                    continue

                # Ensure legs exist and are not empty
                if not directions[0].get('legs'):
                    print(f"Warning: Route for mode '{mode}' from '{origin}' to '{destination}' lacks 'legs' data.")
                    continue
                leg = directions[0]['legs'][0]

                # Ensure distance and duration exist in the leg
                if 'distance' not in leg or 'duration' not in leg:
                    print(
                        f"Warning: Leg for mode '{mode}' from '{origin}' to '{destination}' lacks distance or duration data."
                    )
                    continue

                info = {
                    'mode': mode,
                    'distance': leg['distance']['text'],
                    'duration': leg['duration']['text'],
                    'distance_meters': leg['distance']['value'],  # Raw distance in meters
                    'duration_seconds': leg['duration']['value'],  # Raw duration in seconds
                }
                # Add fare info if available
                if 'fare' in directions[0]:
                    info['fare'] = directions[0]['fare'].get('text')
                routes.append(info)
            except Exception as e:
                print(f"An unexpected error occurred during route planning for mode '{mode}': {e}")
        return routes

    def plan_with_waypoints(
        self, origin: str, destination: str, waypoints: list, mode: str = 'driving', departure_time: datetime = None
    ):
        """
        Plans an optimized route visiting a list of waypoints between an origin and destination.
        Uses OpenStreetMap and OSRM routing engine for route optimization.

        Args:
            origin: Starting point (address, place name, or lat/lng tuple/dict)
            destination: End point (address, place name, or lat/lng tuple/dict)
            waypoints: List of intermediate points (list of strings, lat/lng tuples/dicts)
            mode: Travel mode (default: 'driving'). Optimization works best for 'driving'.
            departure_time: Optional datetime object (default: now) for traffic estimation.

        Returns:
            Dictionary with optimized route details, or None if no route is found.
            Example format:
            {
                'path_sequence': List[str],         # List of addresses in optimized order (Origin, WptX, WptY,..., Dest)
                'waypoint_original_indices': List[int], # Order original waypoints were visited (0-based index)
                'total_duration_text': str,         # Formatted total duration (e.g., "2 hours 30 mins")
                'total_duration_seconds': int,      # Raw total duration in seconds
                'total_duration_in_traffic_text': str | None, # Formatted duration with traffic (if available)
                'total_duration_in_traffic_seconds': int | None, # Raw duration with traffic (if available)
                'total_distance_text': str,         # Formatted total distance (e.g., "150.5 km / 93.5 miles")
                'total_distance_meters': int,       # Raw total distance in meters
                'fare': str | None                  # Estimated fare text (rare for driving)
            }
        """
        # Handle empty waypoints list by falling back to simple A-B route planning
        if not waypoints:
            print("Warning: No waypoints provided. Calling standard plan_routes for A-B.")
            simple_route_options = self.plan_routes(origin, destination)
            # Find the driving route from the simple options
            driving_route = next((r for r in simple_route_options if r['mode'] == 'driving'), None)
            if driving_route:
                # Addresses from API are resolved; use original input if unavailable in fallback
                start_addr = origin if isinstance(origin, str) else f"Coord: {origin}"
                end_addr = destination if isinstance(destination, str) else f"Coord: {destination}"
                return {
                    'path_sequence': [start_addr, end_addr],  # Simplified path
                    'waypoint_original_indices': [],
                    'total_duration_text': driving_route['duration'],
                    'total_duration_seconds': driving_route['duration_seconds'],
                    'total_duration_in_traffic_text': None,  # Not available from simple plan_routes call here
                    'total_duration_in_traffic_seconds': None,
                    'total_distance_text': driving_route['distance'],
                    'total_distance_meters': driving_route['distance_meters'],
                    'fare': driving_route.get('fare'),
                }
            else:
                print(f"Could not find a driving route from {origin} to {destination} in fallback.")
                return None

        # Set departure time to now if not specified
        if departure_time is None:
            departure_time = datetime.now()

        print(f"Planning optimized route: {origin} -> Waypoints -> {destination} for mode '{mode}'")

        try:
            # Call OpenStreetMap OSRM API for route optimization
            # Uses OSRM routing engine with waypoint optimization
            directions_result = self.osm_service.directions(
                origin,
                destination,
                waypoints=waypoints,
                optimize_waypoints=True,  # Key parameter for optimization
                mode=mode,
                departure_time=departure_time,
                language='en',
            )

            # Check if API returned a valid result
            if not directions_result:
                print("No route found for the given points and mode.")
                return None

            # Get the first recommended route
            route = directions_result[0]
            # 'legs' are the segments between points (origin->wpt1, wpt1->wpt2, ..., wptN->dest)
            legs = route['legs']

            # Calculate total duration and distance by summing up values from each leg
            total_duration_sec = sum(leg['duration']['value'] for leg in legs)
            total_distance_m = sum(leg['distance']['value'] for leg in legs)

            # Calculate duration with traffic if available for all legs
            total_duration_traffic_sec = None
            if all('duration_in_traffic' in leg for leg in legs):
                total_duration_traffic_sec = sum(leg['duration_in_traffic']['value'] for leg in legs)

            # Reconstruct the path sequence using resolved addresses from the API response
            # Start address is from the first leg; end addresses are from each leg
            path_sequence = [legs[0]['start_address']] + [leg['end_address'] for leg in legs]

            # Get the optimized order of the *original* waypoints list (0-based indices)
            optimized_indices = route.get('waypoint_order', [])

            # Prepare the result dictionary
            result = {
                'path_sequence': path_sequence,
                'waypoint_original_indices': optimized_indices,
                'total_duration_text': format_duration(total_duration_sec),
                'total_duration_seconds': total_duration_sec,
                'total_distance_text': format_distance(total_distance_m),
                'total_distance_meters': total_distance_m,
                'fare': route.get('fare', {}).get('text'),  # Extract fare text if present
            }

            # Add traffic duration details if calculated
            if total_duration_traffic_sec is not None:
                result['total_duration_in_traffic_text'] = format_duration(total_duration_traffic_sec)
                result['total_duration_in_traffic_seconds'] = total_duration_traffic_sec
            else:
                result['total_duration_in_traffic_text'] = None
                result['total_duration_in_traffic_seconds'] = None

            return result

        except Exception as e:
            print(f"An unexpected error occurred during optimized route planning: {e}")
            # Optionally re-raise or log the full traceback for debugging
            # import traceback
            # traceback.print_exc()
            return None

    def get_weather(self, lat: float, lng: float, start_date: str, duration: int, summary: bool = True):
        """
        Weather Forecast.

        Args:
            lat: Latitude
            lng: Longitude
            start_date: Start date (YYYY-MM-DD)
            duration: Number of days
            summary: Whether to include an LLM-generated summary.

        Returns:
            Dictionary containing detailed weather forecast and an optional summary.
            Example:
            {
                'detailed_forecast': [
                    {
                        "date": "2023-04-18",
                        "max_temp": "22 °C",
                        "min_temp": "15 °C",
                        "precipitation": "0 mm",
                        "wind_speed": "12 km/h",
                        "precipitation_probability": "5%",
                        "uv_index": "7"
                    },
                    ...
                ],
                'summary': "Concise weather summary..." # or None
            }
        """
        # Get detailed weather data first
        weather_data = self.weather_service.get_weather(lat, lng, start_date, duration)

        # If no weather data, return empty result
        if not weather_data:
            return {'detailed_forecast': [], 'summary': None}

        # Create result dictionary with detailed forecast
        result = {'detailed_forecast': weather_data, 'summary': None}

        # Generate summary if requested and LLM available
        if summary and self.llm and self.weather_summary_writer:
            weather_info = json.dumps(weather_data, indent=2)
            prompt = f"""
            Summarize the following weather forecast in a concise paragraph (max 100 words).
            Include key information about temperature ranges, precipitation, and any notable weather conditions.
            Also mention any precautions travelers should take based on the forecast.
            
            Weather data:
            {weather_info}
            """

            messages = [
                SystemMessage(
                    content="You are a helpful weather assistant that provides concise summaries of weather forecasts for travelers."
                ),
                HumanMessage(content=prompt),
            ]

            try:
                result['summary'] = self.weather_summary_writer.invoke(messages)
            except Exception as e:
                print(f"Error generating weather summary: {e}")
                result['summary'] = None

        return result

    def search_car_rentals(
        self,
        location: str,
        start_date: str,
        end_date: str,
        driver_age: int = 30,
        min_price: float = None,
        max_price: float = None,
        top_n: int = 5,
    ):
        """
        Car Rental Search.

        Args:
            location: Location (city name)
            start_date: Pickup date (YYYY-MM-DD)
            end_date: Return date (YYYY-MM-DD)
            driver_age: Driver's age (default: 30)
            min_price: Minimum price (optional)
            max_price: Maximum price (optional)
            top_n: Number of results to return (default: 5)

        Returns:
            Top N car rental options, including car type, price, pickup/return locations, links, etc.
            Uses mock data if API is not configured or fails.
            Example:
            [
                {
                    "car_model": "Mitsubishi Mirage",
                    "car_group": "Economy",
                    "price": 332.29,
                    "currency": "USD",
                    "pickup_location_name": "Los Angeles International Airport",
                    "supplier_name": "Enterprise",
                    "image_url": "https://cdn.rcstatic.com/images/car_images/web/mitsubishi/mirage_lrg.png"
                },
                ...
            ]
        """
        try:
            # Get location coordinates
            location_data = self.city2geocode(location)
            if not location_data:
                return self._get_mock_car_data(top_n)

            # Parse dates
            pickup_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
            dropoff_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

            # Format dates and times for API
            pickup_date = pickup_date_obj.strftime("%Y-%m-%d")
            pickup_time = "10:00:00"  # Default pickup time
            dropoff_date = dropoff_date_obj.strftime("%Y-%m-%d")
            dropoff_time = "10:00:00"  # Default dropoff time

            # Call the car rental service
            cars = self.car_rental_service.find_available_cars(
                pickup_lat=location_data['lat'],
                pickup_lon=location_data['lng'],
                pickup_date=pickup_date,
                pickup_time=pickup_time,
                dropoff_lat=location_data['lat'],
                dropoff_lon=location_data['lng'],
                dropoff_date=dropoff_date,
                dropoff_time=dropoff_time,
                currency_code="USD",
                driver_age=driver_age,
            )

            # Filter by price if needed
            if cars and min_price is not None:
                cars = [c for c in cars if c.get('price', 0) >= min_price]
            if cars and max_price is not None:
                cars = [c for c in cars if c.get('price', 0) <= max_price]

            # Return top N results or mock data if API returned nothing
            return cars[:top_n] if cars else self._get_mock_car_data(top_n)

        except Exception as e:
            print(f"Error in search_car_rentals: {str(e)}")
            return self._get_mock_car_data()

    def _get_mock_car_data(self, top_n: int = 5):
        """Returns a list of mock car rental data."""
        mock_cars = [
            {
                "car_model": "Toyota Corolla",
                "car_group": "Economy",
                "price": 299.99,
                "currency": "USD",
                "pickup_location_name": "Sample Airport",
                "supplier_name": "Hertz",
                "image_url": "https://example.com/corolla.jpg",
            },
            {
                "car_model": "Honda Civic",
                "car_group": "Compact",
                "price": 349.99,
                "currency": "USD",
                "pickup_location_name": "Sample Airport",
                "supplier_name": "Avis",
                "image_url": "https://example.com/civic.jpg",
            },
            {
                "car_model": "Ford Mustang",
                "car_group": "Sports",
                "price": 599.99,
                "currency": "USD",
                "pickup_location_name": "Sample Airport",
                "supplier_name": "Enterprise",
                "image_url": "https://example.com/mustang.jpg",
            },
            {
                "car_model": "BMW 3 Series",
                "car_group": "Luxury",
                "price": 799.99,
                "currency": "USD",
                "pickup_location_name": "Sample Airport",
                "supplier_name": "Sixt",
                "image_url": "https://example.com/bmw.jpg",
            },
            {
                "car_model": "Mercedes-Benz C-Class",
                "car_group": "Premium",
                "price": 899.99,
                "currency": "USD",
                "pickup_location_name": "Sample Airport",
                "supplier_name": "Europcar",
                "image_url": "https://example.com/mercedes.jpg",
            },
        ]
        return mock_cars[:top_n]

    def search_nearby_places(self, lat: float, lng: float, radius: int = 500):
        """Search for nearby restaurants and provide their details.

        Args:
            lat (float): Latitude
            lng (float): Longitude
            radius (int): Search radius (meters)

        Returns:
            dict: Dictionary containing information about nearby restaurants (top 3 by rating).
                  Returns mock data if API calls fail.
        """
        try:
            # Check if POI API is available
            if not self.poi_api:
                raise Exception("POI API is not initialized")

            # Search for nearby restaurants
            restaurants_result = self.poi_api.get_nearby_places(location=(lat, lng), type='restaurant', radius=radius)

            # Process restaurant information
            processed_restaurants = []
            # Sort all fetched restaurants by rating (descending) before further processing
            # Handle cases where rating might be missing by defaulting to 0 for sorting
            all_fetched_restaurants = restaurants_result.get('results', [])
            all_fetched_restaurants.sort(key=lambda p: p.get('rating', 0), reverse=True)

            for place in all_fetched_restaurants[:3]:  # Only take the top 3 after sorting
                try:
                    # Get detailed information
                    place_details = self.poi_api.get_poi_details(
                        place_id=place['place_id'],
                        fields=['name', 'rating', 'price_level', 'formatted_address', 'photo', 'type', 'geometry'],
                    )

                    if not place_details or 'result' not in place_details:
                        continue

                    place_details = place_details['result']

                    # OSM doesn't have photos like Google Maps - skip photo processing
                    photos = []

                    restaurant = {
                        'name': place_details.get('name', 'Unknown Restaurant'),
                        'type': 'restaurant',
                        'rating': place_details.get('rating', 0),
                        'price_level': place_details.get('price_level', 0),
                        'address': place_details.get('formatted_address', 'Unknown address'),
                        'photos': photos,  # Empty for OSM
                        'features': self._get_restaurant_features(
                            place
                        ),  # Use type info from the original search result
                    }
                    processed_restaurants.append(restaurant)
                except Exception as e:
                    print(f"Error processing restaurant info: {str(e)}")
                    continue

            return {'restaurants': processed_restaurants}

        except Exception as e:
            print(f"Error searching nearby places: {str(e)}")
            # Return mock data
            return {
                'restaurants': [
                    {
                        'name': 'Sample Restaurant',
                        'type': 'restaurant',
                        'rating': 4.5,
                        'price_level': 2,
                        'address': 'Sample Address',
                        'photos': [{'url': 'https://example.com/photo1.jpg', 'width': 800, 'height': 600}],
                        'features': 'Cuisine: Chinese, Western',
                    }
                ]
            }

    def _get_restaurant_features(self, place):
        """Get restaurant features (cuisine types) from place types."""
        features = []
        if 'types' in place:
            if 'chinese_restaurant' in place['types']:
                features.append('Chinese')
            if 'japanese_restaurant' in place['types']:
                features.append('Japanese')
            if 'italian_restaurant' in place['types']:
                features.append('Italian')
            if 'french_restaurant' in place['types']:
                features.append('French')
        return ', '.join(features) if features else 'Cuisine'

    def get_fuel_price(self, location: str):
        """
        Get fuel prices for a specific location.

        Args:
            location (str): Location name (city).

        Returns:
            float: Fuel price in USD per gallon, or None if not found.
        """
        try:
            return get_gas_price(location)
        except Exception as e:
            print(f"Error getting fuel prices: {str(e)}")
            return None
