import json
from langchain_openai import ChatOpenAI


class RecommendAgent:
    def __init__(self, model_name="gpt-4o"):
        """Initialize RecommendAgent with AI model for personalized recommendations"""
        self.model = ChatOpenAI(model=model_name, temperature=0.7)

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

    def recommend_core_attractions(self, user_prefs, attractions):
        """Recommend core attractions based on user preferences"""
        # Extract preferences
        budget = user_prefs.get('budget', 'medium').lower()
        has_kids = user_prefs.get('kids', 'no').lower() == 'yes'
        health = user_prefs.get('health', 'good').lower()

        # Get hobbies, or generate city-specific general hobbies if not provided
        hobbies_raw = user_prefs.get('hobbies', '')

        # Handle both string and list cases for hobbies
        if isinstance(hobbies_raw, list):
            hobbies = ', '.join(str(h) for h in hobbies_raw if h).strip().lower()
        elif isinstance(hobbies_raw, str):
            hobbies = hobbies_raw.strip().lower()
        else:
            hobbies = str(hobbies_raw).strip().lower() if hobbies_raw else ''

        if not hobbies:
            city = user_prefs.get('city', '')
            hobbies = self._get_general_hobbies_for_city(city).lower()

        # Filter attractions based on preferences
        filtered_attractions = []
        for attraction in attractions:
            # Skip if any required field is None
            if not all(
                [
                    attraction.get('price_level') is not None,
                    attraction.get('estimated_duration') is not None,
                    attraction.get('category') is not None,
                ]
            ):
                continue

            # Budget filter
            if budget == 'low' and attraction['price_level'] > 2:
                continue
            elif budget == 'medium' and attraction['price_level'] > 3:
                continue
            elif budget == 'high' and attraction['price_level'] > 4:
                continue

            # Family-friendly filter
            if has_kids and not attraction.get('family_friendly', False):
                continue

            # Health considerations
            if health == 'limited' and attraction.get('accessibility') == 'limited':
                continue

            # Hobbies match
            if hobbies:
                category = attraction.get('category', '').lower()
                if any(hobby in category for hobby in hobbies.split(',')):
                    filtered_attractions.append(attraction)
                continue

            filtered_attractions.append(attraction)

        # Sort by rating and duration
        filtered_attractions.sort(
            key=lambda x: (
                x.get('rating', 0) or 0,  # Handle None values for rating
                -(x.get('estimated_duration', 0) or 0),  # Handle None values for duration, sort descending
            ),
            reverse=True,
        )

        # Return top recommendations
        return filtered_attractions[:10]

    def _create_recommendation_prompt(self, user_prefs, attractions):
        """Create prompt for the LLM to rank attractions"""
        attractions_str = json.dumps(attractions, indent=2)
        user_prefs_str = json.dumps(user_prefs, indent=2)
        return f"""
        Given the following user preferences and attractions, rank the attractions from most suitable to least suitable.
        
        User preferences:
        {user_prefs_str}
        
        Attractions:
        {attractions_str}
        
        Consider the following factors with strong emphasis on user preferences:
        1. Match between user's hobbies and attraction categories - this should be the PRIMARY factor in ranking
        2. Physical accessibility based on user's health status - PRIORITIZE attractions that accommodate the user's health condition
        3. Suitability for children if the user is traveling with kids
        4. Budget constraints
        5. Variety of attraction categories to provide a balanced experience
        
        For users with health considerations, ensure attractions are accessible and not overly strenuous.
        For users with specific hobbies, prioritize attractions that directly match these interests.
        
        Return a list of attraction IDs, ranked from most to least recommended, in this format:
        [
          "attraction_id_1",
          "attraction_id_2",
          ...
        ]
        """

    def _score_attractions(self, user_prefs, attractions):
        """Score attractions based on user preferences (fallback method if LLM ranking is not used or fails)"""
        scored_attractions = []

        for attraction in attractions:
            score = 0

            # Score based on category match
            if "hobbies" in user_prefs and attraction.get("category") in user_prefs["hobbies"]:
                score += 3

            # Score based on health considerations
            if "health" in user_prefs:
                if user_prefs["health"] == "excellent":
                    score += 1  # Any attraction is fine
                elif user_prefs["health"] == "good" and attraction.get("estimated_duration", 3) <= 3:
                    score += 1  # Prefer shorter duration attractions
                elif user_prefs["health"] == "limited" and attraction.get("estimated_duration", 3) <= 2:
                    score += 1  # Strongly prefer shorter attractions

            # Score based on budget
            if "budget" in user_prefs:
                budget_level = {"low": 1, "medium": 2, "high": 3}.get(user_prefs["budget"], 2)
                if attraction.get("price_level", 2) <= budget_level:
                    score += 1

            # Score based on kids
            if user_prefs.get("kids", False) and attraction.get("kid_friendly", False):
                score += 1

            scored_attractions.append((attraction["id"], score, attraction))

        scored_attractions.sort(key=lambda x: x[1], reverse=True)

        # Return sorted attractions (full objects)
        return [item[2] for item in scored_attractions]

    def generate_map_data(self, attractions):
        """Generate map data for frontend visualization"""
        map_data = []

        for attraction in attractions:
            map_data.append(
                {
                    "id": attraction["id"],
                    "name": attraction["name"],
                    "lat": attraction["location"]["lat"],
                    "lng": attraction["location"]["lng"],
                    "category": attraction["category"],
                }
            )

        return map_data

    def get_attraction_details(self, attraction_id, attractions):
        """Get detailed information about a specific attraction"""
        for attraction in attractions:
            if attraction["id"] == attraction_id:
                return attraction

        return None
