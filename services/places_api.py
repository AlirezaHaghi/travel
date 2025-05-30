import os
import sys
import json
import requests
import time
from typing import List, Dict, Optional
from datetime import datetime
import hashlib
from dotenv import load_dotenv

load_dotenv()


class ImageSearchService:
    """Open source image search service using Wikipedia and free placeholder services"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )

    def search_place_images(self, place_name: str, city: str = "", max_images: int = 3) -> List[Dict]:
        """Search for images using only open source and free services"""
        try:
            images = []

            # Method 1: Try Wikipedia first for real images (always free)
            try:
                wiki_image = self._search_wikipedia_image(place_name)
                if wiki_image:
                    images.append(wiki_image)
            except Exception as e:
                print(f"Wikipedia image search failed for {place_name}: {e}")

            # Method 2: Generate reliable placeholder images for remaining slots
            remaining_images = max_images - len(images)
            if remaining_images > 0:
                fallback_images = self._generate_fallback_images(place_name, city, remaining_images)
                images.extend(fallback_images)

            return images[:max_images]

        except Exception as e:
            print(f"Image search error for {place_name}: {e}")
            # Return fallback images even on complete failure
            return self._generate_fallback_images(place_name, city, max_images)

    def _generate_fallback_images(self, place_name: str, city: str, max_images: int) -> List[Dict]:
        """Generate reliable placeholder images using free services"""
        try:
            fallback_images = []

            # Use multiple reliable free placeholder services
            for i in range(max_images):
                # Try multiple placeholder services for reliability and variety
                placeholder_services = [
                    {
                        'url': f"https://via.placeholder.com/400x300/2196F3/FFFFFF?text={place_name.replace(' ', '+')[:20]}",
                        'name': 'placeholder.com',
                    },
                    {
                        'url': f"https://dummyimage.com/400x300/4CAF50/FFFFFF&text={place_name.replace(' ', '+')[:15]}",
                        'name': 'dummyimage.com',
                    },
                    {
                        'url': f"https://fakeimg.pl/400x300/FF9800/FFFFFF/?text={place_name.replace(' ', '+')[:20]}",
                        'name': 'fakeimg.pl',
                    },
                ]

                # Use different services for variety
                service = placeholder_services[i % len(placeholder_services)]

                fallback_images.append(
                    {
                        'url': service['url'],
                        'source': f"placeholder_{service['name']}",
                        'alt_text': f"{place_name} in {city}",
                        'is_placeholder': True,
                    }
                )

            return fallback_images

        except Exception as e:
            print(f"Fallback image generation failed: {e}")
            # Ultimate fallback with simple, reliable placeholder
            return [
                {
                    'url': f"https://via.placeholder.com/400x300/607D8B/FFFFFF?text=No+Image",
                    'source': 'placeholder_simple',
                    'alt_text': f"{place_name} - No image available",
                    'is_placeholder': True,
                }
            ]

    def _search_wikipedia_image(self, place_name: str) -> Optional[Dict]:
        """Search Wikipedia for images (completely free)"""
        try:
            # Wikipedia API to search for pages
            search_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + place_name.replace(' ', '_')

            response = self.session.get(search_url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if data.get('thumbnail'):
                    return {
                        'url': data['thumbnail']['source'],
                        'source': 'wikipedia',
                        'alt_text': data.get('title', place_name),
                        'description': data.get('extract', '')[:100] + '...' if data.get('extract') else '',
                    }
        except Exception as e:
            print(f"Wikipedia image search failed: {e}")

        return None


class EnhancedPlacesService:
    """Open source places service using only free data sources (OpenStreetMap + Wikipedia)"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'SmartHook-Travel-App/1.0 (https://smartHook.com)'})
        self.image_service = ImageSearchService()

        # Cache for API responses
        self.cache = {}
        self.cache_duration = 3600  # 1 hour

    def search_attractions(
        self, city: str, lat: float, lng: float, categories: Optional[List[str]] = None, limit: int = 20
    ) -> List[Dict]:
        """Search for attractions using only open source data sources"""

        if categories is None:
            categories = ['tourist_attraction', 'museum', 'park', 'landmark', 'gallery']

        all_attractions = []

        # Use only Enhanced OSM search (completely free)
        try:
            print(f"[PLACES] Searching OpenStreetMap for attractions in {city}...")
            osm_results = self._search_enhanced_osm(lat, lng, categories, limit * 2)
            all_attractions.extend(osm_results)
            print(f"[PLACES] Found {len(osm_results)} attractions from OpenStreetMap")
        except Exception as e:
            print(f"Enhanced OSM search failed: {e}")

        # Deduplicate and enrich results
        unique_attractions = self._deduplicate_attractions(all_attractions)
        enriched_attractions = self._enrich_attractions(unique_attractions, city, limit)

        return enriched_attractions

    def _search_enhanced_osm(self, lat: float, lng: float, categories: List[str], limit: int) -> List[Dict]:
        """Enhanced OpenStreetMap search with better categorization (completely free)"""

        # Enhanced OSM query with comprehensive tourism tags
        overpass_query = f"""
        [out:json][timeout:25];
        (
          node["tourism"~"attraction|museum|gallery|viewpoint|artwork|zoo|theme_park|information"]
               (around:10000,{lat},{lng});
          way["tourism"~"attraction|museum|gallery|viewpoint|artwork|zoo|theme_park|information"]
              (around:10000,{lat},{lng});
          node["amenity"~"theatre|cinema|library|arts_centre|community_centre"]
               (around:10000,{lat},{lng});
          way["amenity"~"theatre|cinema|library|arts_centre|community_centre"]
              (around:10000,{lat},{lng});
          node["historic"~"monument|memorial|castle|ruins|archaeological_site|building|manor|palace"]
               (around:10000,{lat},{lng});
          way["historic"~"monument|memorial|castle|ruins|archaeological_site|building|manor|palace"]
              (around:10000,{lat},{lng});
          node["leisure"~"park|nature_reserve|garden|stadium|sports_centre"]
               (around:10000,{lat},{lng});
          way["leisure"~"park|nature_reserve|garden|stadium|sports_centre"]
              (around:10000,{lat},{lng});
          node["culture"~"arts_centre|gallery|museum"]
               (around:10000,{lat},{lng});
          way["culture"~"arts_centre|gallery|museum"]
              (around:10000,{lat},{lng});
        );
        out center meta;
        """

        url = "https://overpass-api.de/api/interpreter"
        response = self.session.post(url, data=overpass_query, timeout=30)
        response.raise_for_status()

        data = response.json()
        attractions = []

        for element in data.get('elements', [])[:limit]:
            try:
                tags = element.get('tags', {})

                if element['type'] == 'node':
                    coord_lat, coord_lng = element['lat'], element['lon']
                elif 'center' in element:
                    coord_lat, coord_lng = element['center']['lat'], element['center']['lon']
                else:
                    continue

                name = tags.get('name', tags.get('name:en', 'Unnamed Place'))
                if name == 'Unnamed Place':
                    continue

                # Better category mapping
                category = self._determine_category(tags)

                # Calculate a rating based on tags (more sophisticated)
                rating = self._calculate_osm_rating(tags)

                # Get description from multiple sources
                description = self._get_osm_description(tags, name)

                attraction = {
                    'id': f"osm_{element['type']}/{element['id']}",
                    'name': name,
                    'rating': rating,
                    'user_ratings_total': self._estimate_popularity(tags),
                    'location': {'lat': coord_lat, 'lng': coord_lng},
                    'address': self._format_osm_address(tags),
                    'category': category,
                    'description': description,
                    'website': tags.get('website', tags.get('url', '')),
                    'price_level': self._determine_price_level(tags),
                    'photos': [],
                    'source': 'osm',
                    'types': [category],
                    'opening_hours': tags.get('opening_hours', ''),
                    'phone': tags.get('phone', ''),
                    'wikipedia': tags.get('wikipedia', ''),
                    'wikidata': tags.get('wikidata', ''),
                }

                attractions.append(attraction)

            except Exception as e:
                print(f"Error processing OSM element: {e}")
                continue

        return attractions

    def _determine_category(self, tags: Dict) -> str:
        """Determine attraction category from OSM tags with better logic"""
        # Priority-based category determination
        if tags.get('tourism') == 'museum' or tags.get('amenity') == 'library':
            return 'museum'
        elif tags.get('tourism') in ['attraction', 'viewpoint', 'artwork']:
            return 'tourist_attraction'
        elif tags.get('leisure') in ['park', 'garden', 'nature_reserve']:
            return 'park'
        elif tags.get('historic') or tags.get('tourism') == 'monument':
            return 'landmark'
        elif tags.get('amenity') in ['theatre', 'cinema', 'arts_centre'] or tags.get('tourism') == 'theatre':
            return 'entertainment'
        elif tags.get('tourism') in ['gallery', 'arts_centre']:
            return 'gallery'
        elif tags.get('leisure') in ['stadium', 'sports_centre']:
            return 'sports'
        else:
            return 'tourist_attraction'

    def _calculate_osm_rating(self, tags: Dict) -> float:
        """Calculate a more sophisticated rating based on OSM tags"""
        rating = 3.5  # Base rating

        # Boost for comprehensive information
        if tags.get('wikidata'):
            rating += 0.7
        if tags.get('wikipedia'):
            rating += 0.5
        if tags.get('website'):
            rating += 0.3
        if tags.get('phone'):
            rating += 0.2
        if tags.get('opening_hours'):
            rating += 0.2
        if tags.get('description'):
            rating += 0.3

        # Tourism importance boost
        if tags.get('tourism') == 'attraction':
            rating += 0.4
        elif tags.get('tourism') in ['museum', 'gallery']:
            rating += 0.3
        elif tags.get('historic'):
            rating += 0.4

        # Heritage and cultural significance
        if tags.get('heritage') or tags.get('unesco'):
            rating += 0.6
        if tags.get('artwork_type') or tags.get('monument'):
            rating += 0.3

        # Accessibility features
        if tags.get('wheelchair') == 'yes':
            rating += 0.1

        return min(rating, 5.0)

    def _get_osm_description(self, tags: Dict, name: str) -> str:
        """Generate description from OSM tags"""
        description_parts = []

        # Use existing description first
        if tags.get('description'):
            description_parts.append(tags['description'])

        # Add tourism type information
        if tags.get('tourism'):
            tourism_type = tags['tourism'].replace('_', ' ').title()
            description_parts.append(f"A {tourism_type}")

        # Add historic information
        if tags.get('historic'):
            historic_type = tags['historic'].replace('_', ' ').title()
            description_parts.append(f"Historic {historic_type}")

        # Add amenity information
        if tags.get('amenity'):
            amenity_type = tags['amenity'].replace('_', ' ').title()
            description_parts.append(f"{amenity_type}")

        # If no description found, create a basic one
        if not description_parts:
            category = self._determine_category(tags)
            description_parts.append(f"Popular {category.replace('_', ' ')} in the area")

        return '. '.join(description_parts)[:200] + ('...' if len('. '.join(description_parts)) > 200 else '')

    def _estimate_popularity(self, tags: Dict) -> int:
        """Estimate popularity based on available information"""
        popularity = 25  # Base popularity

        if tags.get('wikidata'):
            popularity += 50
        if tags.get('wikipedia'):
            popularity += 30
        if tags.get('website'):
            popularity += 20
        if tags.get('tourism') == 'attraction':
            popularity += 25
        if tags.get('historic'):
            popularity += 20

        return min(popularity, 500)

    def _format_osm_address(self, tags: Dict) -> str:
        """Format address from OSM tags"""
        parts = []

        if tags.get('addr:housenumber'):
            parts.append(tags['addr:housenumber'])
        if tags.get('addr:street'):
            parts.append(tags['addr:street'])
        if tags.get('addr:city'):
            parts.append(tags['addr:city'])
        if tags.get('addr:country'):
            parts.append(tags['addr:country'])

        return ', '.join(parts) if parts else ''

    def _determine_price_level(self, tags: Dict) -> int:
        """Determine price level from OSM tags"""
        fee = tags.get('fee', '').lower()
        if fee == 'no':
            return 0
        elif fee == 'yes':
            return 2
        elif tags.get('tourism') in ['museum', 'gallery']:
            return 2  # Museums typically have entry fees
        elif tags.get('leisure') in ['park', 'garden']:
            return 0  # Parks are typically free
        else:
            return 1  # Default moderate pricing

    def _deduplicate_attractions(self, attractions: List[Dict]) -> List[Dict]:
        """Remove duplicate attractions based on name and proximity"""
        unique_attractions = []
        seen_names = set()

        for attraction in attractions:
            name_key = attraction['name'].lower().strip()

            # Check if we've seen this name before
            if name_key in seen_names:
                continue

            # Check for proximity duplicates
            is_duplicate = False
            for existing in unique_attractions:
                if (
                    self._calculate_distance(
                        attraction['location']['lat'],
                        attraction['location']['lng'],
                        existing['location']['lat'],
                        existing['location']['lng'],
                    )
                    < 0.1
                ):  # Within 100 meters
                    if self._name_similarity(attraction['name'], existing['name']) > 0.8:
                        is_duplicate = True
                        break

            if not is_duplicate:
                unique_attractions.append(attraction)
                seen_names.add(name_key)

        return unique_attractions

    def _calculate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate distance between two points in kilometers"""
        from math import radians, sin, cos, sqrt, atan2

        R = 6371  # Earth's radius in kilometers

        lat1, lng1, lat2, lng2 = map(radians, [lat1, lng1, lat2, lng2])
        dlat = lat2 - lat1
        dlng = lng2 - lng1

        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlng / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        return R * c

    def _name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two names"""
        name1 = name1.lower().strip()
        name2 = name2.lower().strip()

        if name1 == name2:
            return 1.0

        # Simple similarity based on common words
        words1 = set(name1.split())
        words2 = set(name2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def _enrich_attractions(self, attractions: List[Dict], city: str, limit: int) -> List[Dict]:
        """Enrich attractions with images and additional data"""
        enriched = []

        for attraction in attractions[:limit]:
            try:
                # Add images if not already present
                if not attraction.get('photos'):
                    images = self.image_service.search_place_images(attraction['name'], city, max_images=2)
                    attraction['photos'] = images

                # Ensure required fields
                attraction.setdefault('estimated_duration', self._estimate_duration(attraction))
                attraction.setdefault('price_level', 1)
                attraction.setdefault('description', f"Popular attraction in {city}")

                enriched.append(attraction)

            except Exception as e:
                print(f"Error enriching attraction {attraction.get('name', 'Unknown')}: {e}")
                continue

        # Sort by rating and completeness of information
        enriched.sort(
            key=lambda x: (
                x.get('rating', 0),
                len(x.get('description', '')),
                1 if x.get('wikipedia') else 0,
                1 if x.get('wikidata') else 0,
            ),
            reverse=True,
        )

        return enriched

    def _estimate_duration(self, attraction: Dict) -> float:
        """Estimate visit duration based on category and information available"""
        category = attraction.get('category', 'tourist_attraction')

        duration_map = {
            'museum': 2.5,
            'park': 2.0,
            'tourist_attraction': 1.5,
            'landmark': 1.0,
            'gallery': 1.5,
            'entertainment': 3.0,
            'sports': 2.0,
        }

        base_duration = duration_map.get(category, 2.0)

        # Adjust based on rating and information richness
        rating = attraction.get('rating', 3.5)
        if rating > 4.5:
            base_duration *= 1.3
        elif rating < 3.0:
            base_duration *= 0.8

        # Adjust based on available information (more info = longer visit)
        if attraction.get('wikipedia'):
            base_duration *= 1.2
        if attraction.get('website'):
            base_duration *= 1.1

        return base_duration


# Convenience function for backward compatibility
def create_enhanced_places_service():
    """Create an instance of the enhanced places service"""
    return EnhancedPlacesService()
