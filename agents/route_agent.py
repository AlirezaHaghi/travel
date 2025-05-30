import itertools
import math
from datetime import datetime, timedelta
import networkx as nx
from utils import ask_openai, extract_number
import re
import sys
import os

# Add the parent directory to sys.path to allow imports from services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.information_agent import InformationAgent


class RouteAgent:
    def __init__(self, api_key=None):
        """Initialize RouteAgent with optional API key for distance calculations"""
        self.api_key = api_key
        self.distances_cache = {}
        self.info_agent = None
        try:
            self.info_agent = InformationAgent()
        except Exception as e:
            print(f"Error initializing InformationAgent in RouteAgent: {e}")

    def optimize_daily_route(self, attractions_for_day):
        """
        Optimize the order of attractions for a single day using the InformationAgent's plan_with_waypoints.

        Args:
            attractions_for_day: List of attraction objects with location data

        Returns:
            List of the same attractions in optimal travel order
        """
        if not attractions_for_day or len(attractions_for_day) <= 1:
            return attractions_for_day  # No optimization needed for 0 or 1 attraction

        # Check if we can use the InformationAgent
        if not self.info_agent:
            print("InformationAgent not available. Using fallback TSP solution.")
            return self.get_optimal_route(attractions_for_day)

        try:
            # Extract first attraction as starting point
            origin = attractions_for_day[0]

            # Extract last attraction as destination (complete the loop back to start for simplicity)

            # The rest are waypoints
            waypoints = []
            for attraction in attractions_for_day[1:]:
                if "location" in attraction and "lat" in attraction["location"] and "lng" in attraction["location"]:
                    waypoint_location = f"{attraction['location']['lat']},{attraction['location']['lng']}"
                    waypoints.append(waypoint_location)
                else:
                    print(f"Warning: Attraction {attraction.get('name', 'unknown')} missing location data")

            # Prepare origin and destination strings
            if "location" in origin and "lat" in origin["location"] and "lng" in origin["location"]:
                origin_location = f"{origin['location']['lat']},{origin['location']['lng']}"
                destination_location = origin_location  # Loop back to start
            else:
                print(f"Warning: Origin attraction {origin.get('name', 'unknown')} missing location data")
                return attractions_for_day  # Can't optimize without location data

            # Call plan_with_waypoints
            optimized_route_data = self.info_agent.plan_with_waypoints(
                origin=origin_location, destination=destination_location, waypoints=waypoints, mode='driving'
            )

            if not optimized_route_data:
                print("Failed to get optimized route. Using fallback TSP solution.")
                return self.get_optimal_route(attractions_for_day)

            # Extract the optimized waypoint order
            waypoint_indices = optimized_route_data.get('waypoint_original_indices', [])

            # Create the optimized list of attractions
            optimized_attractions = [origin]  # Start with origin

            # Add waypoints in optimized order
            for idx in waypoint_indices:
                optimized_attractions.append(attractions_for_day[idx + 1])  # +1 because we excluded origin

            return optimized_attractions

        except Exception as e:
            print(f"Error optimizing daily route: {e}")
            # Fallback to internal TSP solver
            return self.get_optimal_route(attractions_for_day)

    def get_optimal_route(self, spots, start_point=None):
        """Calculate optimal route between selected attractions"""
        print(f"[DEBUG] Getting optimal route for spots: {spots}")

        if not spots or len(spots) <= 1:
            print(f"[DEBUG] Not enough spots to calculate route. Returning spots as is: {spots}")
            return spots

        # Get distance matrix
        distance_matrix = self._get_distance_matrix(spots)
        print(f"[DEBUG] Distance matrix calculated: {distance_matrix}")

        # Solve TSP (Traveling Salesman Problem)
        if len(spots) <= 5:
            # For small number of points, brute force is fine
            route = self._solve_tsp_brute_force(spots, distance_matrix)
            print(f"[DEBUG] Brute force TSP solution: {route}")
            return route
        else:
            # For larger problems, use approximate method
            route = self._solve_tsp_approximate(spots, distance_matrix)
            print(f"[DEBUG] Approximate TSP solution: {route}")
            return route

    def _get_distance_matrix(self, spots):
        """Get distance matrix between all pairs of spots using Haversine formula."""
        n = len(spots)
        matrix = [[0.0 for _ in range(n)] for _ in range(n)]

        # Fill the matrix with distances
        for i in range(n):
            for j in range(i + 1, n):
                # Get distance between spot i and spot j
                distance = self._calculate_distance(spots[i], spots[j])
                matrix[i][j] = distance
                matrix[j][i] = distance  # Symmetric

        return matrix

    def _calculate_distance(self, spot1, spot2):
        """Calculate distance between two spots using coordinates"""
        # Check if we have location data
        if "location" not in spot1 or "location" not in spot2:
            return 1  # Default distance if no location data

        # Check cache first
        cache_key = f"{spot1['id']}_{spot2['id']}"
        if cache_key in self.distances_cache:
            return self.distances_cache[cache_key]

        lat1, lon1 = spot1["location"]["lat"], spot1["location"]["lng"]
        lat2, lon2 = spot2["location"]["lat"], spot2["location"]["lng"]

        R = 6371  # Earth radius in km

        # Convert coordinates to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        distance = R * c

        # Cache the result
        self.distances_cache[cache_key] = distance

        return distance

    def _solve_tsp_brute_force(self, spots, distance_matrix):
        """Solve TSP by trying all permutations (only for small problems)"""
        n = len(spots)
        best_distance = float('inf')
        best_order = list(range(n))

        # Try all permutations
        for perm in itertools.permutations(range(n)):
            distance = sum(distance_matrix[perm[i]][perm[i + 1]] for i in range(n - 1))

            if distance < best_distance:
                best_distance = distance
                best_order = perm

        # Return spots in optimal order
        return [spots[i] for i in best_order]

    def _solve_tsp_approximate(self, spots, distance_matrix):
        """Solve TSP using approximation algorithm (Christofides or nearest neighbor)"""
        # Create a complete graph
        G = nx.Graph()
        n = len(spots)

        # Add nodes and edges with distances
        for i in range(n):
            G.add_node(i)
            for j in range(i + 1, n):
                G.add_edge(i, j, weight=distance_matrix[i][j])

        # Find approximate TSP tour
        tour = nx.approximation.traveling_salesman_problem(G, cycle=True)

        # Remove the last node (it's the same as the first to complete the cycle)
        tour = tour[:-1]

        # Return spots in calculated order
        return [spots[i] for i in tour]

    def format_daily_plan_to_itinerary(self, daily_plan_name_dict, all_spots_object_map, start_date_str):
        """Generate daily itinerary based on a pre-defined daily plan of attraction names."""
        itinerary = []
        try:
            current_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        except ValueError:
            print(f"[ERROR] Invalid start_date_str format: {start_date_str}. Expected YYYY-MM-DD.")
            # Fallback to today if date is invalid, or handle error as preferred
            current_date = datetime.now()
            print(f"[WARN] Using current date {current_date.strftime('%Y-%m-%d')} as fallback.")

        # Sort day keys numerically (e.g., "day1", "day2", ...)
        sorted_day_keys = sorted(
            [key for key in daily_plan_name_dict.keys() if key.startswith("day") and re.match(r"day\d+", key)],
            key=lambda x: int(re.findall(r'\d+', x)[0]),
        )

        for day_key in sorted_day_keys:
            day_number = int(re.findall(r'\d+', day_key)[0])
            spot_names_for_day = daily_plan_name_dict.get(day_key, [])

            current_day_spot_objects_raw = []
            for name in spot_names_for_day:
                if name in all_spots_object_map:
                    current_day_spot_objects_raw.append(all_spots_object_map[name])
                else:
                    print(
                        f"[WARN] Attraction name '{name}' from daily plan (day {day_number}) not found in all_spots_object_map."
                    )

            # Optimize the route for this day's attractions
            if current_day_spot_objects_raw and len(current_day_spot_objects_raw) > 1:
                print(f"Optimizing route for day {day_number} with {len(current_day_spot_objects_raw)} attractions...")
                optimized_day_attractions = self.optimize_daily_route(current_day_spot_objects_raw)
                print(f"Route optimization complete for day {day_number}")
                current_day_spot_objects_raw = optimized_day_attractions

            current_day_spots_timed = []
            # Use 8 hours as a guideline for sequential timing within the day
            # The LLM was prompted to consider an 8-hour day, so the sum of durations should ideally be around that.
            start_offset_hours = 0  # Hours from 9 AM, e.g., 0 means 9 AM

            for spot_obj in current_day_spot_objects_raw:
                spot_duration = spot_obj.get("estimated_duration", 2)  # Default to 2 hours if not specified

                spot_with_time = spot_obj.copy()

                # Calculate start and end times for the activity (e.g., from 9:00)
                activity_start_hour = 9 + start_offset_hours
                activity_end_hour = activity_start_hour + spot_duration

                spot_with_time["start_time"] = f"{int(activity_start_hour):02d}:00"
                spot_with_time["end_time"] = f"{int(activity_end_hour):02d}:00"
                current_day_spots_timed.append(spot_with_time)

                start_offset_hours += spot_duration  # Next spot starts after this one

            itinerary.append(
                {"day": day_number, "date": current_date.strftime("%Y-%m-%d"), "spots": current_day_spots_timed}
            )
            current_date += timedelta(days=1)

        return itinerary

    def estimate_budget(self, spots, user_prefs, should_rent_car=False, car_info=None, fuel_price=None):
        """Estimate budget for the selected attractions"""
        # Base daily costs
        base_costs = {
            "low": {"accommodation": 50, "food": 30, "transport": 10},
            "medium": {"accommodation": 100, "food": 60, "transport": 20},
            "high": {"accommodation": 200, "food": 100, "transport": 40},
        }

        # Get budget level from user preferences
        budget_value = user_prefs.get("budget", "medium")

        # Convert budget value to level
        if isinstance(budget_value, str):
            # Remove currency symbols and convert to numeric range
            budget_value = budget_value.replace("$", "").replace(",", "")
            if "–" in budget_value or "-" in budget_value:
                # Take the lower bound of the range
                budget_value = budget_value.split("–")[0].split("-")[0]
            try:
                budget_value = float(budget_value)
                if budget_value < 1000:
                    budget_level = "low"
                elif budget_value < 3000:
                    budget_level = "medium"
                else:
                    budget_level = "high"
            except ValueError:
                budget_level = "medium"
        else:
            budget_level = "medium"

        num_people = user_prefs.get("people", 1)
        num_days = user_prefs.get("days", 1)

        # Calculate base costs
        daily_cost = sum(base_costs[budget_level].values())
        base_total = daily_cost * int(num_days) * int(num_people)

        # Add attraction costs
        attraction_cost = 0
        for spot in spots:
            price_level = spot.get("price_level", 2)
            # Convert price level to actual cost
            cost_map = {0: 0, 1: 10, 2: 20, 3: 30, 4: 50}
            attraction_cost += cost_map.get(price_level, 20) * int(num_people)

        # Calculate total
        total = base_total + attraction_cost

        # Initialize car rental and fuel costs to zero
        car_rental_cost = 0
        fuel_cost = 0

        # Only calculate car rental and fuel costs if car rental is recommended
        if should_rent_car and car_info:
            ai_response = ask_openai(
                prompt=f"""
                here is the car info: {car_info}, and the budget: {budget_level},
                please select the most suitable car for the user
                important: just return the idx of the car in the car_info list,only return the idx(a number),no other words
                """
            )
            print(f"[DEBUG] AI response: {ai_response}", type(ai_response["answer"]))
            recommend_car = int(extract_number(ai_response["answer"]))
            idx = recommend_car - 1 if recommend_car in range(1, len(car_info) + 1) else 0
            print(f"[DEBUG] Selected car idx: {idx}")
            car_rental_cost = car_info[idx]["price"]
            total += car_rental_cost
            print(f"[DEBUG] Car rental cost: {car_rental_cost}")

            # Calculate transport cost if car_rental is true
            route = self.get_optimal_route(spots)
            print(f"[DEBUG] Optimal route: {route}")

            total_distance = 0
            for i in range(len(route) - 1):
                total_distance += self._calculate_distance(route[i], route[i + 1])

            fuel_costs = {
                "low": {"fuel_efficiency": 7.0, "car_type": "Economy"},  # liters/100km
                "medium": {"fuel_efficiency": 8.5, "car_type": "Mid-range"},  # Slightly higher for mid-range cars
                "high": {"fuel_efficiency": 10.0, "car_type": "Luxury"},  # Higher for luxury cars
            }

            # Calculate fuel cost
            fuel_info = fuel_costs[budget_level]
            fuel_consumption = (total_distance * fuel_info["fuel_efficiency"]) / 100  # Total fuel consumption (liters)
            # Add check for None and use a default value if fuel_price is None
            if fuel_price is None:
                print("[WARN] No fuel price available, using default value of $3.50 per gallon")
                fuel_price = 3.50  # Default price in USD per gallon
            fuel_cost = fuel_consumption * fuel_price  # Total fuel cost
            total += fuel_cost
            print(f"[DEBUG] Fuel cost: {fuel_cost}")

        # Return detailed budget
        return {
            "total": round(total, 2),
            "accommodation": base_costs[budget_level]["accommodation"] * int(num_days) * int(num_people),
            "food": base_costs[budget_level]["food"] * int(num_days) * int(num_people),
            "transport": base_costs[budget_level]["transport"] * int(num_days) * int(num_people),
            "attractions": attraction_cost,
            "car_rental": round(car_rental_cost, 2),
            "fuel_cost": round(fuel_cost, 2),
        }
