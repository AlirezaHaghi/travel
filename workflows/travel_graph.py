from agents.chat_agent import ChatAgent
from agents.information_agent import InformationAgent
from agents.recommend_agent import RecommendAgent
from agents.strategy_agent import StrategyAgent
from agents.route_agent import RouteAgent
from agents.communication_agent import CommunicationAgent
from datetime import datetime, timedelta
from langchain.schema import AIMessage

import traceback


class TravelGraphSingleton:
    """Singleton class to manage agent instances that are expensive to create"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TravelGraphSingleton, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not TravelGraphSingleton._initialized:
            print("[DEBUG] Initializing TravelGraph agents (singleton)")
            # Initialize agents only once
            self.info_agent = InformationAgent()
            self.chat_agent = ChatAgent()
            self.strategy_agent = StrategyAgent()
            self.recommend_agent = RecommendAgent()
            self.route_agent = RouteAgent()
            self.comm_agent = CommunicationAgent()
            TravelGraphSingleton._initialized = True
            print("[DEBUG] TravelGraph agents initialized successfully")
        else:
            print("[DEBUG] Using existing TravelGraph agents (singleton)")


# This is a simplified state graph manager since we're not using the actual langgraph library
class TravelGraph:
    _singleton = None

    def __init__(self, session_id=None):
        # Get or create the singleton instance for agents
        if TravelGraph._singleton is None:
            TravelGraph._singleton = TravelGraphSingleton()

        # Use agents from singleton
        self.info_agent = TravelGraph._singleton.info_agent
        self.chat_agent = TravelGraph._singleton.chat_agent
        self.strategy_agent = TravelGraph._singleton.strategy_agent
        self.recommend_agent = TravelGraph._singleton.recommend_agent
        self.route_agent = TravelGraph._singleton.route_agent
        self.comm_agent = TravelGraph._singleton.comm_agent

        # Initialize state for this session/instance
        self.session_id = session_id
        self.state = {
            "user_info": {},
            "attractions": [],
            "selected_attractions": [],
            "additional_attractions": [],
            "should_rent_car": False,
            "itinerary": [],
            "budget": {},
            "ai_recommendation_generated": False,
        }

        print(f"[DEBUG] Created TravelGraph instance for session: {session_id}")

    @classmethod
    def get_shared_agents(cls):
        """Get the shared agent instances"""
        if cls._singleton is None:
            cls._singleton = TravelGraphSingleton()
        return cls._singleton

    def process_step(self, step_name, **kwargs):
        print(f"[DEBUG] Processing step {step_name}")

        if 'ai_recommendation_generated' in kwargs:
            self.state['ai_recommendation_generated'] = str(kwargs['ai_recommendation_generated']).lower() == 'true'

        result = {}
        if step_name == "chat":
            result = self._process_chat(**kwargs)
        elif step_name == "information":
            result = self._process_information(**kwargs)
        elif step_name == "recommend":
            result = self._process_recommend(**kwargs)
        elif step_name == "strategy":
            result = self._process_strategy(**kwargs)
        elif step_name == "route":
            result = self._process_route(**kwargs)
        elif step_name == "communication":
            result = self._process_communication(**kwargs)
        else:
            result = {"error": f"Unknown step: {step_name}"}

        return result

    def _process_chat(self, user_input=None, **kwargs):
        current_user_info = self.state.get("user_info", {}).copy()  # Important to work with a copy for updates
        chat_result = self.chat_agent.collect_info(user_input or "", current_user_info)

        if chat_result.get("state"):
            self.state["user_info"].update(chat_result["state"])
            # print(f"[DEBUG] Updated user_info in _process_chat: {self.state['user_info']}")
            # Remove rental_post from state if it was ever set here, though unlikely for chat
            # self.state.pop("rental_post", None)

        response_data = {
            "state": self.state.copy(),  # Return a fresh copy of the current state
            "stream": chat_result.get("stream"),
            "missing_fields": chat_result.get("missing_fields", []),
        }

        if not chat_result.get("complete", False):
            response_data["next_step"] = "chat"
            return response_data

        # If chat is complete, automatically proceed to information gathering
        # The state (self.state) is already updated, _process_information will use it.
        info_step_result = self._process_information()
        return info_step_result  # This result will contain next_step, stream, data, and state

    def _process_information(self, **kwargs):
        user_prefs = self.state["user_info"]
        city = user_prefs.get("city")

        if not city:

            def error_gen():
                yield AIMessage(content="Please tell me which city you'd like to visit.")

            return {"next_step": "chat", "stream": error_gen(), "missing_fields": ["city"], "state": self.state.copy()}

        city_coordinates = self.info_agent.city2geocode(city)
        if not city_coordinates:

            def error_gen():
                yield AIMessage(content=f"Sorry, I couldn't find coordinates for {city}.")

            return {"next_step": "chat", "stream": error_gen(), "state": self.state.copy()}

        # Get weather summary first
        weather_summary_str = None
        user_start_date = user_prefs.get("start_date", "not decided")
        user_days_str = user_prefs.get("days")

        # Handle "not decided" case by setting start date to 7 days from now
        if user_start_date == "not decided":
            user_start_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
            user_prefs["start_date"] = user_start_date  # Update the state with the new start date

        if user_days_str:
            try:
                num_days = int(user_days_str)
                weather_data_result = self.info_agent.get_weather(
                    city_coordinates["lat"], city_coordinates["lng"], user_start_date, num_days, summary=True
                )
                if weather_data_result and 'summary' in weather_data_result:
                    summary_val = weather_data_result['summary']
                    if hasattr(summary_val, 'content'):  # If AIMessage
                        weather_summary_str = summary_val.content
                    elif isinstance(summary_val, str):
                        weather_summary_str = summary_val
                    self.state["weather_summary"] = weather_summary_str
                    print(f"[DEBUG] Weather summary set in state: '{weather_summary_str}'")
                else:
                    print(f"[DEBUG] Weather summary not found or in unexpected format: {weather_data_result}")
            except ValueError:
                print(f"[ERROR] Invalid 'days' for weather: {user_days_str}")
            except Exception as e:
                print(f"[ERROR] Exception fetching weather summary: {e}")
                traceback.print_exc()
        else:
            print("[DEBUG] Weather info not fetched (no days).")

        # Get attractions - InfoAgent now handles LLM re-ranking internally
        print(f"[DEBUG] Calling info_agent.get_attractions for '{city}' with user_prefs and weather.")
        weather_summary = self.state.get("weather_summary") or ""  # Ensure it's always a string
        attractions_from_info_agent = self.info_agent.get_attractions(
            lat=city_coordinates["lat"],
            lng=city_coordinates["lng"],
            user_prefs=user_prefs,  # Pass full user_prefs
            weather_summary=weather_summary,  # Pass fetched weather summary as string
            number=20,  # Desired number of top attractions after LLM ranking
            poi_type="tourist_attraction",
            # sort_by="rating" # Initial sort inside get_attractions before LLM
        )

        self.state["attractions"] = attractions_from_info_agent if attractions_from_info_agent else []
        print(f"[DEBUG] attractions state updated with {len(self.state['attractions'])} LLM-ranked items.")

        def info_gen_message():
            if self.state["attractions"]:
                yield AIMessage(
                    content=f"I've prepared a personalized list of {len(self.state['attractions'])} attractions in {city} for you, considering your preferences and the weather. Please take a look and select your favorites."
                )
            else:
                yield AIMessage(
                    content=f"I couldn't find attractions in {city} matching your preferences right now. You might want to try different criteria or another city."
                )

        return {
            "next_step": "recommend",
            "stream": info_gen_message(),
            "attractions": self.state["attractions"],  # This list is now LLM-ranked
            "map_data": self.recommend_agent.generate_map_data(
                self.state["attractions"]
            ),  # recommend_agent helps with map data
            "state": self.state.copy(),
        }

    def _process_recommend(self, selected_attraction_ids=None, **kwargs):
        """Process recommend agent step"""
        try:
            user_prefs = self.state["user_info"]
            attractions = self.state["attractions"]

            # Check if we have selected_attraction_ids in kwargs
            if 'selected_attraction_ids' in kwargs and kwargs['selected_attraction_ids']:
                selected_attraction_ids = kwargs['selected_attraction_ids']

            if selected_attraction_ids:
                # User has selected specific attractions
                selected_attractions = [
                    a for a in attractions if a and a.get("id") and a["id"] in selected_attraction_ids
                ]
                self.state["selected_attractions"] = selected_attractions

                # Create a generator that yields a transition message
                def transition_generator():
                    yield AIMessage(content="Processing your selected attractions...")

                return {
                    "next_step": "strategy",
                    "stream": transition_generator(),
                    "selected_attractions": selected_attractions,
                }
            else:
                # Recommend attractions to the user
                if not attractions:
                    return {
                        "next_step": "recommend",
                        "stream": None,
                        "response": "No attractions available for recommendation.",
                        "recommended_attractions": [],
                        "map_data": [],
                    }

                recommended = self.recommend_agent.recommend_core_attractions(
                    user_prefs,
                    attractions,
                )

                # Create a generator that yields the recommendation message
                def recommendation_generator():
                    yield AIMessage(content="Here are some recommended attractions for you.")

                return {
                    "next_step": "recommend",  # Stay on this step until user selects attractions
                    "stream": recommendation_generator(),
                    "recommended_attractions": recommended,
                    "map_data": self.recommend_agent.generate_map_data(recommended),
                }
        except Exception as e:
            print(f"Error in _process_recommend: {str(e)}")
            return {
                "next_step": "error",
                "stream": None,
                "response": "An error occurred while processing recommendations.",
                "error": str(e),
            }

    def _process_strategy(self, **kwargs):
        """Process strategy agent step"""
        # Check if this is a confirm selection request or a satisfaction confirmation
        user_input_lower = kwargs.get('user_input', '').lower()
        is_confirm_selection = user_input_lower == 'here are my selected attractions'
        is_satisfaction_confirmation = 'satisfied with your recommendation' in user_input_lower

        # Log what type of confirmation message we received
        if is_confirm_selection:
            print("[DEBUG] Received initial confirmation of selections")
        elif is_satisfaction_confirmation:
            print("[DEBUG] Received satisfaction confirmation message")
        else:
            print(f"[DEBUG] Received other input: '{user_input_lower}'")

        # Print the current state of should_rent_car for debugging
        if 'should_rent_car' in self.state:
            print(f"[DEBUG] Current should_rent_car value: {self.state['should_rent_car']}")
        else:
            print("[DEBUG] should_rent_car not yet set in state")

        # If recommendations haven't been generated yet and this is the initial confirm selection
        if not self.state['ai_recommendation_generated'] and is_confirm_selection:
            # Update state flags BEFORE generating recommendations
            self.state['ai_recommendation_generated'] = True
            self.state['user_input_processed'] = True

            selected_attractions = self.state["selected_attractions"]
            total_days = self.state["user_info"].get("days", 1)

            # Plan remaining time and suggest additional attractions
            strategy_result = self.strategy_agent.plan_remaining_time(
                selected_spots=selected_attractions,
                total_days=total_days,
                all_attractions=self.state["attractions"],  ## This should be the full list of attractions
                user_prefs=self.state["user_info"],  # Pass user_prefs
                weather_summary=self.state.get("weather_summary"),  # Pass weather_summary
            )

            self.state["attractions"] = strategy_result[
                "additional_attractions"
            ]  ## 现在这里的attractions 是经过筛选的,也是最终的attractions
            self.state["daily_plan"] = strategy_result.get("daily_plan", {})  # Store the daily plan

            # Initialize should_rent_car to False by default
            self.state["should_rent_car"] = False
            print("[DEBUG] Initialized should_rent_car to False")

            # Get AI recommendation about the overall plan
            # This will also analyze the recommendation and update should_rent_car in user_prefs
            ai_recommendation = self.strategy_agent.get_ai_recommendation(
                user_prefs=self.state["user_info"],
                selected_spots=selected_attractions,
                total_days=total_days,
            )

            # Get the should_rent_car value from user_prefs after AI recommendation analysis
            # This value is set by extract_rental_recommendation in strategy_agent.py
            ai_should_rent_car = self.state["user_info"].get("should_rent_car", False)
            self.state["should_rent_car"] = ai_should_rent_car

            print(f"[CRITICAL] AI rental recommendation set should_rent_car to: {ai_should_rent_car}")
            print(f"[DEBUG] Updated state should_rent_car value: {self.state['should_rent_car']}")

            # Create a copy of the state to return
            state_copy = self.state.copy()

            return {
                "next_step": "strategy",
                "stream": ai_recommendation,
                "remaining_hours": strategy_result["remaining_hours"],
                "additional_attractions": strategy_result["additional_attractions"],
                "should_rent_car": self.state["should_rent_car"],
                "state": state_copy,
                "ai_recommendation_generated": True,
                "user_input_processed": True,
            }
        # Handle both cases: either we have already generated recommendations
        # or this is a satisfaction confirmation message
        elif self.state['ai_recommendation_generated'] or is_satisfaction_confirmation:
            print("[DEBUG] Recommendations already generated or satisfaction confirmed, moving to next step")

            # Check if this is a satisfaction confirmation message and we need to process it specially
            if is_satisfaction_confirmation and not self.state['ai_recommendation_generated']:
                print("[CRITICAL] Handling satisfaction confirmation without prior recommendation generation")
                # This means user sent satisfaction message before going through normal flow
                # We need to ensure should_rent_car is correctly set to false in this case
                self.state["should_rent_car"] = False  # Ensure it's false
                print("[DEBUG] Set should_rent_car to False for satisfaction message without prior recommendation")

            # ALWAYS GO TO ROUTE STEP, SKIP COMMUNICATION
            next_step = "route"
            print(f"[CRITICAL] Decision point: Forcing next_step to '{next_step}' to skip car rental communication.")

            # Create a generator that yields the transition message
            def transition_generator():
                if next_step == "route":
                    yield AIMessage(content="Moving to route planning...")
                else:
                    yield AIMessage(content="Moving to car rental options...")

            # Create a copy of the state to return
            state_copy = self.state.copy()

            return {
                "next_step": next_step,
                "stream": transition_generator(),
                "state": state_copy,
                "ai_recommendation_generated": True,
                "user_input_processed": True,
            }
        else:
            print("[DEBUG] Not a confirm selection request and recommendations not generated yet")

            # Create a generator that yields a message asking for confirmation
            def confirmation_generator():
                yield AIMessage(content="Please click the 'Confirm Selection' button to proceed with your travel plan.")

            # Create a copy of the state to return
            state_copy = self.state.copy()
            print(f"[DEBUG] State to be returned: {state_copy}")

            return {
                "next_step": "strategy",
                "stream": confirmation_generator(),
                "state": state_copy,
                "ai_recommendation_generated": False,
                "user_input_processed": False,
            }

    # After strategy step + self.state.get("should_rent_car", False) == True
    def _process_communication(self, response_message=None, **kwargs):
        """Process communication agent step - LOGIC MOSTLY COMMENTED OUT"""
        # First check if car rental is actually recommended (THIS CHECK IS NOW REDUNDANT as we skip this step)
        # if not self.state.get("should_rent_car", False):
        #     # If car rental is NOT recommended, skip to route planning
        #     def skip_rental_generator():
        #         yield AIMessage(content="Moving to the route planning step...")
        #
        #     return {
        #         "next_step": "route",
        #         "stream": skip_rental_generator(),
        #     }

        # if response_message and self.state.get("rental_post"):
        #     # Handle response to rental post
        #     reply = self.comm_agent.handle_rental_response(
        #         self.state["rental_post"],
        #         response_message
        #     )
        #
        #     # Create a generator that yields the response message
        #     def reply_generator():
        #         yield AIMessage(content="Thank you for handling the car rental.")
        #
        #     return {
        #         "next_step": "route",  # Move to route planning after rental communication
        #         "stream": reply_generator(),
        #         "reply": reply
        #     }
        # else:
        #     # Generate rental post
        #     location = self.state["user_info"].get("city", "")
        #     duration = self.state["user_info"].get("days", 1)
        #
        #     rental_post_content = self.comm_agent.post_car_rental_request(
        #         location,
        #         duration,
        #         self.state["user_info"]
        #     )
        #
        #     self.state["rental_post"] = rental_post_content # Store actual content not the UI representation
        #
        #     # Create a generator that yields the response message
        #     def response_generator():
        #         yield AIMessage(content="We recommend renting a car for your trip. I've created a car rental request post for you. Feel free to use it.")
        #
        #     return {
        #         "next_step": "route",  # Continue with route planning while waiting for responses
        #         "stream": response_generator(),
        #         "rental_post": rental_post_content # Return the actual post content
        #     }

        # Fallback / Default behavior if this step is somehow still called:
        print("[WARN] _process_communication was called but should be skipped. Proceeding to route planning.")

        def default_transition_generator():
            yield AIMessage(content="Proceeding to route planning...")

        return {"next_step": "route", "stream": default_transition_generator(), "state": self.state.copy()}

    def _process_route(self, start_date=None, **kwargs):
        """Process route agent step"""
        try:
            # Get start date from user preferences, fallback to provided start_date, then to current date
            start_date = self.state["user_info"].get("start_date") or start_date or datetime.now().strftime("%Y-%m-%d")

            all_attractions_objects = self.state[
                "attractions"
            ]  ## This is the flat list of all planned attraction objects
            daily_plan_name_dict = self.state.get("daily_plan")  # This is {"day1": ["NameA"], ...}

            # print(f"[DEBUG] All attractions: {all_attractions}")

            if not all_attractions_objects:
                return {
                    "next_step": "complete",
                    "response": "No attractions selected for the trip.",
                    "itinerary": [],
                    "budget": {},
                    "optimal_route": [],
                }

            # Generate itinerary first
            days = int(self.state["user_info"].get("days", 1))  # Ensure days is an integer

            itinerary = []
            if daily_plan_name_dict and isinstance(daily_plan_name_dict, dict) and all_attractions_objects:
                all_spots_map = {spot["name"]: spot for spot in all_attractions_objects if spot and "name" in spot}
                itinerary = self.route_agent.format_daily_plan_to_itinerary(
                    daily_plan_name_dict, all_spots_map, start_date
                )
            else:
                print(
                    "[ERROR] Could not generate itinerary: daily_plan_name_dict or all_attractions_objects missing/invalid."
                )
                # Fallback: Potentially use the old generate_itinerary if it was kept and makes sense
                # For now, itinerary remains empty, leading to a response with no itinerary.
                # self.state["itinerary"] will be empty, and confirmation will reflect that.

            # Fix: convert start_date to datetime if it's a string
            if isinstance(start_date, str):
                start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
            else:
                start_date_dt = start_date
            end_date = (start_date_dt + timedelta(days=days)).strftime("%Y-%m-%d")
            # Extract the optimal route from the itinerary
            optimal_route = []
            if itinerary:
                for day_plan_item in itinerary:  # Iterate through list of day plans
                    day_number = day_plan_item.get("day")
                    for spot in day_plan_item.get("spots", []):
                        spot_with_day = spot.copy()  # Avoid modifying original spot in itinerary
                        spot_with_day["day"] = day_number
                        optimal_route.append(spot_with_day)

            # Estimate budget
            car_info = None
            fuel_price = None

            if self.state["should_rent_car"]:
                car_info = self.info_agent.search_car_rentals(
                    self.state["user_info"].get("city", ""),
                    start_date,
                    end_date,
                    driver_age=self.state["user_info"].get("age", 30),
                )
                fuel_price = self.info_agent.get_fuel_price(self.state["user_info"].get("city", ""))
                if fuel_price and car_info:
                    print(
                        f"[DEBUG] Successfully got fuel price and car info, fuel_price: {fuel_price}, car_info: {car_info}"
                    )

            budget = self.route_agent.estimate_budget(
                all_attractions_objects,
                self.state["user_info"],
                self.state["should_rent_car"],
                car_info if self.state["should_rent_car"] else None,
                fuel_price if self.state["should_rent_car"] else None,
            )

            # Store in state
            self.state["itinerary"] = itinerary
            self.state["budget"] = budget

            # Generate confirmation message
            confirmation = self.comm_agent.generate_booking_confirmation(
                itinerary,
                budget,
                self.state["should_rent_car"],
                self.state["user_info"].get("name", "Traveler"),
            )

            return {
                "next_step": "complete",
                "response": confirmation,
                "itinerary": itinerary,
                "budget": budget,
                "optimal_route": optimal_route,
            }

        except Exception as e:
            # Log the error for debugging
            print(f"Error in process route: {str(e)}")
            return {
                "next_step": "error",
                "response": "An error occurred while planning your route. Please try again.",
                "error": str(e),
            }

    def get_current_state(self):
        """Get the current state of the workflow"""
        return self.state
