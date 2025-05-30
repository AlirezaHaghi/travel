from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, List
import os
import json
import uuid
from dotenv import load_dotenv
from workflows.travel_graph import TravelGraph
import asyncio

# Load environment variables
load_dotenv()

app = FastAPI(title="SmartHook Travel API")

# Initialize the shared TravelGraph singleton on startup
# This ensures agents are created once when the application starts
print("[STARTUP] Initializing shared TravelGraph agents...")
_shared_graph = TravelGraph.get_shared_agents()
print("[STARTUP] Shared TravelGraph agents initialized successfully")

# Setup static files and templates
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")

# In-memory session storage (in production, use Redis or similar)
session_storage = {}


def get_or_create_session(session_id: Optional[str] = None) -> TravelGraph:
    """Get existing session or create a new one with shared agents"""
    if session_id is None:
        session_id = str(uuid.uuid4())

    if session_id not in session_storage:
        # Create new session-specific TravelGraph instance that uses shared agents
        session_storage[session_id] = TravelGraph(session_id=session_id)
        print(f"[DEBUG] Created new session: {session_id}")

    return session_storage[session_id]


# Pydantic models for request bodies
class ProcessRequest(BaseModel):
    step: Optional[str] = "chat"
    user_input: Optional[str] = ""
    selected_attraction_ids: Optional[List[str]] = None
    session_id: Optional[str] = None


class StreamParams(BaseModel):
    step: Optional[str] = "chat"
    user_input: Optional[str] = ""
    selected_attraction_ids: Optional[str] = None
    session_id: Optional[str] = None


@app.get("/test-image")
async def test_image():
    """Test image endpoint"""
    return FileResponse("frontend/static/images/background.jpg", media_type="image/jpeg")


@app.get("/static/images/{filename}")
async def serve_image(filename: str):
    """Serve image files"""
    return FileResponse(f"frontend/static/images/{filename}")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the main page"""
    # Load popular attractions
    try:
        with open('frontend/data/popular_attractions.json', 'r') as f:
            popular_attractions = json.load(f)
    except FileNotFoundError:
        popular_attractions = []

    return templates.TemplateResponse("index.html", {"request": request, "popular_attractions": popular_attractions})


@app.post("/api/process")
async def process(request: Request, data: ProcessRequest):
    """Process a step in the travel planning workflow"""
    try:
        # Get or create workflow instance for this session
        workflow = get_or_create_session(data.session_id)

        print(f"[DEBUG] Processing step: {data.step} for session: {workflow.session_id}")

        # Process the current step
        result = workflow.process_step(
            data.step,
            user_input=data.user_input,
            selected_attraction_ids=data.selected_attraction_ids,
        )

        # Add the current state and session_id to the result
        result['state'] = workflow.get_current_state()
        result['session_id'] = workflow.session_id
        return JSONResponse(content=result)

    except Exception as e:
        print(f"[ERROR] in process route: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/attractions/{city}")
async def get_attractions(city: str, request: Request, session_id: Optional[str] = None):
    """Get attractions for a specific city"""
    # Get or create workflow instance for this session
    workflow = get_or_create_session(session_id)
    info_agent = workflow.info_agent

    print(f"[DEBUG] Getting attractions for {city} - session: {workflow.session_id}")

    # Convert city to coordinates
    city_coordinates = info_agent.city2geocode(city)
    if not city_coordinates:
        raise HTTPException(status_code=400, detail=f"Could not find coordinates for {city}")

    # Use default user preferences
    user_prefs = {"city": city}

    try:
        attractions = info_agent.get_attractions(
            lat=city_coordinates["lat"],
            lng=city_coordinates["lng"],
            user_prefs=user_prefs,
            weather_summary="",  # Empty string for weather summary
            number=20,
        )
        return JSONResponse(content=attractions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get attractions: {str(e)}")


@app.get("/api/reset")
async def reset_session(request: Request, session_id: Optional[str] = None):
    """Reset a specific session or create a new one"""
    if session_id and session_id in session_storage:
        # Reset existing session
        session_storage[session_id] = TravelGraph(session_id=session_id)
        print(f"[DEBUG] Reset session: {session_id}")
    else:
        # Create new session
        new_session = get_or_create_session()
        session_id = new_session.session_id
        print(f"[DEBUG] Created new session: {session_id}")

    return JSONResponse(content={"status": "reset successful", "session_id": session_id})


@app.get("/api/sessions")
async def get_sessions_status():
    """Get information about active sessions and shared agents"""
    return JSONResponse(
        content={
            "active_sessions": len(session_storage),
            "session_ids": list(session_storage.keys()),
            "shared_agents_initialized": TravelGraph._singleton is not None,
            "singleton_id": id(TravelGraph._singleton) if TravelGraph._singleton else None,
        }
    )


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session"""
    if session_id in session_storage:
        del session_storage[session_id]
        print(f"[DEBUG] Deleted session: {session_id}")
        return JSONResponse(content={"status": "session deleted", "session_id": session_id})
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@app.get("/api/stream")
async def stream(
    request: Request,
    step: str = "chat",
    user_input: str = "",
    selected_attraction_ids: Optional[str] = None,
    session_id: Optional[str] = None,
):
    """Handle streaming responses"""
    # Get or create workflow instance for this session
    workflow = get_or_create_session(session_id)

    print(f"[DEBUG] Streaming step: {step} for session: {workflow.session_id}")

    # Parse selected_attraction_ids if provided
    parsed_attraction_ids = None
    if selected_attraction_ids:
        try:
            parsed_attraction_ids = json.loads(selected_attraction_ids)
        except json.JSONDecodeError:
            parsed_attraction_ids = None

    # Check if the user is confirming satisfaction with the recommendation
    satisfaction_message = 'satisfied with your recommendation' in user_input.lower()

    if satisfaction_message:
        print(f"[DEBUG] Detected satisfaction message: '{user_input}'")

    async def generate():
        try:
            # Process the step
            result = workflow.process_step(step, user_input=user_input, selected_attraction_ids=parsed_attraction_ids)

            # Check the should_rent_car status right after processing
            current_should_rent_car = workflow.get_current_state().get('should_rent_car', False)
            print(f"[DEBUG] After processing step, should_rent_car = {current_should_rent_car}")

            # Handle streaming response
            if 'stream' in result and result['stream']:
                for chunk in result['stream']:
                    # Handle both string chunks and objects with content attribute
                    content = ""
                    if hasattr(chunk, 'content'):
                        content = chunk.content
                    elif isinstance(chunk, str):
                        content = chunk
                    else:
                        content = str(chunk)

                    if content:
                        yield f"data: {{\"type\": \"chunk\", \"content\": {json.dumps(content)} }}\n\n"
                        await asyncio.sleep(0.01)

            # Send completion data
            completion_data = {
                'type': 'complete',
                'next_step': result.get('next_step'),
                'missing_fields': result.get('missing_fields', []),
                'state': result.get('state'),
                'attractions': result.get('attractions'),
                'map_data': result.get('map_data'),
                'itinerary': result.get('itinerary'),
                'budget': result.get('budget'),
                'response': result.get('response'),
                'optimal_route': result.get('optimal_route'),
                'rental_post': result.get('rental_post'),
            }

            # Handle next step logic for strategy step
            if step == 'strategy':
                current_state = workflow.get_current_state()
                ai_recommendation_generated = current_state.get('ai_recommendation_generated', False)
                should_rent_car = current_state.get('should_rent_car', False)

                print(
                    f"[DEBUG] In stream endpoint, strategy step: ai_recommendation_generated={ai_recommendation_generated}, should_rent_car={should_rent_car}, satisfaction_message={satisfaction_message}"
                )

                # If the AI has provided recommendations
                if ai_recommendation_generated or satisfaction_message:
                    next_step = 'communication' if should_rent_car else 'route'
                    completion_data['next_step'] = next_step
                    print(f"[DEBUG] Setting next_step to '{next_step}' based on should_rent_car={should_rent_car}")

            yield f"data: {json.dumps(completion_data)}\n\n"

        except Exception as e:
            print(f"[ERROR] in stream route: {str(e)}")
            yield f"data: {{\"type\": \"error\", \"error\": {json.dumps(str(e))} }}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/nearby/{attraction_id}")
async def get_nearby_places(attraction_id: str, request: Request, session_id: Optional[str] = None):
    """Get nearby restaurants and street information for an attraction"""
    # Get or create workflow instance for this session
    workflow = get_or_create_session(session_id)
    info_agent = workflow.info_agent

    print(f"[DEBUG] Getting nearby places for {attraction_id} - session: {workflow.session_id}")

    # Parse coordinates from attraction_id
    try:
        lat_str, lng_str = attraction_id.split(',')
        lat, lng = float(lat_str), float(lng_str)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid coordinates format. Use 'lat,lng'.")

    try:
        result = info_agent.search_nearby_places(lat, lng)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get nearby places: {str(e)}")


if __name__ == '__main__':
    import uvicorn

    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # Create a sample attractions.json file if it doesn't exist
    if not os.path.exists('data/attractions.json'):
        sample_data = {
            "Paris": [
                {
                    "id": "eiffel_tower",
                    "name": "Eiffel Tower",
                    "category": "landmark",
                    "location": {"lat": 48.8584, "lng": 2.2945},
                    "estimated_duration": 3,
                    "price_level": 3,
                },
                {
                    "id": "louvre_museum",
                    "name": "Louvre Museum",
                    "category": "museum",
                    "location": {"lat": 48.8606, "lng": 2.3376},
                    "estimated_duration": 4,
                    "price_level": 2,
                },
            ],
            "New York": [
                {
                    "id": "central_park",
                    "name": "Central Park",
                    "category": "nature",
                    "location": {"lat": 40.7812, "lng": -73.9665},
                    "estimated_duration": 3,
                    "price_level": 0,
                },
                {
                    "id": "empire_state_building",
                    "name": "Empire State Building",
                    "category": "landmark",
                    "location": {"lat": 40.7484, "lng": -73.9857},
                    "estimated_duration": 2,
                    "price_level": 3,
                },
            ],
        }

        with open('data/attractions.json', 'w') as f:
            json.dump(sample_data, f)

    # Run the app
    uvicorn.run(app, host="127.0.0.1", port=8000)
