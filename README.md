# This is a code repo for Application Track of CS194

![poster](Vaiage.svg "poster")

## Overview

Vaiage is an AI-powered travel planning application designed to help users create personalized and optimized travel itineraries. It assists with everything from initial preference gathering to detailed daily plans, including attraction recommendations, route optimization, budget estimation, and car rental advice.

## Features

- **Conversational AI Chatbot**: Collects user travel preferences (destination, duration, budget, interests, etc.) through an interactive chat.
- **Personalized Attraction Recommendations**: Leverages AI to suggest points of interest tailored to user preferences, considering factors like weather, ratings, and price levels.
- **Dynamic Itinerary Planning**: Generates optimized daily travel plans, distributing selected attractions efficiently across the trip duration.
- **Route Optimization**: Calculates optimal travel routes between planned attractions to minimize travel time.
- **Weather Integration**: Fetches weather forecasts and incorporates this information into travel planning and recommendations.
- **Car Rental Assistance**: Provides AI-driven advice on whether a car rental is beneficial and can search for car rental options.
- **Budget Estimation**: Estimates the overall trip budget, breaking down costs for accommodation, food, transport, attractions, and potential car rental/fuel expenses.
- **Map Data Generation**: Prepares data for visualizing attractions and routes on a map interface.

## Tech Stack

- **Core Language**: Python
- **AI & LLMs**: Langchain, OpenAI API (GPT-4o, GPT-3.5-turbo)
- **Mapping & Geolocation**: OpenStreetMap APIs (Nominatim Geocoding, Overpass Places, OSRM Directions) - 100% FREE!
- **Car Rentals**: RapidAPI (Booking.com API endpoint)
- **Weather**: Open-Meteo API
- **Fuel Prices**: Custom local data source (`data/global_fuel_prices.json`) and OpenAI for country/price estimation.
- **Networking**: `http.client` (for car rental API), `requests` (for weather API)
- **Route Optimization**: `networkx` (for TSP solving)

## Usage

1.  **Get API Keys**: Obtain your own API keys for:
    *   OpenAI (for AI features)
    *   RapidAPI (for car rental services, optional)
    *   OpenStreetMap (NO API KEY REQUIRED!)
2.  **Clone the Repository**: `git clone <repository_url>`
3.  **Install Dependencies**: `uv sync` or `pip install -r requirements.txt`
4.  **Configure API Keys**: Set your API keys as environment variables or in a `.env` file
5.  **Run the Application**: `uv run python main.py` or `python main.py`

## 🚀 Features

- **AI-Powered Travel Planning**: Create personalized itineraries based on your preferences
- **Weather Integration**: Real-time weather data for better travel planning
- **Smart Route Optimization**: Efficient routing between destinations using OpenStreetMap
- **Points of Interest**: Discover attractions, restaurants, and local gems
- **Multi-Modal Transportation**: Support for driving, walking, biking, and public transit
- **Interactive Maps**: Beautiful Leaflet-powered maps with zero API costs
- **Responsive Design**: Works seamlessly on desktop and mobile devices

## 📋 Prerequisites

*   Python 3.11 or higher
*   OpenAI API key (for AI features)
*   RapidAPI key (for car rental services, optional)
*   OpenStreetMap (NO API KEY REQUIRED!)

## 🛠️ Installation
