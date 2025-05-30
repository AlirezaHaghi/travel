# Travel Graph Optimization - Singleton Pattern

## Problem Solved

Previously, the `TravelGraph` was being created fresh for every API request, which meant:

- **InformationAgent** was initialized for every request
- **ChatAgent** was initialized for every request  
- **StrategyAgent** was initialized for every request
- **RecommendAgent** was initialized for every request
- **RouteAgent** was initialized for every request
- **CommunicationAgent** was initialized for every request

This was extremely inefficient as these agents likely load models, establish connections, and perform other expensive initialization operations.

## Solution Implemented

### Singleton Pattern for Agents

We implemented a singleton pattern that:

1. **Creates agents once** when the application starts
2. **Reuses the same agent instances** across all requests
3. **Maintains separate state** for each user session

### Architecture

```
TravelGraphSingleton (Created Once)
├── InformationAgent (Shared)
├── ChatAgent (Shared)  
├── StrategyAgent (Shared)
├── RecommendAgent (Shared)
├── RouteAgent (Shared)
└── CommunicationAgent (Shared)

TravelGraph Instances (Per Session)
├── session_id: "unique-session-id"
├── state: {} (Independent per session)
└── References to shared agents
```

### Key Benefits

1. **Performance**: Agents are created only once during application startup
2. **Memory Efficiency**: Multiple requests share the same agent instances
3. **Session Isolation**: Each user session maintains its own state
4. **Scalability**: Better resource utilization for concurrent users

### Usage

The API now automatically manages sessions:

```python
# Create or get existing session
workflow = get_or_create_session(session_id)

# Each session has separate state but shared agents
workflow.state  # Unique to this session
workflow.info_agent  # Shared across all sessions
```

### Session Management Endpoints

- `GET /api/sessions` - View active sessions and singleton status
- `GET /api/reset?session_id=<id>` - Reset or create a session
- `DELETE /api/sessions/<session_id>` - Delete a specific session

### Testing

You can verify the singleton is working:

```bash
uv run python -c "
from workflows.travel_graph import TravelGraph
t1 = TravelGraph('session1')
t2 = TravelGraph('session2')
print(f'Agents shared: {t1.info_agent is t2.info_agent}')
print(f'States separate: {id(t1.state) != id(t2.state)}')
"
```

Expected output:
```
Agents shared: True
States separate: True
```

## Implementation Details

### Files Modified

1. **`workflows/travel_graph.py`**: Added `TravelGraphSingleton` class and modified `TravelGraph` to use shared agents
2. **`main.py`**: Added session management and updated all endpoints to use the singleton pattern

### Thread Safety

The current implementation uses in-memory session storage. For production deployment with multiple workers, consider using:

- Redis for session storage
- Proper locking mechanisms
- Database-backed session management 