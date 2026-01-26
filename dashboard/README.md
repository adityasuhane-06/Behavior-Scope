# Behavior Scope Dashboard

Interactive React dashboard for visualizing Behavior Scope analysis results.

## Features

- **Session Management**: Browse and search through all analysis sessions
- **Real-time Data**: View live analysis results through REST API
- **Interactive Charts**: Visualize behavioral scores over time
- **Frame-level Analysis**: Explore detailed frame-by-frame detection data
- **Statistics**: View aggregate statistics across all sessions

## Prerequisites

- Node.js (v16 or higher)
- npm or yarn
- Python 3.8+ (for backend API)

## Installation

1. Install frontend dependencies:
```bash
cd dashboard
npm install
```

2. Install backend dependencies (if not already installed):
```bash
pip install fastapi uvicorn
```

## Running the Application

You need to run both the backend API server and the frontend development server.

### 1. Start the Backend API Server

From the project root directory:

```bash
python utils/api_server.py
```

The API server will start at `http://localhost:8000`
- API documentation: `http://localhost:8000/docs`

### 2. Start the Frontend Development Server

In a new terminal, from the dashboard directory:

```bash
cd dashboard
npm run dev
```

The dashboard will start at `http://localhost:3000`

## Usage

1. Open your browser and navigate to `http://localhost:3000`
2. Browse the list of analysis sessions
3. Click on any session to view detailed results
4. Use the tabs to switch between:
   - **Score Charts**: Line graphs showing behavioral metrics over time
   - **Frame Analysis**: Detailed table view of frame-level detections
5. Navigate to the Statistics page to see aggregate data

## Building for Production

To create a production build:

```bash
cd dashboard
npm run build
```

The optimized files will be in the `dashboard/dist` directory.

To preview the production build:

```bash
npm run preview
```

## API Endpoints

The dashboard uses the following API endpoints:

- `GET /sessions` - List all sessions
- `GET /sessions/{session_id}` - Get session details
- `GET /sessions/{session_id}/frames` - Get frame data
- `GET /statistics` - Get aggregate statistics
- `GET /search` - Search sessions by criteria

## Technology Stack

- **Frontend**: React 18, Vite, Material-UI, Recharts
- **Backend**: FastAPI, Python
- **Database**: SQLite (via AuditDatabase)

## Troubleshooting

### CORS Issues
If you encounter CORS errors, ensure:
- Backend is running on port 8000
- Frontend is running on port 3000
- CORS middleware is properly configured in `utils/api_server.py`

### API Connection Failed
- Verify the backend server is running
- Check that the database file exists
- Ensure no firewall is blocking the connection

### Missing Data
- Run at least one analysis session to populate the database
- Check that `audit_database.db` exists in the project root
