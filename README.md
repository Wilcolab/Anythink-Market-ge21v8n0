# SecureInfo Concierge

SecureInfo Concierge is an educational platform simulating a sophisticated financial assistant application. It integrates LLM capabilities with database retrievals, designed to teach secure data handling and interactions.

## Features

- FastAPI backend with RESTful services
- Azure OpenAI integration for intelligent responses
- SQLite database for storing financial data
- JWT-based authentication for secure endpoints
- Minimalist web interface for user interactions

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application (recommended for development):
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir app
   ```
   This will restart the server automatically when you change code in the `app` directory.

3. Access the application in your web browser on port 8000

4. If you want to login, use the following default credentials:
   - Username: johndoe
   - Password: secret

## Project Structure

- `app/`: Main application directory
  - `api/`: API routes and endpoints
  - `auth/`: Authentication related code
  - `database/`: Database connection and queries
  - `models/`: LLM service implementation
  - `static/`: Static assets (CSS, JavaScript)
  - `templates/`: HTML templates
- `data/`: Directory for storing the SQLite database
- `Dockerfile`: Instructions for building the Docker image
- `docker-compose.yml`: Docker Compose configuration
- `requirements.txt`: Python dependencies

## Main Components

- **API Endpoint**: The `app/api/secure-query` endpoint processes all financial queries and returns AI-generated responses
- **LLM Service**: The `app/models/llm_service.py` manages OpenAI interactions and query interpretation
- **Dashboard**: The main interface at `app/templates/dashboard.html` for submitting queries and viewing responses
- **Login Page**: The authentication interface at `app/templates/index.html` for user access management

