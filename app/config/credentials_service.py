import os
import requests

class CredentialsService:
    """Service for fetching Azure OpenAI credentials from a server"""
    
    def __init__(self):
        """Initialize the credentials service"""
        self.base_url = None
        self.api_key = None
        self.deployment_name = None
        self.api_version = None
        self.initialized = False
        self.credentials = None
        
        self.server_url = os.environ.get("ENGINE_WILCO_AI_URL")
        if not self.server_url:
            raise ValueError("Missing ENGINE_WILCO_AI_URL environment variable")
        
    def fetch_credentials(self):
        """Fetch credentials from the server"""
        try:
            response = requests.get(self.server_url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract credentials from response
            self.base_url = data.get("baseUrl")
            self.api_key = data.get("apiKey")
            self.deployment_name = data.get("llmDeployment")
            self.api_version = data.get("apiVersion", "2025-01-01-preview")
            
            if not all([self.base_url, self.api_key, self.deployment_name]):
                raise ValueError("Incomplete credentials received from server")
            
            self.initialized = True
            self.credentials = data
            return True
            
        except requests.exceptions.RequestException:
            raise
    
    def get_credentials(self):
        """Get the credentials"""
        if not self.initialized:
            self.fetch_credentials()
        
        return {
            "base_url": self.base_url,
            "api_key": self.api_key,
            "deployment_name": self.deployment_name,
            "api_version": self.api_version
        }
