import pytest
import requests

class TestAPIService:
    @pytest.fixture
    def base_url(self):
        """Create a base URL fixture"""
        return "http://cheese-app-api-service:9000"  # Using the service name in docker network
    
    def test_root_endpoint(self, base_url):
        """Test the root endpoint returns correct welcome message"""
        response = requests.get(f"{base_url}/")
        assert response.status_code == 200
        assert response.json() == {"message": "Welcome to AC215"}
    
    def test_llm_rag_health(self, base_url):
        """Test the LLM RAG health endpoint"""
        response = requests.get(f"{base_url}/llm-rag/health")
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_llm_rag_chat_endpoint(self, base_url):
        """Test the chat endpoint with a basic query"""
        test_query = {
            "query": "test question",
            "conversation_id": "test-convo"
        }
        response = requests.post(
            f"{base_url}/llm-rag/chat",
            json=test_query
        )
        assert response.status_code in [200, 422]  # 422 if validation fails
        if response.status_code == 200:
            assert "response" in response.json()

    def test_api_is_running(self, base_url):
        """Test if the API is accessible"""
        try:
            response = requests.get(f"{base_url}/")
            assert response.status_code == 200
            print("API is running and accessible")
        except requests.exceptions.ConnectionError:
            pytest.fail("API is not running or not accessible")