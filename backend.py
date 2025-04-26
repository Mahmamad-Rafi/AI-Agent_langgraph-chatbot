# backend.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from ai_agent import get_response_from_ai_agent  # Import function from ai-agent.py

# Define schema for incoming requests using Pydantic
class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: str
    messages: List[str]
    allow_search: bool

# Initialize FastAPI app
app = FastAPI(title="LangGraph AI Agent")

# List of allowed model names
ALLOWED_MODEL_NAMES = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "llama-3-70b-8192", "gpt-4o-mini"]

# Define /chat POST endpoint
@app.post("/chat")
def chat_endpoint(request: RequestState):
    """
    API Endpoint to interact with the Chatbot using LangGraph and search tools.
    It dynamically selects the model specified in the request.
    """
    if request.model_name not in ALLOWED_MODEL_NAMES:
        return {"error": "Invalid model name. Kindly select a valid AI model."}

    # Extract values from the request
    llm_id = request.model_name
    query = request.messages
    allow_search = request.allow_search
    system_prompt = request.system_prompt
    provider = request.model_provider

    # Get response from AI Agent
    response = get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider)
    return {"response": response}

# Run FastAPI app with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9999)
