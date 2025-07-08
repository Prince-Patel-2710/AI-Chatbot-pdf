from fastapi import FastAPI
from upload.routes import upload_router
from summarize.routes import summarize_router
from chat.routes import chat_router
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os  # Needed to check the key

# Load environment variables from .env file
load_dotenv()

# Print the key to check if it's loaded correctly (for debugging only)
print("🔑 AI_KEY =", os.getenv("AI_KEY"))

# Create FastAPI app
app = FastAPI()

# Allow frontend to connect to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload_router)
app.include_router(summarize_router)
app.include_router(chat_router)
