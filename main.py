import io
import os
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
import torch
from torchvision import models, transforms
from PIL import Image

app = FastAPI(title="Surgical Search API")

# Enable CORS for frontend web and mobile app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase database initialization
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# Only initialize client if BOTH credentials are present
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if (SUPABASE_URL and SUPABASE_KEY) else None

# Initialize MobileNetV3 model for embedding extraction
weights = models.MobileNet_V3_Small_Weights.DEFAULT
model = models.mobilenet_v3_small(weights=weights)
model.classifier = torch.nn.Identity()  # Output raw feature embeddings
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@app.get("/")
def root():
    return {"status": "ok", "message": "Surgical Search Backend API is running!"}


# 1. Text Search Endpoint
@app.get("/search")
def search_text(q: str = Query("", alias="q"), category: str = "All"):
    if not supabase:
        return {"results": [{"title": "Demo Instrument", "description": "Ensure SUPABASE_URL and SUPABASE_KEY are set in Render."}]}

    query = supabase.table("instruments").select("*")
    if q:
        query = query.ilike("name", f"%{q}%")
    if category and category != "All":
        query = query.eq("category", category)

    response = query.execute()
    return {"results": response.data}


# 2. Visual Image Search Endpoint
@app.post("/api/search-visual")
async def search_visual(file: UploadFile = File(...)):
    # Read uploaded image bytes
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Generate PyTorch embedding vector
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(input_tensor).squeeze().tolist()

    if not supabase:
        return {
            "results": [{
                "title": "Visual Match (Demo)",
                "description": "PyTorch embedding generated successfully. Add SUPABASE_KEY to search database."
            }]
        }

    # Match vector embedding using RPC function in Supabase
    rpc_response = supabase.rpc("match_instruments", {
        "query_embedding": embedding, 
        "match_threshold": 0.5, 
        "match_count": 5
    }).execute()
    
    return {"results": rpc_response.data}
