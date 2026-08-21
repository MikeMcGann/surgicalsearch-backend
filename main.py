import io
import os
import asyncpg
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import open_clip

app = FastAPI(title="Surgical Visual Search API")

# Allow mobile PWA cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows your mobile app to connect from anywhere
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and database connection pool
model = None
preprocess = None
db_pool = None

DATABASE_URL = os.getenv("DATABASE_URL")

@app.on_event("startup")
async def startup_event():
    global model, preprocess, db_pool
    
    # 1. Load Lightweight OpenCLIP Vision Model
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model.eval()
    
    # 2. Setup Async PostgreSQL Connection Pool using your Render Environment Variable
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL environment variable is missing!")
        
    db_pool = await asyncpg.create_pool(DATABASE_URL)

@app.on_event("shutdown")
async def shutdown_event():
    if db_pool:
        await db_pool.close()

def generate_image_embedding(image_bytes: bytes) -> list[float]:
    """Converts image bytes from your phone into a 512-dimensional vector embedding."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    processed_img = preprocess(image).unsqueeze(0)
    
    with torch.no_grad():
        image_features = model.encode_image(processed_img)
        # Normalize the feature vector
        image_features /= image_features.norm(dim=-1, keepdim=True)
    
    return image_features.cpu().numpy().flatten().tolist()

@app.post("/api/search-visual")
async def search_visual(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    # Read uploaded photo from phone and convert to vector math
    image_bytes = await file.read()
    query_vector = generate_image_embedding(image_bytes)
    vector_str = f"[{','.join(map(str, query_vector))}]"

    # Query Supabase using pgvector cosine distance operator (<=>)
    query = """
        SELECT 
            p.product_id,
            p.sku,
            p.pattern_name,
            p.title,
            m.name AS manufacturer,
            pi.image_url,
            1 - (pi.embedding <=> $1::vector) AS similarity_score
        FROM product_images pi
        JOIN products p ON pi.product_id = p.product_id
        JOIN manufacturers m ON p.manufacturer_id = m.manufacturer_id
        ORDER BY pi.embedding <=> $1::vector ASC
        LIMIT 3;
    """
    
    async with db_pool.acquire() as conn:
        records = await conn.fetch(query, vector_str)
        
    results = [
        {
            "product_id": str(r["product_id"]),
            "sku": r["sku"],
            "pattern_name": r["pattern_name"],
            "title": r["title"],
            "manufacturer": r["manufacturer"],
            "image_url": r["image_url"],
            "confidence_score": round(float(r["similarity_score"]) * 100, 1)
        }
        for r in records
    ]
    
    return {"status": "success", "matches": results}
