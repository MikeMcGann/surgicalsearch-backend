import io
import os
import asyncpg
from urllib.parse import urlparse, unquote
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models

app = FastAPI(title="Surgical Visual Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
preprocess = None
projection = None
db_pool = None

DATABASE_URL = os.getenv("DATABASE_URL")

@app.on_event("startup")
async def startup_event():
    global model, preprocess, projection, db_pool
    
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL environment variable is missing!")

    raw_url = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    parsed = urlparse(raw_url)
    
    user = parsed.username
    password = unquote(parsed.password) if parsed.password else None
    host = parsed.hostname
    port = parsed.port or 6543
    database = parsed.path.lstrip('/') or 'postgres'

    # Initialize MobileNetV3 Small without forcing CDN download retries
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    base_model = models.mobilenet_v3_small(weights=weights)
    base_model.eval()
    model = base_model
    preprocess = weights.transforms()
    
    # Projection layer to match pgvector(512)
    torch.manual_seed(42)
    projection = nn.Linear(576, 512)
    projection.eval()

    try:
        db_pool = await asyncpg.create_pool(
            user=user,
            password=password,
            host=host,
            port=port,
            database=database,
            ssl="require",
            min_size=1,
            max_size=5,
            statement_cache_size=0  # Required for Supabase PgBouncer (Port 6543)
        )
    except Exception as e:
        print(f"Failed to connect to Supabase: {e}")
        raise e

@app.on_event("shutdown")
async def shutdown_event():
    if db_pool:
        await db_pool.close()

def generate_image_embedding(image_bytes: bytes) -> list[float]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    processed_img = preprocess(image).unsqueeze(0)
    
    with torch.no_grad():
        features = model.features(processed_img)
        features = model.avgpool(features)
        features = torch.flatten(features, 1)
        features_512 = projection(features)
        features_512 /= features_512.norm(dim=-1, keepdim=True)
    
    return features_512.cpu().numpy().flatten().tolist()

@app.post("/api/search-visual")
@app.get("/search")
def search_text(q: str = "", category: str = "All"):
    # Keep your existing code inside this function
async def search_visual(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    image_bytes = await file.read()
    query_vector = generate_image_embedding(image_bytes)
    vector_str = f"[{','.join(map(str, query_vector))}]"

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
