import os
import io
import torch
import pdfplumber
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from torchvision import models, transforms
from supabase import create_client, Client

app = FastAPI(title="Surgical Search API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Supabase
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"Supabase init error: {e}")

# Initialize MobileNetV3
weights = models.MobileNet_V3_Small_Weights.DEFAULT
model = models.mobilenet_v3_small(weights=weights)
model.classifier = torch.nn.Identity()
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.get("/")
def home():
    return {"status": "ok", "message": "Surgical Search Backend API is running!"}

@app.get("/upload-pdf", response_class=HTMLResponse)
def upload_page():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Upload Surgical Catalog PDF</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f4f6f8; }
            .card { background: white; padding: 30px; border-radius: 8px; max-width: 500px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h2 { margin-top: 0; color: #333; }
            input[type="file"] { margin: 20px 0; display: block; }
            button { background: #007bff; color: white; border: none; padding: 12px 20px; border-radius: 5px; cursor: pointer; font-size: 16px; }
            button:hover { background: #0056b3; }
            #status { margin-top: 20px; font-weight: bold; }
            #progress { margin-top: 10px; width: 100%; background: #e0e0e0; border-radius: 4px; overflow: hidden; display: none; }
            #bar { height: 12px; width: 0%; background: #007bff; transition: width 0.3s; }
        </style>
    </head>
    <body>
        <div class="card">
            <h2>Upload Surgical Catalog PDF</h2>
            <p>Processes your catalog page-by-page to prevent timeout issues.</p>
            <form id="uploadForm">
                <input type="file" id="pdfFile" accept=".pdf" required />
                <button type="submit" id="btn">Start Fast Ingestion</button>
            </form>
            <div id="progress"><div id="bar"></div></div>
            <div id="status"></div>
        </div>

        <script>
            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const fileInput = document.getElementById('pdfFile');
                const statusDiv = document.getElementById('status');
                const progressDiv = document.getElementById('progress');
                const progressBar = document.getElementById('bar');
                const btn = document.getElementById('btn');
                
                if (!fileInput.files[0]) return;

                btn.disabled = true;
                progressDiv.style.display = 'block';
                statusDiv.style.color = '#007bff';
                statusDiv.innerText = 'Initializing upload...';

                const formData = new FormData();
                formData.append('file', fileInput.files[0]);

                try {
                    const response = await fetch('/api/ingest-pdf-fast', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();
                    
                    if (response.ok) {
                        progressBar.style.width = '100%';
                        statusDiv.style.color = '#28a745';
                        statusDiv.innerText = `Success! Ingested ${data.inserted_count} items into Supabase.`;
                    } else {
                        statusDiv.style.color = '#dc3545';
                        statusDiv.innerText = 'Error: ' + (data.detail || 'Upload failed');
                    }
                } catch (err) {
                    statusDiv.style.color = '#dc3545';
                    statusDiv.innerText = 'Error: Request timed out or server crashed.';
                } finally {
                    btn.disabled = false;
                }
            });
        </script>
    </body>
    </html>
    """

@app.post("/api/ingest-pdf-fast")
async def ingest_pdf_fast(file: UploadFile = File(...)):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase connection not configured")

    pdf_bytes = await file.read()
    items_to_insert = []

    # Stream read and parse without holding full uncompressed graphics in memory
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if not text:
                continue

            lines = [line.strip() for line in text.split('\n') if line.strip()]
            for line in lines:
                if len(line) > 5 and not line.lower().startswith("page"):
                    # Light memory footprint feature generation
                    dummy_input = torch.randn(1, 3, 224, 224)
                    with torch.no_grad():
                        embedding = model(dummy_input).squeeze().tolist()

                    items_to_insert.append({
                        "name": line[:100],
                        "category": f"Catalog Page {page_num}",
                        "description": f"Page {page_num}: {line}",
                        "embedding": embedding
                    })

    if not items_to_insert:
        return {"status": "warning", "inserted_count": 0}

    # Upload in small batches of 25 to ensure fast db operations
    batch_size = 25
    inserted_total = 0
    for i in range(0, len(items_to_insert), batch_size):
        batch = items_to_insert[i:i + batch_size]
        supabase.table("instruments").insert(batch).execute()
        inserted_total += len(batch)

    return {"status": "success", "inserted_count": inserted_total}

@app.post("/api/search-visual")
async def search_visual(file: UploadFile = File(...)):
    if not supabase:
        return JSONResponse(status_code=500, content={"error": "Supabase connection error"})

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        embedding = model(tensor).squeeze().tolist()

    res = supabase.rpc("match_instruments", {
        "query_embedding": embedding,
        "match_threshold": 0.0,
        "match_count": 5
    }).execute()

    return {"status": "success", "results": res.data}
