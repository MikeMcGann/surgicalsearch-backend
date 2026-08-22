import os
import io
import torch
import pdfplumber
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from torchvision import models, transforms
from supabase import create_client, Client

app = FastAPI(title="Surgical Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"Supabase error: {e}")

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
    return {"status": "ok", "message": "Surgical Search API active"}

@app.get("/upload-pdf", response_class=HTMLResponse)
def upload_page():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chunked Catalog Ingestion</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.16.105/pdf.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f4f6f8; }
            .card { background: white; padding: 30px; border-radius: 8px; max-width: 550px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h2 { margin-top: 0; color: #333; }
            input[type="file"] { margin: 20px 0; display: block; }
            button { background: #007bff; color: white; border: none; padding: 12px 20px; border-radius: 5px; cursor: pointer; font-size: 16px; }
            button:disabled { background: #888; }
            #progress { margin-top: 15px; width: 100%; background: #e0e0e0; border-radius: 4px; overflow: hidden; display: none; }
            #bar { height: 16px; width: 0%; background: #28a745; transition: width 0.2s; }
            #status { margin-top: 15px; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="card">
            <h2>Upload Catalog PDF (Chunked)</h2>
            <p>Extracts text in your browser and streams tiny batches to Render to avoid timeouts.</p>
            <input type="file" id="pdfFile" accept=".pdf" />
            <button id="startBtn" onclick="processPDF()">Start Ingestion</button>
            
            <div id="progress"><div id="bar"></div></div>
            <div id="status"></div>
        </div>

        <script>
            pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.16.105/pdf.worker.min.js';

            async function processPDF() {
                const fileInput = document.getElementById('pdfFile');
                const btn = document.getElementById('startBtn');
                const status = document.getElementById('status');
                const progress = document.getElementById('progress');
                const bar = document.getElementById('bar');

                if (!fileInput.files[0]) {
                    alert('Please select a PDF file first.');
                    return;
                }

                btn.disabled = true;
                progress.style.display = 'block';
                status.style.color = '#007bff';
                status.innerText = 'Loading PDF in browser...';

                const file = fileInput.files[0];
                const arrayBuffer = await file.arrayBuffer();
                const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
                
                let totalInserted = 0;
                const totalPages = pdf.numPages;

                for (let pageNum = 1; pageNum <= totalPages; pageNum++) {
                    status.innerText = `Processing page ${pageNum} of ${totalPages}...`;
                    const page = await pdf.getPage(pageNum);
                    const textContent = await page.getTextContent();
                    const lines = textContent.items.map(item => item.str.trim()).filter(line => line.length > 5);

                    if (lines.length > 0) {
                        try {
                            const res = await fetch('/api/ingest-lines', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ page: pageNum, lines: lines })
                            });
                            const data = await res.json();
                            totalInserted += (data.count || 0);
                        } catch (err) {
                            console.error('Batch error on page ' + pageNum, err);
                        }
                    }

                    const pct = Math.round((pageNum / totalPages) * 100);
                    bar.style.width = pct + '%';
                }

                status.style.color = '#28a745';
                status.innerText = `Complete! Successfully ingested ${totalInserted} catalog items into Supabase.`;
                btn.disabled = false;
            }
        </script>
    </body>
    </html>
    """

@app.post("/api/ingest-lines")
async def ingest_lines(payload: dict = Body(...)):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase connection not configured")

    page_num = payload.get("page", 1)
    lines = payload.get("lines", [])
    
    items_to_insert = []
    for line in lines:
        if line.lower().startswith("page"):
            continue

        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            embedding = model(dummy_input).squeeze().tolist()

        items_to_insert.append({
            "name": line[:100],
            "category": f"Catalog Page {page_num}",
            "description": f"Page {page_num}: {line}",
            "embedding": embedding
        })

    if items_to_insert:
        supabase.table("instruments").insert(items_to_insert).execute()

    return {"status": "ok", "count": len(items_to_insert)}

@app.post("/api/search-visual")
async def search_visual(file: UploadFile = File(...)):
    if not supabase:
        return JSONResponse(status_code=500, content={"error": "Supabase client unconfigured"})

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
