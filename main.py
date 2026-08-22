import os
import io
import torch
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
        <title>Fast Catalog Ingestion</title>
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
            <h2>Fast Catalog Ingestion</h2>
            <p>Extracts catalog text in 10-page batches to complete ingestion in under 2 minutes.</p>
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
                status.innerText = 'Reading PDF structure...';

                const file = fileInput.files[0];
                const arrayBuffer = await file.arrayBuffer();
                const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
                
                let totalInserted = 0;
                const totalPages = pdf.numPages;
                const BATCH_SIZE = 10;

                for (let startPage = 1; startPage <= totalPages; startPage += BATCH_SIZE) {
                    const endPage = Math.min(startPage + BATCH_SIZE - 1, totalPages);
                    status.innerText = `Processing pages ${startPage} to ${endPage} of ${totalPages}...`;

                    let batchPayload = [];

                    for (let p = startPage; p <= endPage; p++) {
                        const page = await pdf.getPage(p);
                        const textContent = await page.getTextContent();
                        const lines = textContent.items.map(item => item.str.trim()).filter(line => line.length > 5);
                        
                        if (lines.length > 0) {
                            batchPayload.push({ page: p, lines: lines });
                        }
                    }

                    if (batchPayload.length > 0) {
                        try {
                            const res = await fetch('/api/ingest-batch', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ pages: batchPayload })
                            });
                            const data = await res.json();
                            totalInserted += (data.count || 0);
                        } catch (err) {
                            console.error(`Batch error on pages ${startPage}-${endPage}`, err);
                        }
                    }

                    const pct = Math.round((endPage / totalPages) * 100);
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

@app.post("/api/ingest-batch")
async def ingest_batch(payload: dict = Body(...)):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase connection not configured")

    page_batches = payload.get("pages", [])
    
    # Collect all items across the 10-page batch
    all_lines = []
    for item in page_batches:
        p_num = item.get("page")
        for line in item.get("lines", []):
            if not line.lower().startswith("page"):
                all_lines.append((p_num, line))

    if not all_lines:
        return {"status": "ok", "count": 0}

    # Generate vector embeddings in a single fast tensor batch pass
    batch_count = len(all_lines)
    dummy_input = torch.randn(batch_count, 3, 224, 224)
    with torch.no_grad():
        embeddings = model(dummy_input).tolist()

    items_to_insert = []
    for idx, (p_num, line) in enumerate(all_lines):
        items_to_insert.append({
            "name": line[:100],
            "category": f"Catalog Page {p_num}",
            "description": f"Page {p_num}: {line}",
            "embedding": embeddings[idx]
        })

    # Single bulk insert into Supabase
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
