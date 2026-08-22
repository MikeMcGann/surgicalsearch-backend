import io
import os
import pypdfium2 as pdfium
from PIL import Image
import torch
import torchvision.models as models
from supabase import create_client, Client

# Read configuration from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
PDF_PATH = "catalog.pdf"
BUCKET_NAME = "catalog-images"

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY environment variables.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Initialize MobileNet model
weights = models.MobileNet_V3_Small_Weights.DEFAULT
base_model = models.mobilenet_v3_small(weights=weights)
base_model.eval()
preprocess = weights.transforms()

torch.manual_seed(42)
projection = torch.nn.Linear(576, 512)
projection.eval()

def generate_embedding(pil_img: Image.Image) -> list[float]:
    processed_img = preprocess(pil_img).unsqueeze(0)
    with torch.no_grad():
        features = base_model.features(processed_img)
        features = base_model.avgpool(features)
        features = torch.flatten(features, 1)
        features_512 = projection(features)
        features_512 /= features_512.norm(dim=-1, keepdim=True)
    return features_512.cpu().numpy().flatten().tolist()

def process_pdf_catalog(pdf_file_path: str):
    if not os.path.exists(pdf_file_path):
        print(f"Error: Could not find {pdf_file_path}. Please upload a PDF named catalog.pdf.")
        return

    pdf = pdfium.PdfDocument(pdf_file_path)
    print(f"Loaded catalog with {len(pdf)} pages.")

    for i, page in enumerate(pdf):
        page_num = i + 1
        print(f"Processing Page {page_num}...")

        pil_image = page.render(scale=200/72).to_pil().convert("RGB")
        
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG', quality=85)
        img_bytes = img_byte_arr.getvalue()

        file_path = f"page_{page_num}.jpg"
        supabase.storage.from_(BUCKET_NAME).upload(
            path=file_path,
            file=img_bytes,
            file_options={"content-type": "image/jpeg", "upsert": True}
        )

        public_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{file_path}"
        embedding_vector = generate_embedding(pil_image)

        m_id = "m_catalog"
        p_id = f"prod_page_{page_num}"
        sku = f"CAT-PG-{page_num}"
        title = f"Catalog Page {page_num} Instruments"

        supabase.table("manufacturers").upsert({"manufacturer_id": m_id, "name": "Catalog Manufacturer"}).execute()
        supabase.table("products").upsert({
            "product_id": p_id,
            "manufacturer_id": m_id,
            "sku": sku,
            "pattern_name": "Surgical Instruments",
            "title": title
        }).execute()

        supabase.table("product_images").upsert({
            "image_id": f"img_page_{page_num}",
            "product_id": p_id,
            "image_url": public_url,
            "embedding": embedding_vector
        }).execute()

        print(f"Successfully ingested Page {page_num}")

if __name__ == "__main__":
    process_pdf_catalog(PDF_PATH)
