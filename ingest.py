import os
import io
import torch
import pypdfium2 as pdfium
from PIL import Image
from transformers import AutoProcessor, AutoModel
from supabase import create_client, Client

# 1. Environment & Client Setup
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY environment variables.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# 2. Configuration Settings
BUCKET_NAME = "catalog-pages"
PDF_PATH = "catalog.pdf"  # Ensure catalog.pdf is uploaded to your GitHub root folder

# 3. Load Multimodal Vector Model (CLIP)
print("Loading CLIP vision model...")
model_id = "openai/clip-vit-base-patch32"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)
model.eval()

def get_image_embedding(image_bytes):
    """Generates a 512-dimension vector from image bytes using CLIP."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    # Normalize vector to unit length for cosine similarity search
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    return image_features[0].tolist()

def process_pdf_catalog(pdf_file_path):
    """Processes each PDF page: converts to image, uploads to storage, and saves vector embedding to DB."""
    if not os.path.exists(pdf_file_path):
        print(f"Error: {pdf_file_path} not found in root directory!")
        return

    print(f"Opening {pdf_file_path}...")
    pdf = pdfium.PdfDocument(pdf_file_path)
    num_pages = len(pdf)
    print(f"Found {num_pages} pages in PDF.")

    for page_index in range(num_pages):
        page_num = page_index + 1
        print(f"Processing Page {page_num}/{num_pages}...")

        # Render PDF page to high-res image
        page = pdf[page_index]
        pil_image = page.render(scale=2).to_pil()

        # Convert PIL image to JPEG bytes
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG', quality=85)
        img_bytes = img_byte_arr.getvalue()

        # Generate vector embedding for this catalog page
        embedding = get_image_embedding(img_bytes)

        # File path inside Supabase Storage Bucket
        file_path = f"page_{page_num}.jpg"

        # Upload image to Supabase Storage
        try:
            supabase.storage.from_(BUCKET_NAME).upload(
                path=file_path,
                file=img_bytes
            )
        except Exception as e:
            print(f"Notice during upload for page {page_num}: {e}")

        # Construct public URL for the uploaded page
        image_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{file_path}"

        # Save metadata and vector embedding to Supabase Database
        data = {
            "page_number": page_num,
            "image_path": file_path,
            "image_url": image_url,
            "embedding": embedding
        }

        try:
            supabase.table("product_images").insert(data).execute()
            print(f"Successfully ingested Page {page_num}")
        except Exception as e:
            print(f"Error inserting Page {page_num} into database: {e}")

if __name__ == "__main__":
    process_pdf_catalog(PDF_PATH)
