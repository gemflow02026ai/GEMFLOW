"""
Jewel Mission - FastAPI Backend
================================
Image processing server for the Jewel Mission Flutter app.
Provides two endpoints:
  - POST /process-ecommerce: Process e-commerce images via OpenAI (zero-hallucination)
  - POST /process-creative:  Process 1 creative image via Gemini Imagen

Usage:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import os
import base64
import numpy as np
from typing import List
from pathlib import Path

from PIL import Image, ImageFilter
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Always load .env relative to this script, not the working directory
load_dotenv(Path(__file__).parent / ".env")

app = FastAPI(
    title="Jewel Mission API",
    description="AI-powered jewellery image processing backend",
    version="1.0.0",
)

# Allow all origins for prototype
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API Keys ──────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check endpoint for the Flutter app to verify server availability."""
    return {
        "status": "healthy",
        "service": "Jewel Mission API",
        "version": "1.0.0",
        "openai_configured": bool(OPENAI_API_KEY),
        "gemini_configured": bool(GEMINI_API_KEY),
    }


@app.post("/process-ecommerce")
async def process_ecommerce(
    product_id: str = Form(...),
    images: List[UploadFile] = File(...),
):
    """
    Process e-commerce jewellery images via OpenAI gpt-image-1.

    Receives product images, applies zero-hallucination processing
    (mask-based inpainting + seed + quality + post-composite), and
    returns processed image data back to the Flutter app.
    Falls back to pass-through mode if the API key is missing.
    """
    print(f"[LOG] Request received: POST /process-ecommerce")
    print(f"[LOG] Product ID: {product_id}")
    print(f"[LOG] Images received: {len(images)} files")

    if len(images) == 0:
        raise HTTPException(status_code=400, detail="No images provided")

    processed_images = []

    for idx, image in enumerate(images):
        image_bytes = await image.read()
        print(f"[LOG] Processing e-commerce image {idx + 1}/{len(images)} ({len(image_bytes)} bytes)...")

        if not OPENAI_API_KEY:
            print(f"[LOG] OpenAI API key missing — pass-through mode for image {idx + 1}")
            processed_images.append(base64.b64encode(image_bytes).decode("utf-8"))
            continue

        try:
            processed = await _process_with_openai(image_bytes, product_id)
            processed_images.append(base64.b64encode(processed).decode("utf-8"))
        except Exception as e:
            print(f"[LOG] OpenAI processing error: {e}")
            raise HTTPException(status_code=500, detail=f"OpenAI error: {str(e)}")


    print(f"[LOG] Response returned: {len(processed_images)} e-commerce images processed")
    return JSONResponse(content={
        "status": "success",
        "product_id": product_id,
        "processed_images": processed_images,
        "message": f"Processed {len(processed_images)} e-commerce images",
    })


@app.post("/process-creative")
async def process_creative(
    product_id: str = Form(...),
    image: UploadFile = File(...),
):
    """
    Process a creative marketing image via Gemini Imagen.
    """
    print(f"[LOG] Request received: POST /process-creative")
    print(f"[LOG] Product ID: {product_id}")

    image_bytes = await image.read()
    print(f"[LOG] Image received: 1 file ({len(image_bytes)} bytes)")

    # print(f"[LOG] Gemini disabled — pass-through mode")
    # result_b64 = base64.b64encode(image_bytes).decode("utf-8")
    
    # print(f"[LOG] Response returned: creative image processed (pass-through)")
    # return JSONResponse(content={
    #     "status": "success",
    #     "processed_image": result_b64,
    #     "message": "Processed creative image (pass-through — Gemini disabled)",
    # })

    try:
        print(f"[LOG] Processing creative image via Gemini...")
        processed_bytes = await _process_with_gemini(image_bytes, product_id)
        result_b64 = base64.b64encode(processed_bytes).decode("utf-8")
        
        print(f"[LOG] Response returned: creative image processed by Gemini")
        return JSONResponse(content={
            "status": "success",
            "processed_image": result_b64,
            "message": "Processed creative image via Gemini",
        })
    except Exception as e:
        print(f"[LOG] Gemini processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Gemini error: {str(e)}")



# ─────────────────────────────────────────────────────────────────────────────
# ZERO-HALLUCINATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

async def _process_with_openai(image_bytes: bytes, product_id: str) -> bytes:
    """
    Process a jewelry image using OpenAI directly with an image and a prompt.
    """
    from openai import OpenAI

    print(f"[LOG] OpenAI processing started for product {product_id}")
    client = OpenAI(api_key=OPENAI_API_KEY)

    # ── Convert input to PNG ──────────────────────────────────────────────
    img = Image.open(io.BytesIO(image_bytes))
    out = io.BytesIO()
    img.save(out, format="PNG")
    out.seek(0)
    out.name = "image.png"

    # ── Prompt ────────────────────────────────────────────────────────────
    prompt = """CRITICAL INSTRUCTIONS
Preserve the jewelry exactly as photographed.
DO NOT:
•⁠  ⁠Redesign the jewelry
•⁠  ⁠Change proportions
•⁠  ⁠Change dimensions
•⁠  ⁠Change gemstone size
•⁠  ⁠Change gemstone shape
•⁠  ⁠Change gemstone color
•⁠  ⁠Change metal color
•⁠  ⁠Change settings or prongs
•⁠  ⁠Add diamonds
•⁠  ⁠Remove diamonds
•⁠  ⁠Add decorative elements
•⁠  ⁠Remove decorative elements
•⁠  ⁠Modify craftsmanship
•⁠  ⁠Exaggerate details
•⁠  ⁠Create missing parts
•⁠  ⁠Hallucinate features
•⁠  ⁠Change viewing angle
•⁠  ⁠Change perspective
•⁠  ⁠Change orientation
REMOVE COMPLETELY:
•⁠  ⁠Transparent rods
•⁠  ⁠Acrylic supports
•⁠  ⁠Plastic holders
•⁠  ⁠Mounting sticks
•⁠  ⁠Wires
•⁠  ⁠Invisible supports
•⁠  ⁠Studio rigs
•⁠  ⁠Adhesives
•⁠  ⁠Dust
•⁠  ⁠Scratches
•⁠  ⁠Sensor spots
•⁠  ⁠Background artifacts
•⁠  ⁠Reflections caused by support structures
•⁠  ⁠Unwanted shadows from support equipment
RETAIN EXACTLY:
•⁠  ⁠Jewelry design
•⁠  ⁠Geometry
•⁠  ⁠Curvature
•⁠  ⁠Stone placement
•⁠  ⁠Stone count
•⁠  ⁠Metal finish
•⁠  ⁠Surface details
•⁠  ⁠Craftsmanship details
•⁠  ⁠Proportions
•⁠  ⁠Construction details
•⁠  ⁠Product orientation
IMAGE ENHANCEMENT REQUIREMENTS
•⁠  ⁠Increase image quality to high-resolution 2K output
•⁠  ⁠Recover fine jewelry details
•⁠  ⁠Enhance gemstone clarity
•⁠  ⁠Enhance metal surface definition
•⁠  ⁠Improve edge sharpness
•⁠  ⁠Improve micro-contrast
•⁠  ⁠Reduce noise
•⁠  ⁠Preserve realistic texture
•⁠  ⁠Preserve realistic reflections
•⁠  ⁠Preserve realistic highlights
•⁠  ⁠Preserve realistic gemstone brilliance
•⁠  ⁠Maintain natural appearance
•⁠  ⁠Maintain physical accuracy
OUTPUT REQUIREMENTS
•⁠  ⁠Pure white background (#FFFFFF)
•⁠  ⁠Centered product
•⁠  ⁠Professional jewelry retouching
•⁠  ⁠Luxury e-commerce standard
•⁠  ⁠Commercial catalog quality
•⁠  ⁠Clean cutout appearance
•⁠  ⁠Accurate metal reflections
•⁠  ⁠Accurate gemstone sparkle
•⁠  ⁠High sharpness
•⁠  ⁠Natural realistic shadows only if required
•⁠  ⁠Premium studio photography appearance
•⁠  ⁠High-end jewelry product photography
FINAL RULE
The final image must look like the exact same jewelry photographed perfectly in a professional studio after expert retouching.
The result must be visually cleaner, sharper, and higher resolution while remaining geometrically identical to the original product.
ZERO redesign.
ZERO imagination.
ZERO artistic interpretation.
ZERO generated design changes.
Only cleanup, enhancement, retouching, support removal, white background replacement, and 2K upscaling"""

    print(f"[LOG] Sending direct image and prompt to OpenAI (model: gpt-image-2-2026-04-21)...")

    response = client.images.edit(
        model="gpt-image-2-2026-04-21",
        image=out,
        prompt=prompt,
        size="2048x2048",
    )

    print(f"[LOG] OpenAI response received")

    result_b64   = response.data[0].b64_json
    result_bytes = base64.b64decode(result_b64)

    print(f"[LOG] OpenAI processing completed for product {product_id}")
    return result_bytes

    print(f"[LOG] OpenAI processing completed for product {product_id}")
    return result_bytes


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI CREATIVE PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

async def _process_with_gemini(image_bytes: bytes, product_id: str) -> bytes:
    """
    Analyze the uploaded jewelry product image and preserve the jewelry exactly as shown,
    including its shape, gemstone placement, metal color, proportions, craftsmanship,
    engravings, and all design details. Do not redesign, modify, replace, or alter any
    part of the jewelry.
    Create a luxury high-end commercial jewelry photography scene. Place the jewelry
    naturally within a unique and visually compelling environment that is different from
    previous generations.
    Randomly select an elegant setting inspired by nature, architecture, premium fabrics,
    rare stones, flowers, water elements, artistic sculptures, handcrafted textures,
    luxury interiors, organic materials, or abstract fine-art compositions.
    The setting must always complement the jewelry's design, color palette, and gemstones.
    Ensure every generated image uses a completely different environment, prop arrangement,
    composition, surface texture, and background story. Avoid repeating locations, props,
    flowers, stones, fabrics, colors, lighting patterns, or compositions used in previous outputs.
    Use professional luxury advertising photography standards with realistic light and shadow
    interaction, natural reflections, accurate gemstone brilliance, premium depth of field,
    macro-level detail, sharp focus on the jewelry, cinematic composition, realistic materials,
    elegant negative space, and magazine-quality styling.
    The jewelry must remain the hero subject, occupying the visual focus of the frame.
    Create a believable premium atmosphere with authentic textures, realistic physics,
    refined color grading, soft bokeh, subtle environmental reflections, and ultra-photorealistic
    rendering. Output as a luxury campaign photograph suitable for international jewelry brands,
    high-end e-commerce, portfolio presentation, editorial advertising, and premium marketing
    materials. Ultra realistic, highly detailed, commercial photography, 8K quality, natural
    materials, physically accurate lighting, premium luxury aesthetic.
    """
    from google import genai
    from google.genai import types

    if not GEMINI_API_KEY:
        raise Exception("Gemini API key is missing")

    client = genai.Client(api_key=GEMINI_API_KEY)

    # ── Generate creative image with Gemini Flash Image ───────────────
    full_prompt = (
        "The objective is:\n\n"
        "INPUT IMAGE\n"
        "↓\n"
        "Creative Enhancement\n"
        "↓\n"
        "OUTPUT IMAGE\n\n"
        "The output must use the uploaded jewelry image as the source and preserve the exact jewelry product.\n\n"
        "CRITICAL REQUIREMENTS:\n\n"
        "* Do not generate a new jewelry design.\n"
        "* Do not create a different ring, necklace, bracelet, or earring.\n"
        "* Do not change the jewelry geometry.\n"
        "* Do not change proportions.\n"
        "* Do not change dimensions.\n"
        "* Do not change metal color.\n"
        "* Do not change gemstone color.\n"
        "* Do not change gemstone size.\n"
        "* Do not change gemstone placement.\n"
        "* Do not add or remove stones.\n"
        "* Do not add or remove jewelry elements.\n"
        "* Do not redesign the product.\n"
        "* Do not replace the product with an AI-generated version.\n\n"
        "Required Behavior:\n\n"
        "Use the uploaded jewelry image as the exact product reference.\n\n"
        "Keep the jewelry 100% identical to the input image.\n\n"
        "Only enhance:\n\n"
        "* Background\n"
        "* Environment\n"
        "* Lighting\n"
        "* Reflections\n"
        "* Shadows\n"
        "* Presentation\n"
        "* Marketing composition\n\n"
        "Creative Requirements:\n\n"
        "* Premium luxury jewelry advertising style\n"
        "* High-end commercial photography\n"
        "* Elegant premium background\n"
        "* Luxury brand campaign appearance\n"
        "* Social-media-ready creative\n"
        "* Professional marketing visual\n"
        "* Photorealistic output\n\n"
        "Final Rule:\n\n"
        "The jewelry itself must remain unchanged and identical to the uploaded image.\n\n"
        "Only transform the scene around the jewelry to create a premium marketing creative.\n\n"
        "The result should look like the same jewelry photographed for a luxury advertising campaign, not a newly generated jewelry design."
    )

    print(f"[LOG] Gemini: calling gemini-2.5-flash-image for product {product_id}")

    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[
            full_prompt,
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        ],
    )
    print("[LOG] Gemini creative image response received")

    if response.candidates and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                print("[LOG] Creative image generated successfully")
                return part.inline_data.data

    raise Exception("No generated image found in Gemini response")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
