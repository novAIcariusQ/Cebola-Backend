import base64
import json
import logging
import os
import re

import requests

logger = logging.getLogger("CebolaAPI")

MINICPM5_API_BASE = os.getenv("MINICPM5_API_BASE", "http://127.0.0.1:8000/v1")
MINICPM5_API_KEY = os.getenv("MINICPM5_API_KEY", "not-needed")
MINICPM5_MODEL = os.getenv("MINICPM5_MODEL", "MiniCPM5")
GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY", "")


class AiServiceError(Exception):
    pass


def _parse_json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.DOTALL)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise AiServiceError("MiniCPM5 returned an invalid JSON response")


def extract_text_with_google_vision(image_bytes: bytes) -> str:
    if GOOGLE_VISION_API_KEY:
        return _extract_text_via_rest_api(image_bytes)

    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    if credentials_path and os.path.isfile(credentials_path):
        return _extract_text_via_client_library(image_bytes)

    raise AiServiceError(
        "Google Vision is not configured. Set GOOGLE_VISION_API_KEY or GOOGLE_APPLICATION_CREDENTIALS."
    )


def _extract_text_via_rest_api(image_bytes: bytes) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"
    payload = {
        "requests": [
            {
                "image": {"content": encoded},
                "features": [{"type": "TEXT_DETECTION", "maxResults": 1}],
            }
        ]
    }

    try:
        response = requests.post(url, json=payload, timeout=45)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Google Vision REST request failed: %s", exc)
        raise AiServiceError("Google Vision OCR request failed") from exc

    data = response.json()
    responses = data.get("responses") or []
    if not responses:
        return ""

    annotations = responses[0].get("textAnnotations") or []
    if not annotations:
        return ""

    return (annotations[0].get("description") or "").strip()


def _extract_text_via_client_library(image_bytes: bytes) -> str:
    try:
        from google.cloud import vision
    except ImportError as exc:
        raise AiServiceError(
            "google-cloud-vision is required when using GOOGLE_APPLICATION_CREDENTIALS"
        ) from exc

    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    result = client.text_detection(image=image)

    if result.error.message:
        raise AiServiceError(f"Google Vision OCR failed: {result.error.message}")

    annotations = result.text_annotations
    if not annotations:
        return ""

    return (annotations[0].description or "").strip()


def generate_product_copy(ocr_text: str) -> dict:
    prompt = (
        "You help Portuguese marketplace merchants create product listings.\n"
        "Use the text extracted from a product photo to write a title and description.\n\n"
        f"Extracted text:\n{ocr_text or '(no readable text detected)'}\n\n"
        "Respond with valid JSON only, without markdown:\n"
        '{"title": "...", "description": "..."}\n\n'
        "Write in Portuguese. Keep the title under 80 characters. "
        "Write a description of 2-4 sentences suitable for e-commerce."
    )

    url = f"{MINICPM5_API_BASE.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {MINICPM5_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": MINICPM5_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
        "max_tokens": 512,
    }

    try:
        response = requests.post(url, headers=headers, json=body, timeout=120)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.error("MiniCPM5 request failed: %s", exc)
        raise AiServiceError("MiniCPM5 text generation request failed") from exc

    payload = response.json()
    choices = payload.get("choices") or []
    if not choices:
        raise AiServiceError("MiniCPM5 returned an empty response")

    content = choices[0].get("message", {}).get("content", "")
    if not content:
        raise AiServiceError("MiniCPM5 returned an empty message")

    parsed = _parse_json_object(content)
    title = str(parsed.get("title", "")).strip()
    description = str(parsed.get("description", "")).strip()

    if not title or not description:
        raise AiServiceError("MiniCPM5 response is missing title or description")

    return {"title": title, "description": description}


def describe_product_from_image(image_bytes: bytes) -> dict:
    ocr_text = extract_text_with_google_vision(image_bytes)
    logger.info("Google Vision extracted %s characters of text", len(ocr_text))
    return generate_product_copy(ocr_text)
