import base64
import json
import logging
import os
import re

import requests

logger = logging.getLogger("CebolaAPI")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _ai_settings() -> dict:
    return {
        "google_vision_api_key": _env("GOOGLE_VISION_API_KEY"),
        "minicpm5_api_base": _env("MINICPM5_API_BASE", "http://127.0.0.1:8000/v1"),
        "minicpm5_api_key": _env("MINICPM5_API_KEY", "not-needed"),
        "minicpm5_model": _env("MINICPM5_MODEL", "MiniCPM5"),
    }


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
    settings = _ai_settings()
    if settings["google_vision_api_key"]:
        return _extract_text_via_rest_api(image_bytes, settings["google_vision_api_key"])

    credentials_path = _env("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path and os.path.isfile(credentials_path):
        return _extract_text_via_client_library(image_bytes)

    raise AiServiceError(
        "Google Vision is not configured. Set GOOGLE_VISION_API_KEY or GOOGLE_APPLICATION_CREDENTIALS."
    )


def _extract_text_via_rest_api(image_bytes: bytes, api_key: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
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
    except requests.RequestException as exc:
        logger.error("Google Vision REST request failed: %s", exc)
        raise AiServiceError(f"Google Vision OCR request failed: {exc}") from exc

    data = response.json()
    if response.status_code >= 400:
        message = data.get("error", {}).get("message", response.text)
        logger.error("Google Vision API error: %s", message)
        raise AiServiceError(f"Google Vision OCR failed: {message}")

    responses = data.get("responses") or []
    if not responses:
        return ""

    vision_error = responses[0].get("error", {}).get("message")
    if vision_error:
        raise AiServiceError(f"Google Vision OCR failed: {vision_error}")

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
    settings = _ai_settings()
    prompt = (
        "You help Portuguese marketplace merchants create product listings.\n"
        "Use the text extracted from a product photo to write a title and description.\n\n"
        f"Extracted text:\n{ocr_text or '(no readable text detected)'}\n\n"
        "Respond with valid JSON only, without markdown:\n"
        '{"title": "...", "description": "..."}\n\n'
        "Write in Portuguese. Keep the title under 80 characters. "
        "Write a description of 2-4 sentences suitable for e-commerce."
    )

    base_url = settings["minicpm5_api_base"].rstrip("/")
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings['minicpm5_api_key']}",
        "Content-Type": "application/json",
    }
    body = {
        "model": settings["minicpm5_model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
        "max_tokens": 512,
    }

    try:
        response = requests.post(url, headers=headers, json=body, timeout=120)
    except requests.RequestException as exc:
        logger.error("MiniCPM5 request failed: %s", exc)
        raise AiServiceError(
            f"MiniCPM5 text generation request failed at {url}: {exc}"
        ) from exc

    payload = response.json()
    if response.status_code >= 400:
        message = payload.get("error", {}).get("message", response.text)
        logger.error("MiniCPM5 API error: %s", message)
        raise AiServiceError(f"MiniCPM5 text generation failed: {message}")
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
