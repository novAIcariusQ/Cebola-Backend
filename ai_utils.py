import base64
import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger("CebolaAPI")

DEFAULT_LLM_ENDPOINTS = (
    "http://127.0.0.1:8000/v1",
    "http://127.0.0.1:11434/v1",
    "http://127.0.0.1:8080/v1",
)
DEFAULT_LLM_MODELS = (
    "MiniCPM5",
    "minicpm5",
    "openbmb/MiniCPM5-1B-Instruct",
    "minicpm",
)


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _ai_settings() -> dict:
    return {
        "google_vision_api_key": _env("GOOGLE_VISION_API_KEY"),
        "minicpm5_api_base": _env("MINICPM5_API_BASE"),
        "minicpm5_api_key": _env("MINICPM5_API_KEY", "not-needed"),
        "minicpm5_model": _env("MINICPM5_MODEL"),
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


def _normalize_ocr_lines(ocr_text: str) -> List[str]:
    lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]
    if lines:
        return lines

    compact = re.sub(r"\s+", " ", ocr_text).strip()
    return [compact] if compact else []


def _generate_from_ocr_fallback(ocr_text: str) -> dict:
    lines = _normalize_ocr_lines(ocr_text)
    if not lines:
        return {
            "title": "Produto Artesanal Regional",
            "description": (
                "Produto selecionado de origem local, ideal para quem valoriza qualidade e frescura. "
                "Perfeito para o dia a dia ou para oferecer em ocasiões especiais."
            ),
        }

    title_source = lines[0]
    title = title_source[:80].strip(" ,.;:-")
    body_lines = lines[1:] or lines
    description = " ".join(body_lines)
    description = re.sub(r"\s+", " ", description).strip()

    if len(description) < 40:
        description = (
            f"{title_source}. Produto de qualidade, selecionado para o mercado Cebola. "
            "Ideal para consumo fresco e para quem procura sabores autênticos da região."
        )

    if len(description) > 500:
        description = description[:497].rstrip() + "..."

    return {"title": title, "description": description}


def _llm_candidates() -> List[Tuple[str, str]]:
    settings = _ai_settings()
    candidates: List[Tuple[str, str]] = []

    if settings["minicpm5_api_base"]:
        model = settings["minicpm5_model"] or DEFAULT_LLM_MODELS[0]
        candidates.append((settings["minicpm5_api_base"], model))

    for base in DEFAULT_LLM_ENDPOINTS:
        for model in DEFAULT_LLM_MODELS:
            pair = (base, model)
            if pair not in candidates:
                candidates.append(pair)

    return candidates


def _request_minicpm_copy(base_url: str, model: str, ocr_text: str, api_key: str) -> dict:
    prompt = (
        "You help Portuguese marketplace merchants create product listings.\n"
        "Use the text extracted from a product photo to write a title and description.\n\n"
        f"Extracted text:\n{ocr_text or '(no readable text detected)'}\n\n"
        "Respond with valid JSON only, without markdown:\n"
        '{"title": "...", "description": "..."}\n\n'
        "Write in Portuguese. Keep the title under 80 characters. "
        "Write a description of 2-4 sentences suitable for e-commerce."
    )

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
        "max_tokens": 512,
    }

    response = requests.post(url, headers=headers, json=body, timeout=90)
    payload: Dict = {}
    try:
        payload = response.json()
    except ValueError:
        payload = {}

    if response.status_code >= 400:
        message = payload.get("error", {}).get("message", response.text)
        raise AiServiceError(f"MiniCPM5 text generation failed at {url}: {message}")

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


def generate_product_copy(ocr_text: str) -> dict:
    settings = _ai_settings()
    api_key = settings["minicpm5_api_key"]
    errors: List[str] = []

    for base_url, model in _llm_candidates():
        try:
            result = _request_minicpm_copy(base_url, model, ocr_text, api_key)
            logger.info("Generated product copy via %s model=%s", base_url, model)
            return result
        except (AiServiceError, requests.RequestException) as exc:
            message = str(exc)
            errors.append(message)
            logger.warning("LLM candidate failed (%s, %s): %s", base_url, model, message)

    logger.warning(
        "MiniCPM5 unavailable, using OCR fallback. Attempts: %s",
        "; ".join(errors[:3]),
    )
    return _generate_from_ocr_fallback(ocr_text)


def describe_product_from_image(image_bytes: bytes) -> dict:
    ocr_text = extract_text_with_google_vision(image_bytes)
    logger.info("Google Vision extracted %s characters of text", len(ocr_text))
    return generate_product_copy(ocr_text)
