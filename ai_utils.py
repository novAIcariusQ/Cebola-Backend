import base64
import json
import logging
import os
import re
from typing import Dict, List, Tuple, TypedDict

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

GENERIC_LABELS = {
    "font",
    "graphic design",
    "graphics",
    "logo",
    "screenshot",
    "parallel",
    "electric blue",
    "azure",
    "text",
    "number",
    "symbol",
    "design",
    "art",
}

LABEL_PT = {
    "food": "alimentos",
    "fruit": "fruta",
    "vegetable": "legumes",
    "bread": "pão",
    "bottle": "garrafa",
    "drink": "bebida",
    "clothing": "vestuário",
    "shoe": "calçado",
    "building": "edifício",
    "skyscraper": "arranha-céus",
    "house": "casa",
    "snow": "neve",
    "winter": "inverno",
    "plant": "planta",
    "flower": "flor",
    "furniture": "mobiliário",
    "toy": "brinquedo",
    "electronics": "eletrónica",
    "computer": "computador",
    "mobile phone": "telemóvel",
    "book": "livro",
    "jewelry": "joalharia",
    "cosmetics": "cosméticos",
    "packaged goods": "produto embalado",
}


class ScoredLabel(TypedDict):
    description: str
    score: float


class ImageAnalysis(TypedDict):
    ocr_text: str
    labels: List[ScoredLabel]
    logos: List[ScoredLabel]


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


def _parse_scored_items(items: List[dict]) -> List[ScoredLabel]:
    parsed: List[ScoredLabel] = []
    for item in items or []:
        description = str(item.get("description", "")).strip()
        if not description:
            continue
        parsed.append(
            {
                "description": description,
                "score": float(item.get("score", 0) or 0),
            }
        )
    return parsed


def _empty_analysis() -> ImageAnalysis:
    return {"ocr_text": "", "labels": [], "logos": []}


def analyze_image_with_google_vision(image_bytes: bytes) -> ImageAnalysis:
    settings = _ai_settings()
    if settings["google_vision_api_key"]:
        return _analyze_via_rest_api(image_bytes, settings["google_vision_api_key"])

    credentials_path = _env("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path and os.path.isfile(credentials_path):
        return _analyze_via_client_library(image_bytes)

    raise AiServiceError(
        "Google Vision is not configured. Set GOOGLE_VISION_API_KEY or GOOGLE_APPLICATION_CREDENTIALS."
    )


def extract_text_with_google_vision(image_bytes: bytes) -> str:
    return analyze_image_with_google_vision(image_bytes)["ocr_text"]


def _analyze_via_rest_api(image_bytes: bytes, api_key: str) -> ImageAnalysis:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    payload = {
        "requests": [
            {
                "image": {"content": encoded},
                "features": [
                    {"type": "TEXT_DETECTION", "maxResults": 1},
                    {"type": "LABEL_DETECTION", "maxResults": 12},
                    {"type": "LOGO_DETECTION", "maxResults": 5},
                ],
            }
        ]
    }

    try:
        response = requests.post(url, json=payload, timeout=45)
    except requests.RequestException as exc:
        logger.error("Google Vision REST request failed: %s", exc)
        raise AiServiceError(f"Google Vision request failed: {exc}") from exc

    data = response.json()
    if response.status_code >= 400:
        message = data.get("error", {}).get("message", response.text)
        logger.error("Google Vision API error: %s", message)
        raise AiServiceError(f"Google Vision failed: {message}")

    responses = data.get("responses") or []
    if not responses:
        return _empty_analysis()

    vision_error = responses[0].get("error", {}).get("message")
    if vision_error:
        raise AiServiceError(f"Google Vision failed: {vision_error}")

    result = responses[0]
    annotations = result.get("textAnnotations") or []
    ocr_text = (annotations[0].get("description") or "").strip() if annotations else ""

    return {
        "ocr_text": ocr_text,
        "labels": _parse_scored_items(result.get("labelAnnotations")),
        "logos": _parse_scored_items(result.get("logoAnnotations")),
    }


def _analyze_via_client_library(image_bytes: bytes) -> ImageAnalysis:
    try:
        from google.cloud import vision
    except ImportError as exc:
        raise AiServiceError(
            "google-cloud-vision is required when using GOOGLE_APPLICATION_CREDENTIALS"
        ) from exc

    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    result = client.annotate_image(
        {
            "image": image,
            "features": [
                {"type_": vision.Feature.Type.TEXT_DETECTION},
                {"type_": vision.Feature.Type.LABEL_DETECTION, "max_results": 12},
                {"type_": vision.Feature.Type.LOGO_DETECTION, "max_results": 5},
            ],
        }
    )

    if result.error.message:
        raise AiServiceError(f"Google Vision failed: {result.error.message}")

    annotations = result.text_annotations
    ocr_text = (annotations[0].description or "").strip() if annotations else ""

    labels = [
        {"description": label.description, "score": float(label.score)}
        for label in result.label_annotations
        if label.description
    ]
    logos = [
        {"description": logo.description, "score": float(logo.score)}
        for logo in result.logo_annotations
        if logo.description
    ]

    return {"ocr_text": ocr_text, "labels": labels, "logos": logos}


def _normalize_ocr_lines(ocr_text: str) -> List[str]:
    lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]
    if lines:
        return lines

    compact = re.sub(r"\s+", " ", ocr_text).strip()
    return [compact] if compact else []


def _meaningful_labels(labels: List[ScoredLabel], min_score: float = 0.62) -> List[str]:
    meaningful: List[str] = []
    for item in labels:
        description = item["description"].strip()
        if item["score"] < min_score:
            continue
        if description.lower() in GENERIC_LABELS:
            continue
        if description not in meaningful:
            meaningful.append(description)
    return meaningful[:6]


def _label_to_portuguese(label: str) -> str:
    return LABEL_PT.get(label.lower(), label)


def _truncate(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _build_vision_context(analysis: ImageAnalysis) -> str:
    sections: List[str] = []

    ocr_text = analysis["ocr_text"].strip()
    sections.append(
        f"Text visible in the image:\n{ocr_text}" if ocr_text else "Text visible in the image:\n(none)"
    )

    logos = [item["description"] for item in analysis["logos"][:5]]
    if logos:
        sections.append("Detected brands/logos: " + ", ".join(logos))

    labels = _meaningful_labels(analysis["labels"])
    if labels:
        sections.append("Visual content detected: " + ", ".join(labels))

    if len(sections) == 1 and not ocr_text:
        sections.append("Visual content detected: (no strong labels)")

    return "\n\n".join(sections)


def _description_from_labels(labels: List[str]) -> str:
    translated = [_label_to_portuguese(label) for label in labels[:4]]
    if len(translated) == 1:
        subject = translated[0]
        return (
            f"Produto fotografado com aspeto visual associado a {subject.lower()}. "
            "Revise o título e acrescente detalhes como quantidade, origem ou modo de utilização."
        )

    joined = ", ".join(translated[:-1]) + f" e {translated[-1]}"
    return (
        f"Artigo apresentado na imagem com elementos visuais como {joined.lower()}. "
        "Ajuste a descrição com preço, composição e outras informações relevantes para o cliente."
    )


def _generate_smart_fallback(analysis: ImageAnalysis) -> dict:
    lines = _normalize_ocr_lines(analysis["ocr_text"])
    logos = [item["description"] for item in analysis["logos"][:3]]
    labels = _meaningful_labels(analysis["labels"])

    if lines:
        title = _truncate(lines[0], 80).strip(" ,.;:-")
        body = " ".join(lines[1:]) if len(lines) > 1 else ""
        body = _truncate(body, 500)

        if body and len(body) >= 40:
            return {"title": title, "description": body}

        if logos:
            description = (
                f"{title}. Produto ou marca associada a {logos[0]}. "
                "Complete a descrição com detalhes do artigo, composição e utilização."
            )
            return {"title": title, "description": _truncate(description, 500)}

        if labels:
            description = f"{title}. {_description_from_labels(labels)}"
            return {"title": title, "description": _truncate(description, 500)}

        return {
            "title": title,
            "description": (
                f"{title}. Texto identificado na imagem; complete a descrição com detalhes do produto."
            ),
        }

    if logos:
        title = _truncate(logos[0], 80)
        description = (
            f"Produto relacionado com a marca {logos[0]}. "
            f"{_description_from_labels(labels) if labels else 'Adicione detalhes sobre o artigo, composição e utilização.'}"
        )
        return {"title": title, "description": _truncate(description, 500)}

    if labels:
        primary = _label_to_portuguese(labels[0])
        title = _truncate(primary.capitalize(), 80)
        return {"title": title, "description": _description_from_labels(labels)}

    return {
        "title": "Novo produto",
        "description": (
            "Não foi possível identificar texto ou elementos claros na imagem. "
            "Tente uma foto mais próxima, com boa luz e o produto em destaque."
        ),
    }


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


def _request_minicpm_copy(base_url: str, model: str, vision_context: str, api_key: str) -> dict:
    prompt = (
        "You help Portuguese marketplace merchants create product listings.\n"
        "Use the image analysis below (text, brands/logos, and visual labels) to write a title and description.\n"
        "If there is little or no text, infer a sensible product listing from logos and visual content.\n"
        "Do not invent specific prices, weights, or certifications that are not supported by the analysis.\n"
        "Avoid generic filler; be specific to what was detected.\n\n"
        f"{vision_context}\n\n"
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


def generate_product_copy(analysis: ImageAnalysis) -> dict:
    settings = _ai_settings()
    api_key = settings["minicpm5_api_key"]
    vision_context = _build_vision_context(analysis)
    errors: List[str] = []

    for base_url, model in _llm_candidates():
        try:
            result = _request_minicpm_copy(base_url, model, vision_context, api_key)
            logger.info("Generated product copy via %s model=%s", base_url, model)
            return result
        except (AiServiceError, requests.RequestException) as exc:
            message = str(exc)
            errors.append(message)
            logger.warning("LLM candidate failed (%s, %s): %s", base_url, model, message)

    logger.warning(
        "MiniCPM5 unavailable, using vision-aware fallback. Attempts: %s",
        "; ".join(errors[:3]),
    )
    return _generate_smart_fallback(analysis)


def describe_product_from_image(image_bytes: bytes) -> dict:
    analysis = analyze_image_with_google_vision(image_bytes)
    logger.info(
        "Google Vision analysis: %s chars of text, %s labels, %s logos",
        len(analysis["ocr_text"]),
        len(analysis["labels"]),
        len(analysis["logos"]),
    )
    return generate_product_copy(analysis)
