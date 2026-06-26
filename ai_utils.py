import base64
import json
import logging
import os
import random
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
    "close-up",
    "close up",
    "clip art",
    "game",
}

VAGUE_LABELS = {
    "toe",
    "foot",
    "nail",
    "finger",
    "hand",
    "skin",
    "human body",
    "body part",
    "joint",
    "thumb",
}

MARKETING_RULES = (
    "Write like a professional Portuguese e-commerce listing for the Cebola marketplace.\n"
    "- Title: catchy, commercial, under 80 characters; never raw image tags in English\n"
    "- Description: 2-4 persuasive sentences; vary tone and structure each time\n"
    "- Highlight quality, presentation, and why a customer would want this item\n"
    "- Use different openings, benefits, and CTAs; avoid repeating the same template\n"
    "- Never list technical vision labels (e.g. toe, foot, close-up, clip art) in the output\n"
    "- Do not invent prices, weights, certifications, or stock levels\n"
    "- Use Portuguese (Portugal), natural and sales-ready"
)

TITLE_SUFFIXES = (
    "selecionado",
    "premium",
    "exclusivo",
    "em destaque",
    "original",
    "da loja",
    "especial",
)

TITLE_JOINERS = (" — ", " | ", ": ", " · ")

DESCRIPTION_INTROS_BRAND = (
    "Apresentamos {brand}, uma peça pensada para quem gosta de comprar com confiança.",
    "{brand} chega à Cebola com uma imagem forte e apresentação cuidada.",
    "Da marca {brand}, este artigo destaca-se logo à primeira vista.",
)

DESCRIPTION_INTROS_LABELS = (
    "Um artigo com inspiração em {highlights}, feito para chamar atenção na vitrine.",
    "Combina {highlights} num produto com aspeto cuidado e apelo comercial.",
    "Inspirado em {highlights}, ideal para quem procura algo diferente no catálogo.",
    "Com referências visuais a {highlights}, este item traz personalidade à loja.",
)

DESCRIPTION_INTROS_GENERIC = (
    "Peça com personalidade própria, pronta para entrar no seu catálogo.",
    "Artigo visualmente marcante, pensado para gerar curiosidade e cliques.",
    "Produto com boa presença na fotografia e aspeto comercial imediato.",
)

DESCRIPTION_MIDDLES = (
    "Funciona bem para ofertas sazonais, novidades ou destaques da semana.",
    "Aposta segura para clientes que valorizam originalidade e boa apresentação.",
    "Transmite qualidade visual e facilita a decisão de compra online.",
    "Dá um toque distinto à loja sem complicar a experiência do cliente.",
    "Combina bem com campanhas promocionais e vitrines temáticas.",
)

DESCRIPTION_CTAS = (
    "Veja quantidade, preço e envio antes de publicar.",
    "Ajuste stock e condições de entrega conforme o seu negócio.",
    "Confirme detalhes finais e coloque já à venda.",
    "Revise preço e disponibilidade antes de finalizar o anúncio.",
    "Complete os detalhes comerciais e publique quando estiver pronto.",
)

WRITING_STYLES = (
    "Use a warm, friendly boutique tone.",
    "Use a concise, premium brand tone.",
    "Use an energetic, promotional tone.",
    "Use a descriptive artisan-market tone.",
    "Use a playful, catchy tone suitable for impulse buys.",
    "Use a clear, practical tone focused on usefulness.",
)

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
    "person": "pessoa",
    "human": "pessoa",
    "room": "divisão",
    "bathroom": "casa de banho",
    "shower": "chuveiro",
    "plastic": "plástico",
    "container": "recipiente",
    "barrel": "barril",
    "bucket": "balde",
    "blue": "azul",
    "tile": "azulejo",
    "game": "jogo",
    "slot machine": "slot machine",
    "luck": "sorte",
}


class ScoredLabel(TypedDict):
    description: str
    score: float


class ImageAnalysis(TypedDict):
    ocr_text: str
    labels: List[ScoredLabel]
    logos: List[ScoredLabel]
    objects: List[ScoredLabel]


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _ai_settings() -> dict:
    return {
        "google_vision_api_key": _env("GOOGLE_VISION_API_KEY"),
        "gemini_api_key": _env("GEMINI_API_KEY"),
        "gemini_model": _env("GEMINI_MODEL", "gemini-2.0-flash"),
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
    return {"ocr_text": "", "labels": [], "logos": [], "objects": []}


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
                    {"type": "LABEL_DETECTION", "maxResults": 15},
                    {"type": "LOGO_DETECTION", "maxResults": 5},
                    {"type": "OBJECT_LOCALIZATION", "maxResults": 10},
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
        "objects": _parse_object_annotations(result.get("localizedObjectAnnotations")),
    }


def _parse_object_annotations(items: List[dict]) -> List[ScoredLabel]:
    parsed: List[ScoredLabel] = []
    for item in items or []:
        description = str(item.get("name", "")).strip()
        if not description:
            continue
        parsed.append(
            {
                "description": description,
                "score": float(item.get("score", 0) or 0),
            }
        )
    return parsed


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
                {"type_": vision.Feature.Type.LABEL_DETECTION, "max_results": 15},
                {"type_": vision.Feature.Type.LOGO_DETECTION, "max_results": 5},
                {"type_": vision.Feature.Type.OBJECT_LOCALIZATION, "max_results": 10},
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
    objects = [
        {"description": obj.name, "score": float(obj.score)}
        for obj in result.localized_object_annotations
        if obj.name
    ]

    return {"ocr_text": ocr_text, "labels": labels, "logos": logos, "objects": objects}


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


def _visual_terms(analysis: ImageAnalysis, min_score: float = 0.45) -> List[str]:
    terms: List[str] = []
    for source in (analysis["objects"], analysis["labels"]):
        for item in source:
            description = item["description"].strip()
            if item["score"] < min_score:
                continue
            if description.lower() in GENERIC_LABELS:
                continue
            if description not in terms:
                terms.append(description)
    return terms[:8]


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

    objects = [item["description"] for item in analysis["objects"][:6]]
    if objects:
        sections.append("Detected objects: " + ", ".join(objects))

    labels = _meaningful_labels(analysis["labels"])
    if labels:
        sections.append("Visual content detected: " + ", ".join(labels))

    if len(sections) == 1 and not ocr_text:
        relaxed = _visual_terms(analysis)
        if relaxed:
            sections.append("Visual content detected: " + ", ".join(relaxed))
        else:
            sections.append("Visual content detected: (no strong labels)")

    return "\n\n".join(sections)


def _product_like_labels(labels: List[str]) -> List[str]:
    return [
        label
        for label in labels
        if label.lower() not in VAGUE_LABELS and label.lower() not in GENERIC_LABELS
    ]


def _random_style_hint() -> str:
    return random.choice(WRITING_STYLES)


def _highlights_phrase(labels: List[str]) -> str:
    translated = [_label_to_portuguese(label).lower() for label in _product_like_labels(labels)[:3]]
    if not translated:
        return "detalhes únicos"
    if len(translated) == 1:
        return translated[0]
    if len(translated) == 2:
        return f"{translated[0]} e {translated[1]}"
    return f"{translated[0]}, {translated[1]} e {translated[2]}"


def _market_title_from_labels(labels: List[str], logos: List[str], ocr_title: str = "") -> str:
    if ocr_title:
        return _truncate(ocr_title.strip(" ,.;:-"), 80)
    if logos:
        patterns = (
            "{brand}",
            "{brand} — edição Cebola",
            "{brand} | coleção",
            "Novidade {brand}",
        )
        return _truncate(random.choice(patterns).format(brand=logos[0]), 80)

    product_labels = _product_like_labels(labels)
    if product_labels:
        primary = _label_to_portuguese(product_labels[0]).title()
        suffix = random.choice(TITLE_SUFFIXES)
        if len(product_labels) > 1 and random.random() < 0.6:
            secondary = _label_to_portuguese(product_labels[1]).title()
            joiner = random.choice(TITLE_JOINERS)
            return _truncate(f"{primary}{joiner}{secondary}", 80)
        patterns = (
            f"{primary} {suffix}",
            f"{suffix.title()} {primary}",
            f"{primary} da loja",
            f"Linha {primary}",
        )
        return _truncate(random.choice(patterns), 80)

    return random.choice(
        (
            "Artigo em destaque",
            "Novidade da loja",
            "Peça exclusiva",
            "Produto da semana",
            "Escolha especial Cebola",
        )
    )


def _market_description_from_labels(labels: List[str], logos: List[str]) -> str:
    product_labels = _product_like_labels(labels)
    highlights = _highlights_phrase(labels)

    if logos:
        intro = random.choice(DESCRIPTION_INTROS_BRAND).format(brand=logos[0])
    elif product_labels:
        intro = random.choice(DESCRIPTION_INTROS_LABELS).format(highlights=highlights)
    else:
        intro = random.choice(DESCRIPTION_INTROS_GENERIC)

    middle = random.choice(DESCRIPTION_MIDDLES)
    cta = random.choice(DESCRIPTION_CTAS)
    return _truncate(f"{intro} {middle} {cta}", 500)


def _generate_smart_fallback(analysis: ImageAnalysis) -> dict:
    lines = _normalize_ocr_lines(analysis["ocr_text"])
    logos = [item["description"] for item in analysis["logos"][:3]]
    labels = _meaningful_labels(analysis["labels"]) or _visual_terms(analysis)

    if lines:
        title = _truncate(lines[0], 80).strip(" ,.;:-")
        body = " ".join(lines[1:]) if len(lines) > 1 else ""
        body = _truncate(body, 500)

        if body and len(body) >= 40:
            return {"title": title, "description": body}

        if logos:
            description = _truncate(
                random.choice(
                    (
                        f"{title} — referência {logos[0]}. {random.choice(DESCRIPTION_MIDDLES)} {random.choice(DESCRIPTION_CTAS)}",
                        f"Edição {logos[0]} com foco em {title}. {random.choice(DESCRIPTION_MIDDLES)} {random.choice(DESCRIPTION_CTAS)}",
                    )
                ),
                500,
            )
            return {"title": title, "description": description}

        if labels:
            description = _market_description_from_labels(labels, logos)
            return {"title": _market_title_from_labels(labels, logos, title), "description": description}

        return {
            "title": title,
            "description": _truncate(
                random.choice(
                    (
                        f"{title}. {random.choice(DESCRIPTION_INTROS_GENERIC)} {random.choice(DESCRIPTION_CTAS)}",
                        f"{title} entra no catálogo com boa presença visual. {random.choice(DESCRIPTION_MIDDLES)} {random.choice(DESCRIPTION_CTAS)}",
                    )
                ),
                500,
            ),
        }

    if logos:
        title = _truncate(logos[0], 80)
        return {
            "title": title,
            "description": _market_description_from_labels(labels, logos),
        }

    if labels:
        return {
            "title": _market_title_from_labels(labels, logos),
            "description": _market_description_from_labels(labels, logos),
        }

    return {
        "title": "Novo produto",
        "description": (
            "Não foi possível analisar a imagem com detalhe suficiente. "
            "Configure GEMINI_API_KEY no servidor para descrições visuais completas, "
            "ou envie uma foto mais próxima com boa luz."
        ),
    }


def _detect_image_mime(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG"):
        return "image/png"
    if image_bytes.startswith(b"GIF"):
        return "image/gif"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _generate_with_gemini_vision(image_bytes: bytes, analysis: ImageAnalysis) -> dict:
    settings = _ai_settings()
    api_key = settings["gemini_api_key"]
    model = settings["gemini_model"]
    if not api_key:
        raise AiServiceError("GEMINI_API_KEY is not configured")

    encoded = base64.b64encode(image_bytes).decode("utf-8")
    mime_type = _detect_image_mime(image_bytes)
    vision_context = _build_vision_context(analysis)
    prompt = (
        "You help Portuguese marketplace merchants write sales-ready product listings from photos.\n"
        "Look at the image and write copy that would attract buyers on an online marketplace.\n"
        "Use the automated hints below only as support; trust the photo first.\n\n"
        f"Style for this listing: {_random_style_hint()}\n"
        "Vary sentence structure and vocabulary; do not reuse the same template as previous listings.\n\n"
        f"{vision_context}\n\n"
        "Return ONLY valid JSON, without markdown:\n"
        '{"title": "...", "description": "..."}\n\n'
        f"{MARKETING_RULES}"
    )

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        f"?key={api_key}"
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": mime_type, "data": encoded}},
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.9,
            "topP": 0.95,
            "maxOutputTokens": 512,
        },
    }

    try:
        response = requests.post(url, json=payload, timeout=90)
    except requests.RequestException as exc:
        raise AiServiceError(f"Gemini vision request failed: {exc}") from exc

    data = response.json()
    if response.status_code >= 400:
        message = data.get("error", {}).get("message", response.text)
        raise AiServiceError(f"Gemini vision failed: {message}")

    candidates = data.get("candidates") or []
    if not candidates:
        raise AiServiceError("Gemini vision returned no candidates")

    parts = candidates[0].get("content", {}).get("parts") or []
    content = next((part.get("text", "") for part in parts if part.get("text")), "")
    if not content:
        raise AiServiceError("Gemini vision returned an empty message")

    parsed = _parse_json_object(content)
    title = str(parsed.get("title", "")).strip()
    description = str(parsed.get("description", "")).strip()
    if not title or not description:
        raise AiServiceError("Gemini vision response is missing title or description")

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


def _request_minicpm_copy(base_url: str, model: str, vision_context: str, api_key: str) -> dict:
    prompt = (
        "You help Portuguese marketplace merchants create sales-ready product listings.\n"
        "Use the image analysis below to write copy that helps sell the item on an online marketplace.\n"
        "If there is little or no text, infer a compelling listing from logos and visual content.\n\n"
        f"Style for this listing: {_random_style_hint()}\n"
        "Vary sentence structure and vocabulary; do not reuse the same template as previous listings.\n\n"
        f"{vision_context}\n\n"
        "Respond with valid JSON only, without markdown:\n"
        '{"title": "...", "description": "..."}\n\n'
        f"{MARKETING_RULES}"
    )

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.85,
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
        "Google Vision analysis: %s chars of text, %s labels, %s logos, %s objects",
        len(analysis["ocr_text"]),
        len(analysis["labels"]),
        len(analysis["logos"]),
        len(analysis["objects"]),
    )

    settings = _ai_settings()
    if settings["gemini_api_key"]:
        try:
            result = _generate_with_gemini_vision(image_bytes, analysis)
            logger.info("Generated product copy via Gemini vision model=%s", settings["gemini_model"])
            return result
        except (AiServiceError, requests.RequestException) as exc:
            logger.warning("Gemini vision failed, using text pipeline: %s", exc)

    return generate_product_copy(analysis)
