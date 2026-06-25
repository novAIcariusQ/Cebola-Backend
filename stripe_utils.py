import logging
import os
from typing import List, Optional, Tuple, Union

import stripe

logger = logging.getLogger("CebolaStripe")

STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "").strip()
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "").strip()
STRIPE_CURRENCY = os.environ.get("STRIPE_CURRENCY", "eur").strip().lower()
STRIPE_SUCCESS_URL = os.environ.get(
    "STRIPE_SUCCESS_URL",
    "http://localhost:5173/checkout/success?session_id={CHECKOUT_SESSION_ID}",
).strip()
STRIPE_CANCEL_URL = os.environ.get(
    "STRIPE_CANCEL_URL",
    "http://localhost:5173/basket",
).strip()


class StripeNotConfiguredError(Exception):
    pass


def is_stripe_configured() -> bool:
    return bool(STRIPE_SECRET_KEY)


def _configure_stripe() -> None:
    if not STRIPE_SECRET_KEY:
        raise StripeNotConfiguredError("STRIPE_SECRET_KEY is not configured")
    stripe.api_key = STRIPE_SECRET_KEY


def build_mock_payment_url(order_id: str, guest_order_id: str) -> str:
    template = os.environ.get(
        "PAYMENT_URL_TEMPLATE",
        "http://127.0.0.1:3003/api/orders/{order_id}/mock-pay",
    )
    return template.format(order_id=order_id, guest_order_id=guest_order_id)


def create_checkout_session(
    order_id: str,
    guest_order_id: str,
    shop_name: str,
    line_items: List[dict],
    customer_email: Optional[str] = None,
) -> Union[str, Tuple[str, str]]:
    if not is_stripe_configured():
        logger.warning("Stripe is not configured; using mock payment URL for order %s", order_id)
        return build_mock_payment_url(order_id, guest_order_id)

    _configure_stripe()

    stripe_line_items = []
    for item in line_items:
        unit_amount = int(round(float(item["price"]) * 100))
        if unit_amount < 1:
            raise ValueError(f"Invalid price for Stripe line item: {item['title']}")
        stripe_line_items.append({
            "price_data": {
                "currency": STRIPE_CURRENCY,
                "product_data": {
                    "name": item["title"],
                    "metadata": {
                        "product_id": item.get("product_id") or "",
                        "shop_name": shop_name,
                    },
                },
                "unit_amount": unit_amount,
            },
            "quantity": int(item["quantity"]),
        })

    session_params = {
        "mode": "payment",
        "line_items": stripe_line_items,
        "success_url": STRIPE_SUCCESS_URL,
        "cancel_url": STRIPE_CANCEL_URL,
        "client_reference_id": order_id,
        "metadata": {
            "order_id": order_id,
            "guest_order_id": guest_order_id,
        },
    }
    if customer_email:
        session_params["customer_email"] = customer_email

    session = stripe.checkout.Session.create(**session_params)
    if not session.url:
        raise RuntimeError("Stripe Checkout Session was created without a redirect URL")
    return session.url, session.id


def construct_webhook_event(payload: bytes, signature_header: Optional[str]):
    if not STRIPE_WEBHOOK_SECRET:
        raise StripeNotConfiguredError("STRIPE_WEBHOOK_SECRET is not configured")
    _configure_stripe()
    return stripe.Webhook.construct_event(payload, signature_header, STRIPE_WEBHOOK_SECRET)
