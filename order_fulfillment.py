import io
import json
import logging
import os
import smtplib
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Optional

import qrcode
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

from database import db

logger = logging.getLogger("CebolaFulfillment")


def parse_order_ids(raw_value: Optional[str]) -> List[str]:
    if not raw_value:
        return []
    try:
        parsed = json.loads(raw_value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass
    return []


def append_user_order_id(user_id: str, order_id: str) -> None:
    user = db.execute_one("SELECT order_ids FROM users WHERE id = %s", (user_id,))
    if not user:
        return
    order_ids = parse_order_ids(user.get("order_ids"))
    if order_id not in order_ids:
        order_ids.append(order_id)
    db.execute_write(
        "UPDATE users SET order_ids = %s WHERE id = %s",
        (json.dumps(order_ids), user_id),
    )


def build_qr_png(order_id: str, guest_order_id: str) -> bytes:
    payload = f"order:{order_id}|guest:{guest_order_id}"
    image = qrcode.make(payload)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def build_receipt_pdf(
    order: dict,
    items: List[dict],
    shop_name: str,
) -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 20 * mm

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(20 * mm, y, "Cebola Order Receipt")
    y -= 10 * mm

    pdf.setFont("Helvetica", 11)
    pdf.drawString(20 * mm, y, f"Shop: {shop_name}")
    y -= 6 * mm
    pdf.drawString(20 * mm, y, f"Order ID: {order['id']}")
    y -= 6 * mm
    if order.get("guest_order_id"):
        pdf.drawString(20 * mm, y, f"Guest order ID: {order['guest_order_id']}")
        y -= 6 * mm
    pdf.drawString(20 * mm, y, f"Status: {order.get('status', '')}")
    y -= 6 * mm
    pdf.drawString(20 * mm, y, f"Total: {float(order['total_amount']):.2f} EUR")
    y -= 10 * mm

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(20 * mm, y, "Items")
    y -= 8 * mm
    pdf.setFont("Helvetica", 10)

    for item in items:
        line = (
            f"{item['product_title']} x{item['quantity']} "
            f"@ {float(item['price_at_time']):.2f} EUR"
        )
        pdf.drawString(20 * mm, y, line[:90])
        y -= 6 * mm
        if y < 30 * mm:
            pdf.showPage()
            y = height - 20 * mm

    pdf.drawString(20 * mm, 20 * mm, "Present this QR code at pickup.")
    pdf.save()
    return buffer.getvalue()


def _smtp_settings() -> Optional[dict]:
    host = os.environ.get("SMTP_HOST", "").strip()
    port = int(os.environ.get("SMTP_PORT", "587"))
    username = os.environ.get("SMTP_USERNAME", "").strip()
    password = os.environ.get("SMTP_PASSWORD", "").strip()
    sender = os.environ.get("SMTP_FROM_EMAIL", username).strip()
    use_tls = os.environ.get("SMTP_USE_TLS", "true").lower() in ("1", "true", "yes")
    if not host or not sender:
        return None
    return {
        "host": host,
        "port": port,
        "username": username,
        "password": password,
        "sender": sender,
        "use_tls": use_tls,
    }


def send_order_receipt_email(
    to_email: str,
    order_id: str,
    guest_order_id: str,
    qr_png: bytes,
    pdf_bytes: bytes,
) -> bool:
    settings = _smtp_settings()
    if not settings:
        logger.warning(
            "SMTP is not configured; skipping receipt email for order %s to %s",
            order_id,
            to_email,
        )
        return False

    message = MIMEMultipart()
    message["Subject"] = f"Cebola order confirmation {guest_order_id or order_id}"
    message["From"] = settings["sender"]
    message["To"] = to_email
    message.attach(
        MIMEText(
            (
                "Thank you for your order.\n\n"
                f"Order ID: {order_id}\n"
                f"Guest order ID: {guest_order_id}\n\n"
                "Your QR code and PDF receipt are attached."
            ),
            "plain",
            "utf-8",
        )
    )

    qr_attachment = MIMEImage(qr_png, name=f"order-{guest_order_id or order_id}-qr.png")
    message.attach(qr_attachment)

    pdf_attachment = MIMEApplication(pdf_bytes, _subtype="pdf")
    pdf_attachment.add_header(
        "Content-Disposition",
        "attachment",
        filename=f"order-{guest_order_id or order_id}.pdf",
    )
    message.attach(pdf_attachment)

    with smtplib.SMTP(settings["host"], settings["port"], timeout=30) as smtp:
        if settings["use_tls"]:
            smtp.starttls()
        if settings["username"]:
            smtp.login(settings["username"], settings["password"])
        smtp.send_message(message)

    logger.info("Receipt email sent for order %s to %s", order_id, to_email)
    return True


def deduct_order_stock(order_id: str) -> None:
    item_rows = db.execute_query("SELECT * FROM order_items WHERE order_id = %s", (order_id,))
    for item in item_rows:
        if not item.get("product_id"):
            continue
        product = db.execute_one("SELECT * FROM products WHERE id = %s", (item["product_id"],))
        if not product:
            continue
        new_quantity = int(product["quantity"]) - int(item["quantity"])
        is_available = new_quantity > 0 and bool(product["is_available"])
        db.execute_write(
            "UPDATE products SET quantity = %s, is_available = %s WHERE id = %s",
            (new_quantity, is_available, item["product_id"]),
        )


def fulfill_paid_order(order_id: str, customer_email: Optional[str] = None) -> None:
    order = db.execute_one("SELECT * FROM orders WHERE id = %s", (order_id,))
    if not order:
        logger.error("Cannot fulfill missing order %s", order_id)
        return
    if order.get("status") == "paid":
        logger.info("Order %s already fulfilled", order_id)
        return

    shop = db.execute_one("SELECT name FROM shops WHERE id = %s", (order["shop_id"],))
    shop_name = shop["name"] if shop else order["shop_id"]
    item_rows = db.execute_query("SELECT * FROM order_items WHERE order_id = %s", (order_id,))
    guest_order_id = order.get("guest_order_id") or ""
    qr_payload = order.get("qr_code_data") or f"order:{order_id}"

    qr_png = build_qr_png(order_id, guest_order_id)
    pdf_bytes = build_receipt_pdf(order, item_rows, shop_name)

    resolved_email = customer_email or order.get("customer_email") or None

    db.execute_write(
        "UPDATE orders SET status = %s, customer_email = %s, qr_code_data = %s WHERE id = %s",
        ("paid", resolved_email, qr_payload, order_id),
    )
    deduct_order_stock(order_id)

    if order.get("user_id"):
        append_user_order_id(order["user_id"], order_id)
        if not resolved_email:
            user = db.execute_one("SELECT email FROM users WHERE id = %s", (order["user_id"],))
            resolved_email = user.get("email") if user else None

    if resolved_email:
        send_order_receipt_email(resolved_email, order_id, guest_order_id, qr_png, pdf_bytes)
