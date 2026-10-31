
"""
Lesson 2.7: WhatsApp Order Extractor

Extract orders from messy English and Roman Urdu messages,
validate the results, calculate trusted prices in Python,
and save orders to JSON and CSV files.
"""

import csv
import json
import re
from pathlib import Path

from langchain.chat_models import init_chat_model


# The menu is the source of truth for prices.
PRICES = {
    "cupcake": 250,
    "chocolate cake": 1500,
    "brownie": 300,
    "donut": 200,
}

MESSAGES = [
    "Hi, I need 1 chocolate cake. My name is Sara, address: DHA Lahore.",
    "Salam! 2 cupcakes aur 1 chocolate cake chahiye. Ali, house 12 Gulberg",
    "1 pizza and 2 brownies, Bilal, Model Town",
    "Hi do you deliver to Johar Town?",
    "Bhai 3 brownies de dein please. Sana, Johar Town",
]

ORDER_SCHEMA = {
    "type": "object",
    "properties": {
        "is_order": {"type": "boolean"},
        "customer_name": {"type": ["string", "null"]},
        "address": {"type": ["string", "null"]},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item": {"type": "string"},
                    "quantity": {"type": "integer"},
                },
                "required": ["item", "quantity"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["is_order", "customer_name", "address", "items"],
    "additionalProperties": False,
}

# Why: Temperature zero helps make extraction more consistent.
# Why: Create the model once, before processing the messages.
extractor = init_chat_model(
    "ollama:llama3.2",
    temperature=0,
    format=ORDER_SCHEMA,
)

ROOT = Path(__file__).resolve().parent.parent
ORDERS_FILE = ROOT / "orders.json"
CSV_FILE = ROOT / "orders.csv"
HUNT_FILE = ROOT / "docs" / "hallucination-hunt.md"


def extract_order(message: str) -> dict | None:
    """Extract structured order data from a customer message."""
    prompt = f"""
Extract order details from this customer message.

Return JSON matching the supplied schema.
- Set is_order to false for questions or greetings that are not orders.
- Do not invent customer names or addresses; use null if missing.
- Use lowercase item names.
- Include off-menu items so they can be flagged by Python.
- Extract positive integer quantities.
- Do not calculate prices or totals.

Customer message:
{message}
"""
    response = extractor.invoke(prompt)

    try:
        data = json.loads(response.text)
    except json.JSONDecodeError:
        print("  CHECK: Invalid JSON returned by AI; message skipped.")
        return None

    if not isinstance(data, dict):
        print("  CHECK: Expected a JSON object; message skipped.")
        return None

    return data


def name_is_in_message(name: str | None, message: str) -> bool:
    """Check that a name appears in the original message as a whole phrase."""
    if not isinstance(name, str) or not name.strip():
        return False

    pattern = r"(?<!\w)" + re.escape(name.strip()) + r"(?!\w)"
    return re.search(pattern, message, re.IGNORECASE) is not None


def process_order(message: str, data: dict) -> tuple[dict | None, int]:
    """Validate extracted data and calculate totals from PRICES."""
    checks = 0

    is_order = data.get("is_order", False)
    customer_name = data.get("customer_name", None)
    address = data.get("address", None)
    items = data.get("items", [])

    if not isinstance(is_order, bool):
        print("  CHECK: is_order must be true or false.")
        return None, 1

    if not is_order:
        print("  Not an order: needs a human reply.")
        return None, checks

    if not isinstance(items, list):
        print("  CHECK: items must be a list.")
        return None, 1

    if not name_is_in_message(customer_name, message):
        print(f"  CHECK: customer name {customer_name!r} is not verified.")
        customer_name = None
        checks += 1

    if address is not None and (
        not isinstance(address, str)
        or not address.strip()
        or address.casefold() not in message.casefold()
    ):
        print(f"  CHECK: address {address!r} is not verified.")
        address = None
        checks += 1

    clean_items = []
    total = 0

    for entry in items:
        if not isinstance(entry, dict):
            print("  CHECK: Invalid item entry; skipped.")
            checks += 1
            continue

        item_name = entry.get("item", "")
        quantity = entry.get("quantity", 0)

        if not isinstance(item_name, str) or not item_name.strip():
            print("  CHECK: Missing item name; skipped.")
            checks += 1
            continue

        item_name = item_name.strip().casefold()

        if item_name not in PRICES:
            print(f"  CHECK: '{item_name}' is not on the menu")
            checks += 1
            continue

        if (
            not isinstance(quantity, int)
            or isinstance(quantity, bool)
            or quantity <= 0
        ):
            print(f"  CHECK: Invalid quantity for '{item_name}'; skipped.")
            checks += 1
            continue

        # Why: Only trusted menu prices are used to calculate money.
        line_total = PRICES[item_name] * quantity
        total += line_total

        print(f"  {quantity} x {item_name} = {line_total} PKR")

        clean_items.append({
            "item": item_name,
            "quantity": quantity,
            "unit_price": PRICES[item_name],
            "line_total": line_total,
        })

    if not clean_items:
        print("  CHECK: No valid menu items found; order not saved.")
        checks += 1
        return None, checks

    print(
        f"  Customer: {customer_name or 'Not verified'}, "
        f"Address: {address or 'Not provided'}"
    )
    print(f"  Total: {total} PKR")

    return {
        "customer_name": customer_name,
        "address": address,
        "original_message": message,
        "items": clean_items,
        "total": total,
    }, checks


def save_csv(orders: list[dict]) -> None:
    """Save one CSV row per accepted order item."""
    with CSV_FILE.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["customer", "item", "quantity", "line_total"])

        for order in orders:
            for item in order.get("items", []):
                # Each item gets its own spreadsheet row.
                writer.writerow([
                    order.get("customer_name") or "Not verified",
                    item.get("item", ""),
                    item.get("quantity", 0),
                    item.get("line_total", 0),
                ])


def main() -> None:
    """Run extraction, validation, reporting, and file exports."""
    orders = []
    total_value = 0
    total_checks = 0

    for message in MESSAGES:
        print(f"\nMessage: {message}")

        try:
            data = extract_order(message)
        except Exception as error:
            print(
                f"  CHECK: Extraction failed ({type(error).__name__}); "
                "message skipped."
            )
            total_checks += 1
            continue

        if data is None:
            total_checks += 1
            continue

        order, checks = process_order(message, data)
        total_checks += checks

        if order is not None:
            orders.append(order)
            total_value += order.get("total", 0)

    # Save structured orders as readable JSON.
    with ORDERS_FILE.open("w", encoding="utf-8") as file:
        json.dump(orders, file, indent=2, ensure_ascii=False)

    save_csv(orders)

    print("\n" + "=" * 50)
    print(
        f"Orders found: {len(orders)} | "
        f"Total value: {total_value:,} PKR | "
        f"Checks: {total_checks}"
    )
    print(f"Saved {len(orders)} orders to {ORDERS_FILE.name}")
    print(f"Saved item rows to {CSV_FILE.name}")

    HUNT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not HUNT_FILE.exists():
        HUNT_FILE.write_text(
            "# Hallucination Hunt — Lesson 2.7\n\n"
            "Run these tests and record actual results.\n\n"
            "- [ ] Off-menu pizza is flagged and excluded from totals.\n"
            "- [ ] Invented customer name is flagged.\n"
            "- [ ] Missing name is marked unverified.\n"
            "- [ ] A non-order is not saved as an order.\n"
            "- [ ] Invalid JSON does not crash the batch.\n"
            "- [ ] Zero or negative quantities are rejected.\n"
            "- [ ] Multiple-item totals match manual calculations.\n\n"
            "## Findings\n\n"
            "Record test inputs, observed behavior, and fixes here.\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
