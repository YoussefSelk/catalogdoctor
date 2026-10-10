"""Seeded synthetic catalog with a separate, cell-level injection manifest."""

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from catalogdoctor.models import scalar


def default_rules() -> str:
    return files("catalogdoctor").joinpath("default_rules.yml").read_text(encoding="utf-8")


def generate_catalog(seed: int = 42, rows: int = 1500) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if rows < 200:
        raise ValueError("Demo generation requires at least 200 rows")
    rng = np.random.default_rng(seed)
    templates = {
        "Electronics": [
            (
                "Wireless Bluetooth Headphones",
                "Audio headset with wireless pairing and rechargeable battery",
            ),
            (
                "Portable USB Speaker",
                "Compact audio speaker for music streaming and bluetooth sound",
            ),
        ],
        "Clothing": [
            ("Cotton Crewneck Shirt", "Soft breathable cotton apparel with comfortable fabric fit"),
            (
                "Lightweight Running Jacket",
                "Weather resistant outerwear with zip pockets and woven fabric",
            ),
        ],
        "Home": [
            (
                "Linen Cushion Cover",
                "Decorative sofa cushion with woven textile for living room decor",
            ),
            (
                "Wooden Storage Shelf",
                "Home organizer furniture with shelves for living room storage",
            ),
        ],
        "Kitchen": [
            (
                "Stainless Steel Saucepan",
                "Cooking pot with heat resistant handles for kitchen cookware",
            ),
            (
                "Silicone Baking Utensils",
                "Kitchen cooking tools for baking mixing and preparing food",
            ),
        ],
        "Sports": [
            ("Yoga Exercise Mat", "Fitness workout mat with non slip grip for yoga training"),
            (
                "Adjustable Dumbbell Set",
                "Strength training weights for exercise and fitness workouts",
            ),
        ],
    }
    categories = list(templates)
    records = []
    for i in range(rows):
        category = categories[i % len(categories)]
        name, description = templates[category][int(rng.integers(2))]
        edition = int(rng.integers(1, 80))
        records.append(
            {
                "product_id": i + 1,
                "sku": f"PRD-{i + 1:04d}",
                "name": f"{name} — Edition {edition}",
                "description": f"{description}. Series {edition}; "
                f"finish {rng.choice(['classic', 'premium', 'compact'])}.",
                "category": category,
                "price": round(float(rng.lognormal(3.4, 0.65)), 2),
                "stock": int(rng.poisson(35)),
                "supplier": f"Supplier {int(rng.integers(1, 9))}",
                "created_at": (
                    pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(rng.integers(365)))
                ).strftime("%Y-%m-%d"),
            }
        )
    df = pd.DataFrame(records)
    manifest: list[dict[str, Any]] = []

    def inject(position: int, field: str, value: Any, kind: str) -> None:
        manifest.append(
            {
                "row_ref": position,
                "field": field,
                "original": scalar(df.iloc[position][field]),
                "injected": scalar(value),
                "kind": kind,
            }
        )
        df.at[position, field] = value

    for i in range(10, 25):
        inject(i, "name", None, "missing_name")
    for i in range(35, 47):
        inject(i, "price", np.nan, "missing_price")
    for i in range(60, 68):
        inject(i, "product_id", int(df.iloc[i - 20]["product_id"]), "duplicate_id")
        inject(i, "sku", str(df.iloc[i - 20]["sku"]), "duplicate_sku")
    for i in range(80, 88):
        inject(i, "price", -round(float(rng.uniform(1, 100)), 2), "invalid_price")
    for i in range(100, 108):
        inject(i, "stock", -int(rng.integers(1, 20)), "negative_stock")
    for i in range(120, 125):
        inject(i, "category", "Unknown", "invalid_category")
    for i in range(140, 148):
        inject(i, "price", round(float(rng.uniform(250, 800)), 2), "unusual_price")
    for i in range(160, 175):
        current = str(df.iloc[i]["category"])
        inject(
            i,
            "category",
            categories[(categories.index(current) + 1) % len(categories)],
            "suspicious_label",
        )
    for i in range(180, 183):
        inject(i, "created_at", "2025-02-30", "invalid_date")
    for i in range(rows - 5, rows):
        for field in df.columns:
            inject(i, field, df.iloc[i - 100][field], "exact_duplicate")
    return df, manifest


def write_examples(directory: str | Path) -> None:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    df, manifest = generate_catalog()
    df.to_csv(destination / "catalog.csv", index=False, encoding="utf-8")
    df.to_parquet(destination / "catalog.parquet", index=False)
    engine = create_engine(f"sqlite:///{(destination / 'catalog.db').as_posix()}")
    try:
        df.to_sql("products", engine, if_exists="replace", index=False)
    finally:
        engine.dispose()
    (destination / "rules.yml").write_text(default_rules(), encoding="utf-8")
    (destination / "ground_truth.json").write_text(
        json.dumps(
            {"seed": 42, "row_reference": "zero-based original position", "issues": manifest},
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
