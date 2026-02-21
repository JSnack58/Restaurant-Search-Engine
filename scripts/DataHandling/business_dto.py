"""
Business Data Transfer Object
==============================
Parses a raw ES ``_source`` dict into a clean, typed dataclass.
Attribute values are classified against the constants in ``constants.py``
so downstream code (query generation, text building) never touches raw dicts.

Usage
-----
    from DataHandling.business_dto import BusinessDTO

    dto = BusinessDTO.from_raw(hit["_source"])
    print(dto.categories)       # ['Italian', 'Pizza']
    print(dto.boolean_attrs)    # {'OutdoorSeating', 'GoodForKids'}
    print(dto.phrases)          # ['outdoor seating', 'kid-friendly', ...]
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from types import MappingProxyType

from constants import (
    ATTRIBUTE_PHRASES,
    BAR_CATEGORIES,
    BOOLEAN_ATTRIBUTES,
    ENUM_ATTRIBUTES,
    NESTED_ATTRIBUTES,
    RESTAURANT_CATEGORIES,
)

_VALID_CATEGORIES = RESTAURANT_CATEGORIES | BAR_CATEGORIES


@dataclass(frozen=True, slots=True)
class BusinessDTO:
    """Immutable, pre-digested representation of a Yelp business."""

    business_id: str
    name: str
    city: str
    state: str
    address: str
    latitude: float
    longitude: float
    categories: tuple[str, ...]                         # filtered & stripped
    boolean_attrs: frozenset[str]                       # e.g. {"OutdoorSeating", "GoodForKids"}
    enum_attrs: MappingProxyType                        # e.g. {"Alcohol": "full_bar", "WiFi": "free"}
    nested_attrs: MappingProxyType                      # e.g. {"Ambience": frozenset({"romantic", "casual"})}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_raw(cls, raw: dict) -> BusinessDTO:
        """Build a ``BusinessDTO`` from a raw ES ``_source`` dict.

        Parameters
        ----------
        raw : dict
            A document straight from ``yelp_businesses_raw``.
        """
        # ── Identity & location ──────────────────────────────────────
        business_id = raw.get("business_id", "")
        name = raw.get("name", "")
        city = raw.get("city", "")
        state = raw.get("state", "")
        address = raw.get("address", "")
        latitude = float(raw.get("latitude", 0.0))
        longitude = float(raw.get("longitude", 0.0))

        # ── Categories ───────────────────────────────────────────────
        raw_cats = raw.get("categories", "")
        if isinstance(raw_cats, list):
            stripped = [c.strip() for c in raw_cats]
        elif raw_cats:
            stripped = [c.strip() for c in raw_cats.split(",")]
        else:
            stripped = []
        categories = tuple(c for c in stripped if c in _VALID_CATEGORIES)

        # ── Attributes ───────────────────────────────────────────────
        attrs = raw.get("attributes") or {}
        boolean_attrs: set[str] = set()
        enum_attrs: dict[str, str] = {}
        nested_attrs: dict[str, set[str]] = {}

        # ✏️ TODO: iterate over `attrs.items()` and classify each key:
        #
        # For each (key, value) in attrs:
        #   1. If key is in BOOLEAN_ATTRIBUTES and value == "True":
        #          add key to boolean_attrs
        #
        #   2. If key is in ENUM_ATTRIBUTES:
        #          clean the value (strip u'' wrapper), store in enum_attrs[key]
        #
        #   3. If key is in NESTED_ATTRIBUTES:
        #          parse the dict string with ast.literal_eval(value),
        #          collect sub-keys where the sub-value is True,
        #          store as nested_attrs[key] = {sub_keys...}
        #
        #   4. Skip keys that don't match any category (they're not useful)
        #
        # Remember to handle None values and malformed strings gracefully.
        for attr, val in attrs.items():
            if attr in BOOLEAN_ATTRIBUTES and val == "True":
                boolean_attrs.add(attr)
            elif attr in ENUM_ATTRIBUTES:
                enum_attrs[attr] = str(val).replace("u'", "").replace("'", "").strip()
            elif attr in NESTED_ATTRIBUTES:
                try:
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, dict):
                        nested_attrs[attr] = {k for k, v in parsed.items() if v is True}
                except (ValueError, SyntaxError):
                    pass
        return cls(
            business_id=business_id,
            name=name,
            city=city,
            state=state,
            address=address,
            latitude=latitude,
            longitude=longitude,
            categories=categories,
            boolean_attrs=frozenset(boolean_attrs),
            enum_attrs=MappingProxyType(enum_attrs),
            nested_attrs=MappingProxyType({k: frozenset(v) for k, v in nested_attrs.items()}),
        )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def location(self) -> str:
        """Return ``'city, state'`` string."""
        return f"{self.city}, {self.state}"

    @property
    def phrases(self) -> list[str]:
        """Return natural-language phrases for all active attributes.

        Looks up each active attribute in ``ATTRIBUTE_PHRASES`` from
        constants.py and returns matching phrases.

        ✏️ TODO: implement this property.  Hints:
        - For boolean_attrs: look up each key directly in ATTRIBUTE_PHRASES
        - For enum_attrs: look up "{key}.{value}" in ATTRIBUTE_PHRASES
        - For nested_attrs: for each (key, sub_keys), look up "{key}.{sub_key}"
        - Return a list of all matched phrase strings
        """
        res = []
        res.extend([phrase for attr in self.boolean_attrs if (phrase :=ATTRIBUTE_PHRASES.get(attr))])
        res.extend([phrase for attr, val in self.enum_attrs.items() if (phrase := ATTRIBUTE_PHRASES.get(f"{attr}.{val}"))])
        res.extend([
            phrase
            for attr, val in self.nested_attrs.items()
            for nested_val in val
            if (phrase := ATTRIBUTE_PHRASES.get(f"{attr}.{nested_val}"))
        ])
        return res
