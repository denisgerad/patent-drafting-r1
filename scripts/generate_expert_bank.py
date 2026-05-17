#!/usr/bin/env python3
"""
scripts/generate_expert_bank.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PHASE 1 (OFFLINE) — Run this script ONCE per product type to generate
a Claude-powered expert question bank. The bank is stored locally and
used offline during all future patent processing. No patent data is
ever sent to Claude.

What is sent to Claude:
  - Product type label        (e.g. "flexible_heater_film")
  - Mechanism description     (e.g. "resistive heating element in polymer film")
  - Deployment context        (from checklist anti_patterns / application notes)
  - §112 enablement dimensions (generic legal framework, not patent-specific)

What is NEVER sent to Claude:
  - Patent document content
  - Inventor details
  - Any proprietary specifications

Usage:
  python scripts/generate_expert_bank.py --product-type flexible_heater_film
  python scripts/generate_expert_bank.py --product-type light_guide_plate
  python scripts/generate_expert_bank.py --all

Requirements:
  ANTHROPIC_API_KEY must be set in .env
  The product type must exist in product_type_checklists.json

Output:
  domain_reference_questions/{product_type}_bank.json
  Review and edit this file before putting it into production.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Allow running from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_ROOT             = Path(__file__).parent.parent
_CHECKLISTS_PATH  = _ROOT / "product_type_checklists.json"
_REF_DIR          = _ROOT / "domain_reference_questions"
_REF_DIR.mkdir(exist_ok=True)

# ── Mechanism hints (same as in scrutiny_workflow) ────────────────────────────
_MECHANISM_HINTS: dict[str, str] = {
    "flexible_heater_film":       "resistive heating element embedded in flexible polymer film",
    "light_guide_plate":          "dot pattern / light extraction structure on LGP surface",
    "oled_display":               "organic emissive stack with flexible substrate and encapsulation barrier",
    "led_array_pcb":              "interleaved day/NVG LED array on single PCB",
    "optical_coating":            "thin film optical layer stack with controlled refractive index",
    "power_electronics":          "switching power converter topology with magnetic components",
    "digital_fpga_asic":          "digital logic architecture with clock domain management",
    "sensor_signal_acquisition":  "analog front-end and ADC signal chain",
    "wireless_communication":     "RF transceiver and antenna design",
    "motor_actuator_drive":       "motor drive inverter with closed-loop control",
    "embedded_firmware":          "RTOS task scheduler and firmware architecture",
    "algorithm_method_patent":    "computational algorithm with defined input-output mapping",
    "machine_learning_ai":        "neural network model architecture and training process",
    "communication_protocol":     "message format and state machine protocol",
    "database_data_structure":    "data schema and index structure",
    "identity_verification_system": "time-limited code generation algorithm and identity mapping",
}


# ── Claude API call (no patent data) ─────────────────────────────────────────

def call_claude(prompt: str, model: str = "claude-sonnet-4-20250514") -> str:
    """Send prompt to Claude and return text response."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        message = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except ImportError:
        raise ImportError("pip install anthropic")
    except KeyError:
        raise ValueError("ANTHROPIC_API_KEY not set in .env")


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_expert_bank_prompt(
    product_type: str,
    mechanism: str,
    domain: str,
    focus_areas: list[str],
    units: list[str],
    anti_patterns: list[str],
) -> str:
    """
    Build the prompt sent to Claude.
    Contains ONLY product-type metadata — no patent document content.
    """
    focus_text    = "\n".join(f"  - {f}" for f in focus_areas) if focus_areas else "  - (general domain)"
    units_text    = ", ".join(units) if units else "standard SI units"
    ap_text       = "\n".join(f"  - {a}" for a in anti_patterns[:4]) if anti_patterns else "  - (none)"

    return f"""You are a senior patent analyst with deep expertise in {domain}.

Your task: generate a comprehensive expert question bank for the following invention category.
This bank will be used by a patent drafting system to ask inventors the right questions
to satisfy 35 U.S.C. §112 enablement requirements.

INVENTION CATEGORY: {product_type.replace("_", " ").title()}
ENABLING MECHANISM: {mechanism}
DOMAIN: {domain}

FOCUS AREAS for this category:
{focus_text}

MEASUREMENT UNITS required: {units_text}

SCOPE BOUNDARIES (do NOT generate questions about these):
{ap_text}

TASK: Generate 30-40 expert questions that a senior patent analyst would ask
an inventor of this type of invention. The inventor has provided a minimal
disclosure (one paragraph describing what the invention is). Your questions
must fill in ALL the technical gaps needed for §112 enablement.

Organise questions into exactly these six enablement dimensions:

[Physical Structure & Geometry]
Questions about dimensions, shapes, patterns, spatial arrangements, tolerances.
Require specific numeric values and drawings.

[Materials & Composition]
Questions about material identity, grade, supplier, composition, properties.
Require specific material names, CAS numbers, or grades — not "standard material".

[Fabrication & Process]
Questions about how the invention is made, step by step.
Require process parameters, temperatures, pressures, durations, tolerances.

[Performance Metrics & Prior Art Comparison]
Questions about measured performance with units.
Require comparison to prior art or conventional design baseline.

[Environmental & Operational Requirements]
Questions about operating conditions, qualification standards, deployment context.
Require specific standards (MIL-STD, ISO, IEC), ranges, and test methods.

[Failure Modes & Safety]
Questions about known failure mechanisms, safety limits, protection methods.
Require specific limits, test conditions, and mitigation strategies.

REQUIREMENTS for each question:
- Name the specific parameter, component, or condition being asked about
- Specify the unit of measurement where applicable
- Request drawings, formulas, or test data where relevant
- Do NOT use vague phrasing ("describe the design", "explain how it works")
- Each question should be answerable with a number, drawing, formula, or procedure

OUTPUT FORMAT: Return ONLY a JSON object. No preamble, no explanation, no markdown.
{{
  "product_type": "{product_type}",
  "domain": "{domain}",
  "mechanism": "{mechanism}",
  "generated_at": "ISO-8601 timestamp",
  "model_used": "{product_type}",
  "categories": [
    {{
      "name": "Physical Structure & Geometry",
      "questions": ["question 1", "question 2", ...]
    }},
    {{
      "name": "Materials & Composition",
      "questions": [...]
    }},
    {{
      "name": "Fabrication & Process",
      "questions": [...]
    }},
    {{
      "name": "Performance Metrics & Prior Art Comparison",
      "questions": [...]
    }},
    {{
      "name": "Environmental & Operational Requirements",
      "questions": [...]
    }},
    {{
      "name": "Failure Modes & Safety",
      "questions": [...]
    }}
  ],
  "total_questions": 0,
  "review_status": "PENDING_EXPERT_REVIEW"
}}
"""


# ── Bank generation ───────────────────────────────────────────────────────────

def generate_bank(product_type: str, dry_run: bool = False) -> dict:
    """
    Generate expert question bank for one product type.
    Returns the bank dict. Saves to domain_reference_questions/{type}_bank.json.
    """
    checklists = json.loads(_CHECKLISTS_PATH.read_text(encoding="utf-8"))
    if product_type not in checklists:
        raise ValueError(
            f"Product type '{product_type}' not found in product_type_checklists.json. "
            f"Available: {[k for k in checklists if not k.startswith('_')]}"
        )

    entry        = checklists[product_type]
    mechanism    = _MECHANISM_HINTS.get(product_type, f"{product_type} novel mechanism")
    domain       = entry.get("role", product_type.replace("_", " ").title())
    focus_areas  = entry.get("focus_areas", [])
    units        = entry.get("technical_units", [])
    anti_patterns = entry.get("anti_patterns", [])

    prompt = build_expert_bank_prompt(
        product_type, mechanism, domain, focus_areas, units, anti_patterns
    )

    logger.info("Generating expert bank for: %s", product_type)
    logger.info("Prompt length: %d chars", len(prompt))
    logger.info("Mechanism: %s", mechanism)

    if dry_run:
        logger.info("[DRY RUN] Would send this prompt to Claude:")
        print("\n" + "="*60)
        print(prompt)
        print("="*60 + "\n")
        return {}

    logger.info("Calling Claude Sonnet...")
    raw_response = call_claude(prompt)

    # Parse JSON response
    try:
        # Strip markdown fences if present
        import re
        clean = re.sub(r"```(?:json)?", "", raw_response).strip("`").strip()
        bank = json.loads(clean)
    except json.JSONDecodeError:
        # Try extracting first {...} block
        import re
        m = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if m:
            bank = json.loads(m.group())
        else:
            raise ValueError(f"Could not parse Claude response as JSON:\n{raw_response[:500]}")

    # Add metadata
    bank["generated_at"] = datetime.utcnow().isoformat() + "Z"
    bank["model_used"]   = "claude-sonnet-4-20250514"
    bank["review_status"] = "PENDING_EXPERT_REVIEW"

    # Count total questions
    total = sum(len(c.get("questions", [])) for c in bank.get("categories", []))
    bank["total_questions"] = total

    # Save to file
    out_path = _REF_DIR / f"{product_type}_bank.json"
    out_path.write_text(json.dumps(bank, indent=2, ensure_ascii=False))

    logger.info(
        "Saved: %s  (%d questions across %d categories)",
        out_path.name, total, len(bank.get("categories", []))
    )
    logger.info("IMPORTANT: Review %s before marking as APPROVED", out_path.name)

    return bank


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate expert question banks using Claude (offline, one-time per product type)"
    )
    parser.add_argument(
        "--product-type", "-p",
        help="Product type key from product_type_checklists.json"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Generate banks for all product types in checklists"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print prompt without calling Claude API"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available product types"
    )
    args = parser.parse_args()

    checklists = json.loads(_CHECKLISTS_PATH.read_text(encoding="utf-8"))
    available  = [k for k in checklists if not k.startswith("_")]

    if args.list:
        print("Available product types:")
        for pt in available:
            bank_path = _REF_DIR / f"{pt}_bank.json"
            status = "✅ bank exists" if bank_path.exists() else "⬜ no bank yet"
            print(f"  {pt:<40} {status}")
        return

    if args.all:
        targets = available
    elif args.product_type:
        targets = [args.product_type]
    else:
        parser.print_help()
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Expert Bank Generator — Phase 1 (Offline)")
    print(f"{'='*60}")
    print(f"Targets: {targets}")
    print(f"Dry run: {args.dry_run}")
    print(f"{'='*60}\n")

    for pt in targets:
        print(f"\n--- {pt} ---")
        try:
            bank = generate_bank(pt, dry_run=args.dry_run)
            if not args.dry_run:
                print(f"✅ {pt}: {bank.get('total_questions', 0)} questions generated")
                print(f"   Review: domain_reference_questions/{pt}_bank.json")
                print(f"   Set review_status to 'APPROVED' when ready for production")
        except Exception as exc:
            print(f"❌ {pt}: {exc}")
            logger.exception("Bank generation failed for %s", pt)

    print(f"\n{'='*60}")
    print("Next steps:")
    print("1. Review each _bank.json file — edit questions as needed")
    print("2. Change 'review_status' from 'PENDING_EXPERT_REVIEW' to 'APPROVED'")
    print("3. The system will use approved banks automatically in production")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
