from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from src.data.clean_complaints import clean_complaints_dataframe
from src.data.clean_recalls import clean_recalls_dataframe
from src.rag.metadata_tagger import tag_document


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DOCS_DIR = RAW_DIR / "automotive_docs"
RECALLS_PATH = RAW_DIR / "recalls_sample.csv"
COMPLAINTS_PATH = RAW_DIR / "complaints_sample.csv"

GENERATED_SOURCE = "generated_fallback_nhtsa_style_not_real"
USER_SOURCE = "real_or_user_supplied_nhtsa_style"


def _needs_fallback(path: Path, required_columns: list[str], min_rows: int = 1) -> bool:
    if not path.exists():
        return True
    try:
        df = pd.read_csv(path)
    except Exception:
        return True
    if len(df) < min_rows:
        return True
    return any(column not in df.columns for column in required_columns)


def ensure_sample_datasets(row_count: int = 200) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    recall_columns = [
        "CAMPNO",
        "MFGNAME",
        "COMPONENT",
        "YEARTXT",
        "MAKETXT",
        "MODELTXT",
        "SUMMARY",
        "CONSEQUENCE",
        "REMEDY",
    ]
    complaint_columns = ["ODINO", "MFR_NAME", "COMPONENT", "VEHICLE_MODEL_YEAR", "MAKE", "MODEL", "CDESCR"]

    if _needs_fallback(RECALLS_PATH, recall_columns):
        _generate_fallback_recalls(row_count).to_csv(RECALLS_PATH, index=False)
    if _needs_fallback(COMPLAINTS_PATH, complaint_columns):
        _generate_fallback_complaints(row_count).to_csv(COMPLAINTS_PATH, index=False)


def _vehicle_catalog() -> list[dict[str, str]]:
    return [
        {"manufacturer": "Northstar Mobility LLC", "make": "Northstar", "model": "Trail LX"},
        {"manufacturer": "Orion Automotive Group", "make": "Orion", "model": "Metro EV"},
        {"manufacturer": "Apex Motor Works", "make": "Apex", "model": "Summit"},
        {"manufacturer": "Helio Motors Inc.", "make": "Helio", "model": "Voltiva"},
        {"manufacturer": "Continental Passenger Vehicles", "make": "CPV", "model": "Ranger"},
        {"manufacturer": "Riverton Truck Co.", "make": "Riverton", "model": "Hauler"},
        {"manufacturer": "Cobalt Auto Corporation", "make": "Cobalt", "model": "Vista"},
        {"manufacturer": "Meridian Vehicle Systems", "make": "Meridian", "model": "Crossline"},
    ]


def _component_scenarios() -> list[dict[str, str]]:
    return [
        {
            "component": "AIR BAGS",
            "issue": "the air bag control module may intermittently lose communication",
            "consequence": "air bags may not deploy as intended during a crash, increasing injury risk",
            "remedy": "dealers will inspect the wiring harness and update the restraint control software",
        },
        {
            "component": "SERVICE BRAKES, HYDRAULIC",
            "issue": "the brake actuator may report incorrect pressure under repeated stop-and-go operation",
            "consequence": "reduced brake assist can extend stopping distance and increase crash risk",
            "remedy": "dealers will replace the hydraulic control unit and recalibrate the brake actuator",
        },
        {
            "component": "ELECTRICAL SYSTEM",
            "issue": "a body control module software fault may reset the instrument cluster",
            "consequence": "loss of driver information can distract the driver and increase crash risk",
            "remedy": "dealers will install updated body control module software free of charge",
        },
        {
            "component": "STEERING",
            "issue": "the steering intermediate shaft fastener may have been under-torqued",
            "consequence": "steering looseness can reduce vehicle control and increase crash risk",
            "remedy": "dealers will inspect and tighten the fastener to the specified torque",
        },
        {
            "component": "FUEL SYSTEM, GASOLINE",
            "issue": "a fuel line quick connector may not be fully seated",
            "consequence": "fuel leakage in the presence of an ignition source can increase fire risk",
            "remedy": "dealers will inspect the connector and replace the fuel line if needed",
        },
        {
            "component": "SEAT BELTS",
            "issue": "the second-row buckle anchor may not meet retention specifications",
            "consequence": "occupant restraint performance may be reduced during a crash",
            "remedy": "dealers will replace the buckle anchor assembly",
        },
        {
            "component": "POWER TRAIN",
            "issue": "a transmission park pawl calibration error may allow vehicle movement",
            "consequence": "unexpected vehicle rollaway can increase injury or crash risk",
            "remedy": "dealers will update the transmission control module calibration",
        },
        {
            "component": "SUSPENSION",
            "issue": "front lower control arm bushings may crack prematurely",
            "consequence": "degraded handling may increase crash risk",
            "remedy": "dealers will inspect and replace affected lower control arms",
        },
        {
            "component": "TIRES",
            "issue": "the tire placard may show an incorrect cold inflation pressure",
            "consequence": "improper tire inflation can reduce handling stability and increase crash risk",
            "remedy": "dealers will install a corrected tire information placard",
        },
    ]


def _generate_fallback_recalls(row_count: int) -> pd.DataFrame:
    # Generated fallback data is synthetic NHTSA-style data, not real NHTSA data.
    # It exists only so Phase 1 can run without bundled government CSVs.
    rng = random.Random(360)
    vehicles = _vehicle_catalog()
    scenarios = _component_scenarios()
    rows = []

    for index in range(row_count):
        vehicle = rng.choice(vehicles)
        scenario = rng.choice(scenarios)
        model_year = rng.randint(2015, 2026)
        campaign_year = rng.randint(2020, 2026)
        campaign_number = f"{campaign_year % 100:02d}V{100000 + index:06d}"
        summary = (
            f"{vehicle['manufacturer']} is recalling certain {model_year} "
            f"{vehicle['make']} {vehicle['model']} vehicles because {scenario['issue']}."
        )
        rows.append(
            {
                "CAMPNO": campaign_number,
                "MFGNAME": vehicle["manufacturer"],
                "COMPONENT": scenario["component"],
                "YEARTXT": model_year,
                "MAKETXT": vehicle["make"],
                "MODELTXT": vehicle["model"],
                "SUMMARY": summary,
                "CONSEQUENCE": scenario["consequence"].capitalize() + ".",
                "REMEDY": scenario["remedy"].capitalize() + ".",
                "document_id": f"recall-{campaign_number}",
                "document_type": "recall_notice",
                "sensitivity_level": "public",
                "DATA_SOURCE": GENERATED_SOURCE,
            }
        )

    return pd.DataFrame(rows)


def _generate_fallback_complaints(row_count: int) -> pd.DataFrame:
    # Generated fallback data is synthetic NHTSA-style data, not real NHTSA data.
    # It should never be treated as a real complaint extract.
    rng = random.Random(361)
    vehicles = _vehicle_catalog()
    scenarios = _component_scenarios()
    symptoms = [
        "warning lamp illuminated while driving",
        "intermittent warning appeared during cold start",
        "dealer could not duplicate the concern on the first visit",
        "condition occurred twice within the same month",
        "owner reported reduced confidence during highway operation",
    ]
    rows = []

    for index in range(row_count):
        vehicle = rng.choice(vehicles)
        scenario = rng.choice(scenarios)
        model_year = rng.randint(2015, 2026)
        complaint_id = 11500000 + index
        description = (
            f"The owner of a {model_year} {vehicle['make']} {vehicle['model']} reported "
            f"{symptoms[index % len(symptoms)]}. The concern involved {scenario['component'].lower()} "
            f"and may relate to {scenario['issue']}."
        )
        rows.append(
            {
                "ODINO": complaint_id,
                "MFR_NAME": vehicle["manufacturer"],
                "COMPONENT": scenario["component"],
                "VEHICLE_MODEL_YEAR": model_year,
                "MAKE": vehicle["make"],
                "MODEL": vehicle["model"],
                "CDESCR": description,
                "CRASH": "N",
                "FIRE": "N",
                "INJURED": 0,
                "DEATHS": 0,
                "DATEA": f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/2025",
                "document_id": f"complaint-{complaint_id}",
                "document_type": "consumer_complaint",
                "sensitivity_level": "public",
                "DATA_SOURCE": GENERATED_SOURCE,
            }
        )

    return pd.DataFrame(rows)


def load_recalls() -> pd.DataFrame:
    ensure_sample_datasets()
    return clean_recalls_dataframe(pd.read_csv(RECALLS_PATH))


def load_complaints() -> pd.DataFrame:
    ensure_sample_datasets()
    return clean_complaints_dataframe(pd.read_csv(COMPLAINTS_PATH))


def load_document_corpus() -> list[dict[str, str]]:
    documents = []
    for path in sorted(DOCS_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        tags = tag_document(path)
        documents.append(
            {
                "id": path.stem,
                "text": text,
                "source": str(path.relative_to(PROJECT_ROOT)),
                "document_type": tags["document_type"],
                "sensitivity_level": tags["sensitivity_level"],
                "DATA_SOURCE": "phase1_sample_document",
            }
        )
    return documents


def recall_rows_to_documents(recalls: pd.DataFrame) -> list[dict[str, str]]:
    documents = []
    for _, row in recalls.iterrows():
        text = (
            f"Recall campaign {row['CAMPNO']} for {row['YEARTXT']} {row['MAKETXT']} {row['MODELTXT']}. "
            f"Manufacturer: {row['MFGNAME']}. Component: {row['COMPONENT']}. "
            f"Summary: {row['SUMMARY']} Consequence: {row['CONSEQUENCE']} Remedy: {row['REMEDY']}"
        )
        documents.append(
            {
                "id": row["document_id"],
                "text": text,
                "source": "data/raw/recalls_sample.csv",
                "document_type": row["document_type"],
                "sensitivity_level": row["sensitivity_level"],
                "DATA_SOURCE": row["DATA_SOURCE"],
            }
        )
    return documents


def complaint_rows_to_documents(complaints: pd.DataFrame) -> list[dict[str, str]]:
    documents = []
    for _, row in complaints.iterrows():
        text = (
            f"Complaint {row['ODINO']} for {row['VEHICLE_MODEL_YEAR']} {row['MAKE']} {row['MODEL']}. "
            f"Manufacturer: {row['MFR_NAME']}. Component: {row['COMPONENT']}. "
            f"Description: {row['CDESCR']}"
        )
        documents.append(
            {
                "id": row["document_id"],
                "text": text,
                "source": "data/raw/complaints_sample.csv",
                "document_type": row["document_type"],
                "sensitivity_level": row["sensitivity_level"],
                "DATA_SOURCE": row["DATA_SOURCE"],
            }
        )
    return documents


def load_all_documents() -> list[dict[str, str]]:
    recalls = load_recalls()
    complaints = load_complaints()
    return recall_rows_to_documents(recalls) + complaint_rows_to_documents(complaints) + load_document_corpus()


def get_data_source_label() -> str:
    recalls = load_recalls()
    complaints = load_complaints()
    sources = set(recalls["DATA_SOURCE"].unique()).union(set(complaints["DATA_SOURCE"].unique()))
    if GENERATED_SOURCE in sources:
        return "Data Source: generated fallback NHTSA-style sample data, not real NHTSA data"
    return "Data Source: user-supplied or real NHTSA-style CSV data"


def load_dashboard_data() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    recalls = load_recalls()
    complaints = load_complaints()
    return recalls, complaints, get_data_source_label()
