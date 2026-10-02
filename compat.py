"""Access the shared MiPN domain and SAOP implementation without copying data."""

import os
import sys
from pathlib import Path


MIPN_ROOT = Path(os.environ.get("MIPN_ROOT", Path(__file__).resolve().parent.parent / "MiPN-based")).resolve()
if not (MIPN_ROOT / "domain").is_dir():
    raise FileNotFoundError(f"MiPN domain directory not found: {MIPN_ROOT / 'domain'}")
if str(MIPN_ROOT) not in sys.path:
    sys.path.append(str(MIPN_ROOT))

from envs.domain_loader import load_genius_domain  # noqa: E402
from sao.my_sao import MySAOMechanism  # noqa: E402
from sao.my_negotiators import (  # noqa: E402
    AgentGG, AgentK, Atlas3, AverageTitForTatNegotiator,
    HardHeaded, TimeBasedNegotiator,
)


SCRIPTED = ("Boulware", "Linear", "Conceder", "Atlas3", "TitForTat1", "TitForTat2", "AgentK", "HardHeaded", "AgentGG")

KNOWN_DOMAINS = (
    "Laptop", "ItexvsCypress", "IS_BT_Acquisition", "Grocery",
    "thompson", "Car", "EnergySmall_A",
)

UNKNOWN_DOMAINS = ("Coffee", "Camera", "Lunch", "SmartPhone", "Kitchen")


def scripted_opponent(name, slot, noise=False):
    label = f"{name}{slot}"
    if name in ("Boulware", "Linear", "Conceder"):
        return TimeBasedNegotiator(
            name=label,
            aspiration_type={"Boulware": 10.0, "Linear": 1.0, "Conceder": 0.2}[name],
            add_noise=noise,
        )
    if name in ("TitForTat1", "TitForTat2"):
        return AverageTitForTatNegotiator(name=label, gamma=int(name[-1]), add_noise=noise)
    if name in ("AgentK", "HardHeaded", "AgentGG", "Atlas3"):
        return {"AgentK": AgentK, "HardHeaded": HardHeaded, "AgentGG": AgentGG, "Atlas3": Atlas3}[name](name=label, add_noise=noise)
    raise ValueError(f"Unknown scripted opponent: {name}")
