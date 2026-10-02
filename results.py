"""Existing seven-column TSV contract, with pool/style metadata appended."""

import csv
from pathlib import Path

from opponent_pool.pfsp import negotiation_score


FIELDS = ("my_util", "opp_util1", "opp_util2", "social", "nash", "agreement", "step",
          "style", "opponent1_id", "opponent2_id", "opponent1_source", "opponent2_source",
          "negotiation_score", "seed", "domain")


def result_path(model_dir, agents, issue, deterministic=False, noise=False):
    letter = lambda value: "T" if value else "F"
    return (Path(model_dir) / "csv" / f"{agents[0]}-{agents[1]}" / issue /
            f"det={deterministic}_noise={noise}" /
            f"{issue}-{agents[0]}-{agents[1]}-d{letter(deterministic)}-n{letter(noise)}.tsv")


def write_results(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def row_from_info(info, style, opponents, seed, *, domain="", length_weight=-0.005,
                  welfare_weight=0.1):
    agreement = info["agreement"]
    return {"my_util": info["my_util"], "opp_util1": info["opp_util1"],
            "opp_util2": info["opp_util2"], "social": info["social"], "nash": info["nash"],
            "agreement": agreement, "step": info["step"], "style": style,
            "opponent1_id": opponents[0].id, "opponent2_id": opponents[1].id,
            "opponent1_source": opponents[0].kind, "opponent2_source": opponents[1].kind,
            "negotiation_score": negotiation_score(agreement is not None, info["my_util"],
                                                   info["step"], info["social"],
                                                   length_weight=length_weight,
                                                   welfare_weight=welfare_weight), "seed": seed,
            "domain": domain}
