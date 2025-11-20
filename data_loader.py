from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Dict, Any, Optional
import json


@dataclass
class MathSample:
    dataset: str              # "gsm8k" | "math500" | "aime" | "amc23"
    id: str
    question: str
    gold_answer: str
    cot_solution: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


def _read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)



def load_gsm8k_test(data_root: str = "data") -> List[MathSample]:
    path = Path(data_root) / "test" / "gsm8k_test.jsonl"
    samples: List[MathSample] = []

    for idx, record in enumerate(_read_jsonl(path)):
        question = record["question"]
        raw_answer = record["answer"]

        if "####" in raw_answer:
            cot, final = raw_answer.rsplit("####", 1)
            gold = final.strip()
            cot_solution = cot.strip()
        else:
            gold = raw_answer.strip()
            cot_solution = None

        sample = MathSample(
            dataset="gsm8k",
            id=str(record.get("id", f"gsm8k_test_{idx}")),
            question=question,
            gold_answer=gold,
            cot_solution=cot_solution,
            meta={k: v for k, v in record.items() if k not in ["question", "answer"]},
        )
        samples.append(sample)

    return samples


def load_math500_test(data_root: str = "data") -> List[MathSample]:
    path = Path(data_root) / "test" / "math500_test.jsonl"
    samples: List[MathSample] = []

    for record in _read_jsonl(path):
        sample = MathSample(
            dataset="math500",
            id=str(record.get("unique_id", "")),
            question=record["problem"],
            gold_answer=record["answer"].strip(),
            cot_solution=record.get("solution", None),
            meta={
                k: v
                for k, v in record.items()
                if k not in ["problem", "solution", "answer"]
            },
        )
        samples.append(sample)

    return samples


def load_aime_test(data_root: str = "data") -> List[MathSample]:
    path = Path(data_root) / "test" / "aime_test.jsonl"
    samples: List[MathSample] = []

    for record in _read_jsonl(path):
        sample = MathSample(
            dataset="aime",
            id=str(record["id"]),
            question=record["problem"],
            gold_answer=record["answer"].strip(),
            cot_solution=record.get("solution", None),
            meta={
                k: v
                for k, v in record.items()
                if k not in ["problem", "solution", "answer"]
            },
        )
        samples.append(sample)

    return samples


def load_amc23_test(data_root: str = "data") -> List[MathSample]:
    path = Path(data_root) / "test" / "amc23_test.jsonl"
    samples: List[MathSample] = []

    for record in _read_jsonl(path):
        sample = MathSample(
            dataset="amc23",
            id=str(record["id"]),
            question=record["question"],
            gold_answer=record["answer"].strip(),
            cot_solution=None,
            meta={k: v for k, v in record.items() if k not in ["question", "answer"]},
        )
        samples.append(sample)

    return samples

def load_all_test_sets(data_root: str = "data") -> List[MathSample]:
    all_samples: List[MathSample] = []
    all_samples.extend(load_gsm8k_test(data_root))
    all_samples.extend(load_math500_test(data_root))
    all_samples.extend(load_aime_test(data_root))
    all_samples.extend(load_amc23_test(data_root))
    return all_samples
