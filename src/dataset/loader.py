"""Dataset loader and manager for LIMA, Alpaca, and CodeAlpaca."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional
import httpx

logger = logging.getLogger(__name__)

DatasetSource = Literal[
    "lima", "alpaca", "codealpaca", "claude_reasoning",
    "genius_reasoning", "genius_code", "genius_general", "custom"
]

RAW_DATASET_URLS: Dict[str, str] = {
    # Alpaca 52k clean instruction dataset
    "alpaca": "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json",
    # CodeAlpaca 20k programming instruction dataset
    "codealpaca": "https://raw.githubusercontent.com/sahil280114/codealpaca/master/data/code_alpaca_20k.json",
    # LIMA - Less is More for Alignment (GAIR/lima)
    "lima": "https://huggingface.co/datasets/GAIR/lima/resolve/main/train.jsonl",
}


@dataclass
class DatasetItem:
    """Standardized representation of an instruction/conversation sample."""
    id: str
    source: DatasetSource
    category: str
    instruction: str
    input: str
    output: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetManager:
    """Manages downloading, caching, and loading curated datasets."""

    def __init__(self, storage_dir: str | Path = "data/datasets") -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def get_local_path(self, source: DatasetSource) -> Path:
        """Returns the local path for a dataset."""
        if source in ("claude_reasoning", "genius_reasoning", "reasoning"):
            for candidate in [
                "genius_reasoning_exemplars.json",
                "genius_reasoning_exemplars.jsonl",
                "genius_claude_reasoning.json",
                "genius_claude_reasoning.jsonl",
            ]:
                p = self.storage_dir / candidate
                if p.exists():
                    return p
            return self.storage_dir / "genius_reasoning_exemplars.json"

        if source in ("codealpaca", "genius_code", "coding"):
            for candidate in ["genius_code_instruct.json", "codealpaca.json"]:
                p = self.storage_dir / candidate
                if p.exists():
                    return p
            return self.storage_dir / "genius_code_instruct.json"

        if source in ("alpaca", "genius_general", "general"):
            for candidate in ["genius_general_instruct.json", "alpaca.json"]:
                p = self.storage_dir / candidate
                if p.exists():
                    return p
            return self.storage_dir / "genius_general_instruct.json"

        ext = "jsonl" if source == "lima" else "json"
        return self.storage_dir / f"{source}.{ext}"

    def is_cached(self, source: DatasetSource) -> bool:
        """Checks if dataset exists locally and is not empty."""
        p = self.get_local_path(source)
        return p.exists() and p.stat().st_size > 100

    async def download_dataset(
        self,
        source: DatasetSource,
        force: bool = False,
    ) -> Path:
        """Downloads dataset from primary or mirror repository."""
        target_path = self.get_local_path(source)
        if target_path.exists() and not force and target_path.stat().st_size > 1024:
            logger.info(f"Dataset '{source}' already cached at {target_path}")
            return target_path

        url = RAW_DATASET_URLS.get(source)
        if not url:
            raise ValueError(f"No remote URL configured for dataset source: {source}")

        logger.info(f"Downloading dataset '{source}' from {url}...")
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            target_path.write_bytes(resp.content)

        logger.info(f"Successfully saved {source} dataset ({target_path.stat().st_size} bytes)")
        return target_path

    def load_dataset(
        self,
        source: DatasetSource,
        limit: Optional[int] = None,
    ) -> List[DatasetItem]:
        """Loads and normalizes items from local cache or raises error if not downloaded."""
        path = self.get_local_path(source)
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset '{source}' not found locally at {path}. Run download_dataset first."
            )

        items: List[DatasetItem] = []

        if source in ("lima", "claude_reasoning", "genius_reasoning", "custom") and path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if limit and len(items) >= limit:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        instruction = record.get("instruction") or record.get("prompt") or ""
                        output = record.get("output") or record.get("response") or ""
                        inp = record.get("input", "")
                        category = record.get("category", "conversation" if source == "lima" else "reasoning")

                        # Fallback for LIMA conversation format
                        if not instruction and "conversations" in record:
                            convs = record["conversations"]
                            if len(convs) >= 2:
                                instruction, output = convs[0].strip(), convs[1].strip()

                        if instruction:
                            items.append(
                                DatasetItem(
                                    id=f"{source}_{idx}",
                                    source=source,
                                    category=category,
                                    instruction=instruction,
                                    input=inp,
                                    output=output,
                                    metadata=record.get("metadata", {}),
                                )
                            )
                    except json.JSONDecodeError:
                        continue

        elif source in ("alpaca", "codealpaca", "claude_reasoning", "genius_reasoning", "genius_code", "genius_general", "custom"):
            if source in ("codealpaca", "genius_code"):
                default_category = "coding"
            elif source in ("claude_reasoning", "genius_reasoning"):
                default_category = "reasoning"
            else:
                default_category = "general_instruction"

            with open(path, "r", encoding="utf-8") as f:
                records = json.load(f)
                if isinstance(records, dict):
                    records = [records]
                for idx, record in enumerate(records):
                    if limit and len(items) >= limit:
                        break
                    items.append(
                        DatasetItem(
                            id=f"{source}_{idx}",
                            source=source,
                            category=record.get("category", default_category),
                            instruction=record.get("instruction", "").strip(),
                            input=record.get("input", "").strip(),
                            output=record.get("output", "").strip(),
                            metadata=record.get("metadata", {}),
                        )
                    )

        return items

    def load_all_sampled(
        self,
        alpaca_limit: int = 1000,
        codealpaca_limit: int = 1000,
        lima_limit: Optional[int] = None,
        claude_reasoning_limit: Optional[int] = None,
    ) -> List[DatasetItem]:
        """Loads a balanced sample of all available datasets."""
        combined: List[DatasetItem] = []
        for src, lim in [
            ("genius_reasoning", claude_reasoning_limit),
            ("genius_code", codealpaca_limit),
            ("genius_general", alpaca_limit),
            ("claude_reasoning", claude_reasoning_limit),
            ("lima", lima_limit),
            ("alpaca", alpaca_limit),
            ("codealpaca", codealpaca_limit),
        ]:
            if self.is_cached(src):  # type: ignore
                combined.extend(self.load_dataset(src, limit=lim))  # type: ignore
        return combined
