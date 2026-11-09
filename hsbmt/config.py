"""Project paths, language codes and translation directions."""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
LOGS_DIR = ROOT / "logs"

NLLB_BASE_MODEL = "facebook/nllb-200-distilled-600M"
NLLB_HSB_INIT = MODELS_DIR / "nllb-600M-hsb-init"
QWEN_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
XCOMET_MODEL = "Unbabel/XCOMET-XL"

# NLLB language tags. hsb_Latn is not in NLLB-200; models/nllb.py adds it.
NLLB_LANG = {"de": "deu_Latn", "hsb": "hsb_Latn", "cs": "ces_Latn", "en": "eng_Latn", "pl": "pol_Latn"}
LANG_NAME = {"de": "German", "hsb": "Upper Sorbian"}


@dataclass(frozen=True)
class Direction:
    """A translation direction such as de-hsb. File suffixes equal the short codes."""

    src: str
    tgt: str

    @classmethod
    def parse(cls, value: str) -> "Direction":
        src, tgt = value.split("-")
        if {src, tgt} != {"de", "hsb"}:
            raise ValueError(f"Unsupported direction: {value}")
        return cls(src, tgt)

    @property
    def name(self) -> str:
        return f"{self.src}-{self.tgt}"

    @property
    def nllb_src(self) -> str:
        return NLLB_LANG[self.src]

    @property
    def nllb_tgt(self) -> str:
        return NLLB_LANG[self.tgt]

    @property
    def src_name(self) -> str:
        return LANG_NAME[self.src]

    @property
    def tgt_name(self) -> str:
        return LANG_NAME[self.tgt]

    def files(self, split_dir: Path, split: str) -> tuple[Path, Path]:
        """Return (source, target) paths of `split` inside `split_dir`."""
        return split_dir / f"{split}.{self.src}", split_dir / f"{split}.{self.tgt}"


DIRECTIONS = ["de-hsb", "hsb-de"]
