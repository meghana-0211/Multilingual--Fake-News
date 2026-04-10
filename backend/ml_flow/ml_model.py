"""
backend/models/ml_model.py

Adapter layer between the trained checkpoint (ml_flow/train.py) and the
Flask backend.

Key classes
-----------
FakeNewsDetector       – loads the checkpoint once; used by routes for inference
DetectorAsModelWrapper – wraps FakeNewsDetector so it satisfies the interface
                         that ExplainabilityEngine expects (predict_single /
                         predict_batch / get_attention_weights), without loading
                         the model a second time.
LSTMBERTEnsemble       – alias kept for backward-compat with original app.py
"""

import sys
import os
import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure ml_flow is importable regardless of cwd
# ---------------------------------------------------------------------------
_HERE    = os.path.dirname(os.path.abspath(__file__))   # backend/models/
_BACKEND = os.path.dirname(_HERE)                        # backend/
_ML_FLOW = os.path.join(_BACKEND, "ml_flow")
for _p in (_BACKEND, _ML_FLOW):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train import IndicBERTBiLSTMEnsemble           # noqa: E402
from transformers import AutoTokenizer              # noqa: E402

LSTMBERTEnsemble = IndicBERTBiLSTMEnsemble          # backward-compat alias


# ---------------------------------------------------------------------------
# FakeNewsDetector
# ---------------------------------------------------------------------------

class FakeNewsDetector:
    """
    Loads checkpoint once.  Primary inference object stored on app.config.
    """

    LABEL_MAP = {0: "real", 1: "fake"}

    def __init__(self, model: IndicBERTBiLSTMEnsemble,
                 tokenizer, device: torch.device):
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = device

    @classmethod
    def load(cls, checkpoint_path: str,
             model_name: str = "ai4bharat/indic-bert") -> "FakeNewsDetector":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading FakeNewsDetector from {checkpoint_path} on {device}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model     = IndicBERTBiLSTMEnsemble()
        ckpt      = torch.load(checkpoint_path, map_location=device)

        missing, unexpected = model.load_state_dict(
            ckpt["model_state_dict"], strict=False
        )
        if missing:
            logger.warning(f"Checkpoint missing keys: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected}")

        model.to(device).eval()
        logger.info("FakeNewsDetector ready.")
        return cls(model, tokenizer, device)

    def _encode(self, texts: list, max_length: int = 256) -> dict:
        return self.tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def predict(self, text: str, max_length: int = 256) -> dict:
        """
        Single-text inference.

        Returns
        -------
        {
            "prediction":    "fake" | "real",
            "label":         "Fake" | "Real",
            "confidence":    float,
            "probabilities": {"real": float, "fake": float},
        }
        """
        enc            = self._encode([text], max_length)
        input_ids      = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            out   = self.model(input_ids, attention_mask)
            probs = torch.softmax(out["logits"][0], dim=0)
            idx   = int(probs.argmax())

        return {
            "prediction":    self.LABEL_MAP[idx],
            "label":         self.LABEL_MAP[idx].capitalize(),
            "confidence":    round(float(probs[idx]), 4),
            "probabilities": {
                "real": round(float(probs[0]), 4),
                "fake": round(float(probs[1]), 4),
            },
        }

    def predict_batch_proba(self, texts: list,
                            max_length: int = 256) -> np.ndarray:
        """Batch softmax probs, shape (N, 2).  Used by LIME via the shim."""
        if not texts:
            return np.array([])

        enc            = self._encode(texts, max_length)
        input_ids      = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            out   = self.model(input_ids, attention_mask)
            probs = torch.softmax(out["logits"], dim=1)

        return probs.cpu().numpy()


# ---------------------------------------------------------------------------
# DetectorAsModelWrapper  (shim so ExplainabilityEngine works without a
#                           second checkpoint load)
# ---------------------------------------------------------------------------

class DetectorAsModelWrapper:
    """
    Wraps FakeNewsDetector to satisfy the interface ExplainabilityEngine
    (ml_flow/explainability.py) expects:

        .predict_single(text, return_all_outputs=False)  → dict
        .predict_batch(texts)                            → np.ndarray (N, 2)
        .get_attention_weights(text)                     → dict
    """

    def __init__(self, detector: FakeNewsDetector):
        self._det      = detector
        # Expose attributes ExplainabilityEngine may access directly
        self.tokenizer = detector.tokenizer
        self.device    = detector.device
        self.model     = detector.model

    # -----------------------------------------------------------------------

    def predict_single(self, text: str,
                       return_all_outputs: bool = False) -> dict:
        """
        Matches ModelWrapper.predict_single() signature.
        ExplainabilityEngine.explain_lime() calls this for the base prediction.
        """
        raw = self._det.predict(text)

        # ExplainabilityEngine checks result['prediction'] as an int (0/1)
        result = {
            "prediction":    1 if raw["prediction"] == "fake" else 0,
            "label":         raw["label"],
            "confidence":    raw["confidence"],
            "probabilities": raw["probabilities"],
        }

        if return_all_outputs:
            attn_data = self.get_attention_weights(text)
            result["attention_weights"] = torch.tensor(
                attn_data["attention_weights"]
            )
            result["bert_logits"] = None
            result["lstm_logits"] = None

        return result

    def predict_batch(self, texts: list) -> np.ndarray:
        """Matches ModelWrapper.predict_batch() — called by LIME."""
        return self._det.predict_batch_proba(texts)

    def get_attention_weights(self, text: str) -> dict:
        """
        Matches ModelWrapper.get_attention_weights().

        Returns
        -------
        {
            "tokens":            list[str],
            "attention_weights": np.ndarray  shape (seq_len, seq_len),
            "prediction":        int  0=real / 1=fake
        }
        """
        enc            = self._det._encode([text], max_length=256)
        input_ids      = enc["input_ids"].to(self._det.device)
        attention_mask = enc["attention_mask"].to(self._det.device)

        with torch.no_grad():
            out = self._det.model(input_ids, attention_mask)

        # BiLSTMWithAttention returns shape (1, seq_len, seq_len)
        attn       = out["attention_weights"][0].cpu().numpy()
        actual_len = int(attention_mask[0].sum().item())
        tokens     = self._det.tokenizer.convert_ids_to_tokens(
            input_ids[0].cpu().tolist()
        )

        probs = torch.softmax(out["logits"][0], dim=0)
        pred  = int(probs.argmax())

        return {
            "tokens":            tokens[:actual_len],
            "attention_weights": attn[:actual_len, :actual_len],
            "prediction":        pred,
        }