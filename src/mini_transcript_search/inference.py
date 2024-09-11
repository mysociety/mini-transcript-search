import os
from contextlib import redirect_stderr
from typing import Optional

import numpy as np
import pandas as pd
import requests
from fastembed import TextEmbedding  # type: ignore
from numpy.typing import NDArray


class Inference:
    """
    Class to handle inference for text embeddings.
    Uses hugging faces free api if not local, but setting local
    will use fastembed's approach
    """

    def __init__(
        self, model_id: str, hf_token: Optional[str] = None, local: bool = False
    ):
        self.model_id: str = model_id
        self.hf_token = hf_token if hf_token else os.environ.get("HF_TOKEN", None)
        self.local = local
        self._model = None
        if self.hf_token is None and self.local is False:
            raise ValueError("Need to set hf_token for remote embedding generation.")

    def query_local(self, texts: list[str]) -> list[NDArray[np.float64]]:
        if self._model is None:
            # suppress download progerss bar
            with open(os.devnull, "w") as fnull:
                with redirect_stderr(fnull):
                    self._model = TextEmbedding(model_name=self.model_id)
        return list(self._model.embed(texts))  # type: ignore

    def query_remote(self, texts: list[str]) -> list[NDArray[np.float64]]:
        api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model_id}"
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        response = requests.post(
            api_url,
            headers=headers,
            json={"inputs": texts, "options": {"wait_for_model": True}},
        )
        return response.json()

    def query_id_and_text(self, id_and_text: dict[str, str]) -> pd.DataFrame:
        id_values = list(id_and_text.keys())
        text_values = list(id_and_text.values())
        embeddings = self.query(text_values)
        if isinstance(embeddings, dict):
            raise ValueError(f"Error in embeddings: {embeddings}")
        return pd.DataFrame(
            {
                "id": id_values,
                "text": text_values,
                "embedding": embeddings,
            }
        )

    def query(self, texts: list[str]):
        if len(texts) == 0:
            return []
        if self.local:
            return self.query_local(texts)
        else:
            return self.query_remote(texts)
