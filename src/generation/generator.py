"""Generate grounded answers with a local Qwen model."""

from typing import Any

from src.generation.prompts import build_prompt
from src.schemas import SourceChunk

MODEL_NAME = "Qwen/Qwen3-0.6B"


class Generator:
    """Load Qwen and generate answers from retrieved sources."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        """Store the model name without loading model weights yet."""
        self.model_name = model_name
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.torch: Any | None = None
        self.device = "cpu"

    def load_model(self) -> bool:
        """Load the tokenizer and model weights."""
        if self.model is not None and self.tokenizer is not None:
            return True

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer: Any = AutoTokenizer.from_pretrained(self.model_name)
            model: Any = AutoModelForCausalLM.from_pretrained(self.model_name)
            model.to(self.device)
            model.eval()
            self.tokenizer = tokenizer
            self.model = model
            self.torch = torch
        except Exception:
            self.model = None
            self.tokenizer = None
            self.torch = None
            return False
        return True

    def answer(
        self,
        question: str,
        sources: list[SourceChunk],
        max_new_tokens: int = 256,
    ) -> str:
        """Generate an answer grounded in retrieved source chunks."""
        if not question.strip():
            return "Please provide a question."
        if not sources:
            return "I could not answer because no sources were retrieved."
        if max_new_tokens < 1 or not self.load_model():
            return "The local model could not be loaded."

        tokenizer = self.tokenizer
        model = self.model
        torch = self.torch
        if tokenizer is None or model is None or torch is None:
            return "The local model could not be loaded."

        try:
            prompt = build_prompt(question, sources)
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            answer_tokens = output[0][inputs.input_ids.shape[1]:]
            generated = tokenizer.decode(
                answer_tokens,
                skip_special_tokens=True,
            )
            return str(generated).strip()
        except Exception:
            return "The local model could not generate an answer."
