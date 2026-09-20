#!/usr/bin/env python3
"""Minimal Qwen2.5-Omni planner adapter for BAEA-Adaptive."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

from baea import run_adaptive, run_fixed


class Qwen25OmniPlanner:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-Omni-7B", cache_dir: str | None = None):
        if not torch.cuda.is_available():
            raise RuntimeError("This example requires a CUDA GPU")
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto", cache_dir=cache_dir,
        ).eval()
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_id, cache_dir=cache_dir)

    def generate_text(self, audio_path: Path, prompt: str, *, max_new_tokens: int = 96) -> str:
        conversation = [{"role": "user", "content": [
            {"type": "audio", "audio": str(audio_path)},
            {"type": "text", "text": prompt},
        ]}]
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False,
        )
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = self.processor(
            text=text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True, use_audio_in_video=False,
        ).to(self.model.device).to(self.model.dtype)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, use_audio_in_video=False, return_audio=False,
                max_new_tokens=max_new_tokens, do_sample=False,
            )
        if isinstance(generated, tuple):
            generated = generated[0]
        return self.processor.batch_decode(
            generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def __call__(self, audio_path: Path, prompt: str) -> str:
        return self.generate_text(audio_path, prompt)

    @staticmethod
    def _validate_answer(raw: str) -> dict:
        match = re.search(r"\{.*\}", raw, re.S)
        answer = json.loads(match.group()) if match else {}
        # Normalize one unambiguous misspelled verdict key without changing its value.
        if "rationale" in answer and "verdict" not in answer and len(answer) == 2:
            other = next(key for key in answer if key != "rationale")
            if str(answer[other]).strip().lower() in {"genuine", "spoof"}:
                answer = {"rationale": answer["rationale"], "verdict": answer[other]}
        if set(answer) != {"rationale", "verdict"}:
            raise ValueError(f"invalid fields: {sorted(answer)}")
        if not isinstance(answer["rationale"], str) or not answer["rationale"].strip():
            raise ValueError("rationale must be a non-empty string")
        answer["verdict"] = str(answer["verdict"]).strip().lower()
        if answer["verdict"] not in {"genuine", "spoof"}:
            raise ValueError(f"invalid verdict: {answer['verdict']}")
        return answer

    def finalize(self, audio_path: Path, question: str, evidence: list[dict]) -> tuple[dict, list[str]]:
        prompt = f"""Answer the audio-forensics question using only the measured evidence below.
Question: {question}
Evidence: {json.dumps(evidence, ensure_ascii=False)}
Return one JSON object only with exactly these fields:
{{"rationale":"concise evidence-grounded explanation","verdict":"genuine or spoof"}}
The verdict must be exactly "genuine" or "spoof". Cite concrete measured features and
directions in the rationale. Do not invent measurements or mention hidden labels."""
        attempts = []
        for attempt in range(2):
            raw = self.generate_text(audio_path, prompt, max_new_tokens=256)
            attempts.append(raw)
            try:
                return self._validate_answer(raw), attempts
            except (ValueError, json.JSONDecodeError) as exc:
                if attempt:
                    raise ValueError(f"Final answer failed schema validation twice: {exc}") from exc
                prompt = f"""Repair only the JSON schema of this answer: {raw}
Return JSON only with exactly two fields, rationale and verdict. Preserve the intended
reasoning and decision. verdict must be exactly genuine or spoof. No other fields."""
        raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--policy", choices=("fixed", "adaptive"), required=True)
    parser.add_argument("--objective", choices=("classification", "rationale"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Omni-7B")
    parser.add_argument("--cache-dir")
    parser.add_argument("--generate-final", action="store_true")
    parser.add_argument(
        "--question",
        default="Is this recording genuine or spoofed? Explain using acoustic evidence.",
    )
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    alm = None
    if args.policy == "fixed":
        result = run_fixed(args.audio, reference, args.objective)
    else:
        alm = Qwen25OmniPlanner(args.model, args.cache_dir)
        result = run_adaptive(
            args.audio, reference, args.objective,
            alm,
        )
    if args.generate_final:
        alm = alm or Qwen25OmniPlanner(args.model, args.cache_dir)
        answer, attempts = alm.finalize(
            args.audio, args.question, result["selected_evidence"],
        )
        result.update(answer)
        result["final_answer_attempts"] = attempts
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
