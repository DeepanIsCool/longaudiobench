"""
Baseline runner for LongAudioBench.

Supports cascaded baselines (Whisper + LLM) and native LALMs.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time
import json
import torch
import numpy as np


@dataclass
class ModelConfig:
    """Configuration for a model baseline."""
    name: str
    model_type: str  # "cascaded" or "native"
    model_path: str
    device: str = "cuda"
    max_audio_length: int = 3600  # seconds
    sample_rate: int = 16000
    generation_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.generation_config is None:
            self.generation_config = {
                "max_new_tokens": 512,
                "temperature": 0.1,
                "top_p": 0.9,
                "do_sample": True,
            }


class BaseBaseline(ABC):
    """Abstract base class for model baselines."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.processor = None
    
    @abstractmethod
    def load_model(self):
        """Load the model and processor."""
        pass
    
    @abstractmethod
    def predict(self, audio_path: str, prompt: str) -> Dict[str, Any]:
        """Run inference on audio with prompt."""
        pass
    
    def _load_audio(self, audio_path: str) -> np.ndarray:
        """Load and resample audio."""
        import librosa
        audio, sr = librosa.load(audio_path, sr=self.config.sample_rate)
        
        # Truncate if too long
        max_samples = self.config.max_audio_length * self.config.sample_rate
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        return audio
    
    def run_evaluation(
        self, 
        instances: List[Any],  # List of TaskInstance
        task_evaluator: callable
    ) -> Dict[str, Any]:
        """Run full evaluation on a set of instances."""
        predictions = []
        latencies = []
        
        for instance in instances:
            start_time = time.time()
            response = self.predict(instance.audio_path, instance.prompt)
            latency = (time.time() - start_time) * 1000  # ms
            
            latencies.append(latency)
            predictions.append({
                "instance_id": instance.instance_id,
                "response": response,
                "latency_ms": latency,
            })
        
        # Evaluate
        ground_truths = [inst.ground_truth for inst in instances]
        eval_results = task_evaluator(predictions, ground_truths)
        
        return {
            "model": self.config.name,
            "predictions": predictions,
            "latencies": latencies,
            "metrics": eval_results,
            "avg_latency_ms": float(np.mean(latencies)),
        }


class CascadedBaseline(BaseBaseline):
    """Cascaded baseline: ASR (Whisper) + Text LLM.

    Uses whisper "small" rather than "large-v3": on a 5+ minute background
    clip per instance, large-v3's transcription time dominates total
    runtime (this was most of an observed ~22min/task on a single T4) for
    a pilot where ASR only needs to be good enough to feed the LLM.
    """

    WHISPER_SIZE = "small"

    def load_model(self):
        import whisper
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.asr_model = whisper.load_model(self.WHISPER_SIZE, device=self.config.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def predict(self, audio_path: str, prompt: str) -> Dict[str, Any]:
        # Step 1: Transcribe with Whisper
        result = self.asr_model.transcribe(audio_path, language="en")
        transcript = result["text"]

        # Step 2: Feed transcript + prompt to LLM via its chat template -
        # an instruct model given a raw completion prompt (no special
        # tokens marking user/assistant turns) follows instructions and
        # format requests markedly worse than through its chat template.
        messages = [{"role": "user", "content": f"Transcript: {transcript}\n\nQuestion: {prompt}"}]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.config.device)

        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                **self.config.generation_config,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        return {
            "transcript": transcript,
            "response": response.strip(),
        }


class NativeLALMBaseline(BaseBaseline):
    """Native Audio-LLM baseline (SALMONN, Qwen-Audio, LTU, etc.)."""
    
    def load_model(self):
        # This will be implemented per specific model
        # Placeholder for model-specific loading
        raise NotImplementedError("Subclass must implement load_model")
    
    def predict(self, audio_path: str, prompt: str) -> Dict[str, Any]:
        raise NotImplementedError("Subclass must implement predict")


class QwenAudioBaseline(NativeLALMBaseline):
    """Qwen2-Audio baseline (transformers>=4.45 native support)."""

    def load_model(self):
        from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.config.model_path)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def predict(self, audio_path: str, prompt: str) -> Dict[str, Any]:
        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": prompt},
            ]}
        ]
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audio = self._load_audio(audio_path)

        inputs = self.processor(
            text=text,
            audios=[audio],
            sampling_rate=self.config.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        # Move every tensor, not just input_ids: Qwen2-Audio-7B is ~16.8GB
        # in fp16, bigger than a single T4, so device_map="auto" is likely
        # to split it across both Kaggle T4s - leaving the audio feature
        # tensors on CPU (or the wrong GPU) risks a device-mismatch error
        # that the official single-GPU example doesn't need to worry about.
        inputs = {k: v.to(self.config.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            generate_ids = self.model.generate(**inputs, **self.config.generation_config)
        generate_ids = generate_ids[:, inputs["input_ids"].size(1):]

        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return {"response": response}


class LTUBaseline(NativeLALMBaseline):
    """LTU (Listen, Think, Understand) baseline."""
    
    def load_model(self):
        from transformers import AutoModel, AutoProcessor
        
        self.processor = AutoProcessor.from_pretrained(self.config.model_path)
        self.model = AutoModel.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    
    def predict(self, audio_path: str, prompt: str) -> Dict[str, Any]:
        audio = self._load_audio(audio_path)
        
        inputs = self.processor(
            audio=audio,
            text=prompt,
            sampling_rate=self.config.sample_rate,
            return_tensors="pt",
        ).to(self.config.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **self.config.generation_config,
            )
        
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        return {"response": response}


def get_baseline(model_name: str, config: ModelConfig) -> BaseBaseline:
    """Factory function to get baseline by name."""
    baselines = {
        "cascaded": CascadedBaseline,
        "qwen_audio": QwenAudioBaseline,
        "ltu": LTUBaseline,
    }
    
    if model_name not in baselines:
        raise ValueError(f"Unknown baseline: {model_name}. Available: {list(baselines.keys())}")
    
    return baselines[model_name](config)