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
    """Cascaded baseline: ASR (Whisper) + Text LLM."""
    
    def load_model(self):
        import whisper
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load Whisper
        self.asr_model = whisper.load_model("large-v3", device=self.config.device)
        
        # Load LLM (e.g., Llama-3, Mistral)
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
        
        # Step 2: Feed transcript + prompt to LLM
        full_prompt = f"Transcript: {transcript}\n\nQuestion: {prompt}\nAnswer:"
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.config.device)
        
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


class SALMONNBaseline(NativeLALMBaseline):
    """SALMONN baseline."""
    
    def load_model(self):
        # SALMONN uses a specific architecture
        # This is a placeholder - actual implementation depends on SALMONN release
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


class QwenAudioBaseline(NativeLALMBaseline):
    """Qwen-Audio baseline."""
    
    def load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path, 
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    def predict(self, audio_path: str, prompt: str) -> Dict[str, Any]:
        # Qwen-Audio uses a specific chat template
        from transformers import AutoProcessor
        
        if not hasattr(self, 'processor'):
            self.processor = AutoProcessor.from_pretrained(
                self.config.model_path, 
                trust_remote_code=True
            )
        
        # Process audio
        audio = self._load_audio(audio_path)
        
        # Format conversation
        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": prompt}
            ]}
        ]
        
        inputs = self.processor.apply_chat_template(
            conversation, 
            return_tensors="pt"
        ).to(self.config.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **self.config.generation_config,
            )
        
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
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
        "whisper_llama3": CascadedBaseline,
        "salmonn": SALMONNBaseline,
        "qwen_audio": QwenAudioBaseline,
        "ltu": LTUBaseline,
    }
    
    if model_name not in baselines:
        raise ValueError(f"Unknown baseline: {model_name}. Available: {list(baselines.keys())}")
    
    return baselines[model_name](config)