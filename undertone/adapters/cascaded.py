"""The cascaded ASR+LLM control.

Not one of the thirteen models -- a validity check. It preempts the obvious
objection ("isn't this just ASR plus a language model?") by running exactly that
system through exactly the same protocol: same items, same option shuffling,
same ladder windows, same letter-logit scoring.

Making it a ``ModelAdapter`` rather than a separate script is the point. If the
control had its own harness, any gap between it and the audio models would be
partly a difference of harness. Here the only thing that differs is that this
one reads words instead of listening.

What its failure means, per category:

    P1/P2  Whisper never wrote the muttered or overlapped words down, so no
           language model behind it could have answered. That is the
           audio-necessity claim, and `harvest.asr.needle_recovery` measures it
           directly rather than inferring it from this model's score.
    P4     ASR normalises a self-repair away -- "twenty twelve milligrams"
           loses the seam that says which one survived.
    C1     Hesitancy is not in the words at all.
"""

from __future__ import annotations

import numpy as np

from ..harvest.asr import DEFAULT_MODEL
from .base import SAMPLE_RATE, ModelAdapter, as_temp_wav, primary_device, register

TEXT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

PROMPT = {
    "en": ("Read the transcript and answer the question about it.\n"
           "Reply with the letter of the correct option and nothing else.\n\n"
           "TRANSCRIPT:\n{transcript}\n\n{question}"),
    "hi": ("प्रतिलेख पढ़ें और उसके बारे में पूछे गए प्रश्न का उत्तर दें।\n"
           "केवल सही विकल्प का अक्षर लिखें।\n\n"
           "प्रतिलेख:\n{transcript}\n\n{question}"),
    "bn": ("প্রতিলিপিটি পড়ুন এবং সেটি সম্পর্কে জিজ্ঞাসিত প্রশ্নের উত্তর দিন।\n"
           "শুধুমাত্র সঠিক বিকল্পের অক্ষরটি লিখুন।\n\n"
           "প্রতিলিপি:\n{transcript}\n\n{question}"),
}


@register
class CascadedWhisperLLM(ModelAdapter):
    key = "cascaded_whisper_llm"
    model_id = f"{DEFAULT_MODEL} + {TEXT_MODEL}"
    # Whisper chunks internally, so the ceiling is the text model's context, not
    # an audio encoder. Nothing in the grid comes close.
    max_audio_s = 7200.0
    primary = "logits"
    is_control = True
    notes = ("Cascaded ASR+LLM validity control, not one of the thirteen. Runs "
             "the identical protocol so any gap is the modality, not the harness.")

    asr_model = DEFAULT_MODEL
    text_model = TEXT_MODEL

    def __init__(self, lang: str = "en") -> None:
        super().__init__()
        self.lang = lang

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from ..harvest.asr import _engine

        _engine(self.asr_model)          # warm the ASR engine now, not mid-sweep
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_model)
        self.processor = self.tokenizer
        self.model = self.place(AutoModelForCausalLM.from_pretrained(
            self.text_model, **self.load_kwargs()))

    def transcribe_window(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> str:
        """Transcribe just the window this ladder condition offers.

        Deliberately not cached against the whole recording: at L1 the cascaded
        system is given a 20-second clip and nothing else, exactly as the audio
        models are. Handing it a full-recording transcript at L1 would make it a
        different, easier condition.
        """
        import os

        from ..harvest.asr import _engine

        path = as_temp_wav(audio, sr)
        try:
            segments, _ = _engine(self.asr_model).transcribe(
                path, language=self.lang, beam_size=1, temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=False,   # VAD drops exactly the quiet needles we study
            )
            return " ".join(s.text.strip() for s in segments if s.text).strip()
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def build_inputs(self, audio: np.ndarray, prompt: str, sr: int = SAMPLE_RATE) -> dict:
        transcript = self.transcribe_window(audio, sr) or "(no speech transcribed)"
        text = PROMPT.get(self.lang, PROMPT["en"]).format(
            transcript=transcript, question=prompt)
        chat = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True, tokenize=False)
        inputs = self.tokenizer(chat, return_tensors="pt")
        self.last_transcript = transcript
        return {k: v.to(primary_device(self.model)) for k, v in inputs.items()}
