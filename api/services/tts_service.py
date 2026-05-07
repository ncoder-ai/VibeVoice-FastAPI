"""Core TTS generation service wrapping VibeVoice model."""

import threading
import torch
import numpy as np
from typing import Iterator, List, Optional, Union
from transformers import set_seed
import logging

logger = logging.getLogger(__name__)

from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.streamer import AudioStreamer

from api.config import Settings


class TTSService:
    """Service for TTS generation using VibeVoice model."""

    def __init__(self, settings: Settings):
        """
        Initialize TTS service.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.model = None
        self.processor = None
        self.device = None
        self.dtype = None
        self._model_loaded = False
        # Serialise concurrent generate() calls — VibeVoice's DPM scheduler
        # keeps step_index as instance state on the shared model object, so
        # two parallel inference passes would corrupt each other. With the
        # `stop_check_fn` cancellation hook below, the first thread releases
        # this lock within ~one outer-loop step (~100-500ms) when cancelled,
        # so the next request doesn't hang.
        self._generate_lock = threading.Lock()
    
    def load_model(self):
        """Load VibeVoice model and processor."""
        if self._model_loaded:
            print("Model already loaded")
            return

        print(f"Loading VibeVoice model from {self.settings.vibevoice_model_path}")

        # Get device and dtype
        self.device = self.settings.get_device()
        self.dtype = self.settings.get_dtype()
        attn_implementation = self.settings.get_attn_implementation()

        print(f"Using device: {self.device}, dtype: {self.dtype}, attention: {attn_implementation}")

        # Load processor
        self.processor = VibeVoiceProcessor.from_pretrained(self.settings.vibevoice_model_path)

        # Detect pre-quantized model (e.g. ncoder-ai/VibeVoice-Large-AWQ). When the
        # checkpoint's config.json carries quantization_config, transformers wires the
        # quant layers automatically inside from_pretrained — nothing else to do.
        unified_quantized = self._detect_unified_quantization(self.settings.vibevoice_model_path)
        if unified_quantized:
            logger.info(
                f"Detected pre-quantized {unified_quantized} checkpoint — "
                f"transformers will load quant layers directly via from_pretrained."
            )

        # Determine if we should load to CPU first for ad-hoc torchao quantization.
        # Skipped for pre-quantized models (their kernels need CUDA at load time).
        load_to_cpu_first = (
            self.settings.vibevoice_quantization
            and not unified_quantized
            and self.device == "cuda"
        )

        if load_to_cpu_first:
            print("Loading model to CPU first for quantization (saves GPU memory)...")
            # Use sdpa for CPU loading since flash_attention_2 requires CUDA
            cpu_attn = "sdpa" if attn_implementation == "flash_attention_2" else attn_implementation
            self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                self.settings.vibevoice_model_path,
                torch_dtype=self.dtype,
                device_map="cpu",
                attn_implementation=cpu_attn,
                low_cpu_mem_usage=True,
            )
            self.model.eval()

            # Apply quantization on CPU
            self._apply_quantization()

            # Now move to CUDA
            print("Moving quantized model to CUDA...")
            self.model = self.model.to("cuda")

            # Log final VRAM usage
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                vram_final = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"Final VRAM usage after moving to GPU: {vram_final:.2f} GB")
        else:
            # Standard loading path (no quantization or non-CUDA device)
            try:
                if self.device == "mps":
                    self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                        self.settings.vibevoice_model_path,
                        torch_dtype=self.dtype,
                        attn_implementation=attn_implementation,
                        device_map=None,
                    )
                    self.model.to("mps")
                elif self.device == "cuda":
                    self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                        self.settings.vibevoice_model_path,
                        torch_dtype=self.dtype,
                        device_map="cuda",
                        attn_implementation=attn_implementation,
                    )
                else:
                    self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                        self.settings.vibevoice_model_path,
                        torch_dtype=self.dtype,
                        device_map="cpu",
                        attn_implementation=attn_implementation,
                    )
            except Exception as e:
                if attn_implementation == 'flash_attention_2':
                    print(f"Flash attention failed: {e}")
                    print("Falling back to SDPA attention")
                    attn_implementation = "sdpa"

                    if self.device == "mps":
                        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                            self.settings.vibevoice_model_path,
                            torch_dtype=self.dtype,
                            attn_implementation=attn_implementation,
                            device_map=None,
                        )
                        self.model.to("mps")
                    elif self.device == "cuda":
                        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                            self.settings.vibevoice_model_path,
                            torch_dtype=self.dtype,
                            device_map="cuda",
                            attn_implementation=attn_implementation,
                        )
                    else:
                        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                            self.settings.vibevoice_model_path,
                            torch_dtype=self.dtype,
                            device_map="cpu",
                            attn_implementation=attn_implementation,
                        )
                else:
                    raise e

            self.model.eval()

        # Apply torch.compile for optimized inference
        if self.settings.torch_compile:
            try:
                compile_mode = self.settings.torch_compile_mode
                self.model = torch.compile(self.model, mode=compile_mode, dynamic=True)
                print(f"Model compiled with torch.compile(mode='{compile_mode}', dynamic=True)")
            except Exception as e:
                print(f"torch.compile() failed: {e}, continuing without compilation")

        # Configure noise scheduler
        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config,
            algorithm_type='sde-dpmsolver++',
            beta_schedule='squaredcos_cap_v2'
        )
        
        # Set inference steps
        self.model.set_ddpm_inference_steps(num_steps=self.settings.vibevoice_inference_steps)
        
        self._model_loaded = True
        print("Model loaded successfully")

    def _apply_quantization(self):
        """Apply runtime torchao quantization to the model based on settings.

        Supported VIBEVOICE_QUANTIZATION values:
          - "int8_torchao"          : INT8 weight-only (slowest matmul on Ampere — dequant overhead)
          - "int8_dynamic_torchao"  : W8A8 dynamic activation+weight quant — uses 3090 INT8 tensor cores
          - "int4_torchao"          : INT4 weight-only (smallest, biggest dequant overhead)

        For AWQ-INT4 use VIBEVOICE_MODEL_PATH=ncoder-ai/VibeVoice-Large-AWQ instead;
        transformers auto-loads its embedded quantization_config via from_pretrained.
        """
        quant_method = self.settings.vibevoice_quantization

        if quant_method == "int8_torchao":
            self._apply_torchao_quant(bits=8, mode="weight_only")
        elif quant_method == "int8_dynamic_torchao":
            self._apply_torchao_quant(bits=8, mode="dynamic")
        elif quant_method == "int4_torchao":
            self._apply_torchao_quant(bits=4, mode="weight_only")
        else:
            logger.warning(f"Unknown quantization method: {quant_method}, skipping quantization")

    @staticmethod
    def _detect_unified_quantization(model_path: str) -> str | None:
        """Read config.json from a local dir or HF hub model and return the
        `quant_method` string if the checkpoint is pre-quantized, else None.

        The unified VibeVoice-Large-AWQ checkpoint embeds quantization_config in
        its config.json so transformers wires AWQ layers automatically — no
        separate graft step.
        """
        import os, json
        try:
            if os.path.isdir(model_path):
                cfg_path = os.path.join(model_path, "config.json")
                if not os.path.isfile(cfg_path):
                    return None
                with open(cfg_path, "r") as f:
                    cfg = json.load(f)
            else:
                # HF hub model id — fetch config.json without downloading weights
                from huggingface_hub import hf_hub_download
                cfg_path = hf_hub_download(model_path, filename="config.json")
                with open(cfg_path, "r") as f:
                    cfg = json.load(f)
            qcfg = cfg.get("quantization_config")
            if qcfg and isinstance(qcfg, dict):
                return qcfg.get("quant_method")
        except Exception as e:
            logger.debug(f"Could not probe quantization_config for {model_path}: {e}")
        return None

    def _apply_torchao_quant(self, bits: int = 8, mode: str = "weight_only"):
        """
        Apply torchao quantization to the language model.

        This selectively quantizes only the LLM (Qwen2) decoder and lm_head,
        keeping audio components (tokenizers, diffusion head, connectors) at full precision.

        Args:
            bits: 8 for INT8 (~40% VRAM reduction) or 4 for INT4 (~60% VRAM reduction, smaller).
            mode: "weight_only" — weights INT8/INT4, activations FP16. Bandwidth win, slow on
                  Ampere because of dequant→FP16 before matmul.
                  "dynamic" — weights INT8 + activations dynamically quantized to INT8 per batch.
                  Uses 3090's INT8 tensor cores (568 TOPS) for the matmul itself, no dequant.
                  Only supported with bits=8.
        """
        try:
            from torchao.quantization import (
                quantize_,
                int8_weight_only,
                int4_weight_only,
                int8_dynamic_activation_int8_weight,
            )
        except ImportError:
            logger.error(
                "torchao not installed. Install with: pip install torchao\n"
                "Falling back to full precision."
            )
            return

        # Select quantization function based on bits + mode
        if mode == "dynamic" and bits == 8:
            quant_fn = int8_dynamic_activation_int8_weight()
            quant_name = "INT8 dynamic activation + weight (W8A8)"
        elif bits == 4:
            quant_fn = int4_weight_only()
            quant_name = "INT4 weight-only"
        else:
            quant_fn = int8_weight_only()
            quant_name = "INT8 weight-only"

        # Check if model is on CUDA (for memory logging)
        model_on_cuda = next(self.model.parameters()).is_cuda

        logger.info(f"Applying torchao {quant_name} weight-only quantization...")
        if model_on_cuda:
            logger.info("Model is on CUDA - quantizing in place")
        else:
            logger.info("Model is on CPU - quantizing before moving to GPU (saves VRAM)")

        # Quantize only the language model (Qwen2 decoder) - this is the largest component
        # The audio components (acoustic_tokenizer, semantic_tokenizer, prediction_head, connectors)
        # are kept at full precision to maintain audio quality
        try:
            logger.info("Quantizing language_model (Qwen2 decoder)...")
            quantize_(self.model.model.language_model, quant_fn)

            logger.info("Quantizing lm_head...")
            quantize_(self.model.lm_head, quant_fn)

        except Exception as e:
            logger.error(f"Failed to quantize model: {e}")
            logger.info("Continuing with full precision model")
            return

        logger.info(f"{quant_name} quantization applied successfully")

        # Force garbage collection
        import gc
        gc.collect()

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded
    
    def generate_speech(
        self,
        text: str,
        voice_samples: List[np.ndarray],
        cfg_scale: float = 1.3,
        inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        stream: bool = False,
        cancel_event: Optional[threading.Event] = None,
    ) -> Union[np.ndarray, Iterator[np.ndarray]]:
        """
        Generate speech from text.

        Args:
            text: Input text (formatted with Speaker labels)
            voice_samples: List of voice sample arrays
            cfg_scale: Classifier-free guidance scale
            inference_steps: Number of diffusion steps (None = use default)
            seed: Random seed for reproducibility
            stream: Whether to return streaming iterator
            cancel_event: Optional threading.Event for cooperative cancellation.
                When set (e.g. by a disconnect-detection task), the model's
                undocumented `stop_check_fn` parameter trips at the next outer
                generation step (~100-500ms) and the lock is released so the
                next request can proceed immediately.

        Returns:
            Generated audio array or iterator of audio chunks
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Set seed if provided
        if seed is not None:
            set_seed(seed)

        # Set inference steps if provided
        if inference_steps is not None:
            self.model.set_ddpm_inference_steps(num_steps=inference_steps)

        # Process inputs
        inputs = self.processor(
            text=[text],
            voice_samples=[voice_samples],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Move to device
        target_device = self.device if self.device in ("cuda", "mps") else "cpu"
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(target_device)

        # Build the stop_check_fn closure if a cancel event was provided.
        # VibeVoice's model.generate() (modular/modeling_vibevoice_inference.py
        # line ~432) calls this at the top of every outer step in the diffusion
        # loop and exits cleanly when it returns True.
        stop_check_fn = (lambda: cancel_event.is_set()) if cancel_event is not None else None

        if stream:
            # Return streaming iterator (lock acquired inside _generate_streaming)
            return self._generate_streaming(inputs, cfg_scale, cancel_event=cancel_event)
        else:
            # Generate all at once. Hold the lock so DPM scheduler state isn't
            # corrupted by a parallel call. The lock releases naturally when
            # generation finishes OR cancel fires.
            try:
                with self._generate_lock, torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=None,
                        cfg_scale=cfg_scale,
                        tokenizer=self.processor.tokenizer,
                        generation_config={'do_sample': False},
                        stop_check_fn=stop_check_fn,
                        return_speech=True,
                        verbose=False,
                        refresh_negative=True,
                        show_progress_bar=False
                    )
            finally:
                # Release CUDA cache regardless of outcome (success or cancel).
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    logger.warning(f"torch.cuda.empty_cache() failed: {e}")

            # If cancellation tripped, return None so the caller knows
            if cancel_event is not None and cancel_event.is_set():
                logger.info("Non-streaming generation cancelled by event")
                return None

            # Get audio output
            if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
                audio = outputs.speech_outputs[0]
                if torch.is_tensor(audio):
                    # Convert bfloat16 to float32 before converting to numpy
                    if audio.dtype == torch.bfloat16:
                        audio = audio.float()
                    audio = audio.cpu().numpy()
                return audio
            else:
                raise RuntimeError("No audio generated")
    
    def _generate_streaming(
        self,
        inputs: dict,
        cfg_scale: float,
        cancel_event: Optional[threading.Event] = None,
    ) -> Iterator[np.ndarray]:
        """
        Generate speech with streaming. Supports cooperative cancellation
        via the optional cancel_event using VibeVoice's `stop_check_fn`
        (checked at every outer-loop step ~ 100-500ms granularity).

        Args:
            inputs: Processed model inputs
            cfg_scale: CFG scale
            cancel_event: When set, the model exits at its next outer-loop
                step, AudioStreamer.end() is called, and the generation
                thread is joined with a timeout.

        Yields:
            Audio chunks as numpy arrays
        """
        # Create audio streamer
        audio_streamer = AudioStreamer(
            batch_size=1,
            stop_signal=None,
            timeout=None
        )

        # No-op cancel_event if none was provided so the rest of the path is uniform
        if cancel_event is None:
            cancel_event = threading.Event()

        # stop_check_fn is what makes cancellation FAST. VibeVoice's
        # model.generate() checks this at the top of every outer step
        # in modular/modeling_vibevoice_inference.py around line 432.
        stop_check_fn = lambda: cancel_event.is_set()

        def generate():
            try:
                # Lock guards shared scheduler state across concurrent stream requests.
                # Released within ~one outer-step on cancel because of stop_check_fn.
                with self._generate_lock, torch.no_grad():
                    self.model.generate(
                        **inputs,
                        max_new_tokens=None,
                        cfg_scale=cfg_scale,
                        tokenizer=self.processor.tokenizer,
                        generation_config={'do_sample': False},
                        audio_streamer=audio_streamer,
                        stop_check_fn=stop_check_fn,
                        return_speech=True,
                        verbose=False,
                        refresh_negative=True,
                        show_progress_bar=False
                    )
            except Exception as e:
                logger.error(f"Generation thread error: {e}")
            finally:
                # Always release the consumer (yields stop_signal in queues)
                # so the for-loop below exits even on cancel/error.
                audio_streamer.end()

        generation_thread = threading.Thread(target=generate, daemon=True)
        generation_thread.start()

        # Yield chunks as they arrive. We poll cancel_event between chunks
        # so we exit promptly even before the streamer end signal arrives.
        try:
            audio_stream = audio_streamer.get_stream(0)
            for chunk in audio_stream:
                if cancel_event.is_set():
                    logger.info("Streaming consumer detected cancel event, breaking")
                    break
                if torch.is_tensor(chunk):
                    # Convert bfloat16 to float32 before converting to numpy
                    if chunk.dtype == torch.bfloat16:
                        chunk = chunk.float()
                    chunk = chunk.cpu().numpy()
                yield chunk
        finally:
            # Whether we exit via normal completion, cancel, or upstream
            # exception, signal the worker and wait for it to release the lock.
            cancel_event.set()
            # 5s is plenty: stop_check_fn fires at next outer step (~100-500ms
            # for our 5-step diffusion config), then the lock releases.
            generation_thread.join(timeout=5.0)
            if generation_thread.is_alive():
                logger.warning(
                    "Generation thread still alive 5s after cancel — "
                    "model.generate() may not be honouring stop_check_fn"
                )
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"torch.cuda.empty_cache() failed: {e}")
    
    def format_script_for_single_speaker(self, text: str, speaker_id: int = 0) -> str:
        """
        Format plain text as single-speaker script.
        
        Args:
            text: Plain text input
            speaker_id: Speaker ID to use
            
        Returns:
            Formatted script
        """
        # Split into sentences/paragraphs
        lines = text.strip().split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                formatted_lines.append(f"Speaker {speaker_id}: {line}")
        
        return '\n'.join(formatted_lines)

