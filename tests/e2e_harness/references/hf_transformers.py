"""HuggingFace Transformers reference backend — gold-standard L1 oracle.

Runs HF model inference in a subprocess for GPU memory isolation and returns
per-step logits + generated text for comparison against TRT outputs.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

from .. import save_full_stderr, _case_artifact_dir
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec

logger = logging.getLogger(__name__)


class HfTransformersReference:
    """Run HuggingFace Transformers inference as the reference oracle."""

    @property
    def backend_name(self) -> str:
        return "hf_transformers"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        if stage.name == "full_generation":
            return self._run_full_generation(case, stage, ctx)
        if stage.name == "full_inference":
            return self._run_full_inference(case, stage, ctx)
        if stage.name == "vision_encode":
            # Vision encode is TRT-side only; reference skips this stage
            return StageOutput(
                stage_name=stage.name,
                data={"skipped": True},
                metadata={"reason": "vision_encode handled by TRT runner only"},
            )
        raise ValueError(f"Unknown stage for hf_transformers: {stage.name!r}")

    def _run_full_generation(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run HF model inference in a subprocess, collecting per-step logits.

        Dispatches to task-specific methods for non-standard tasks:
        - text_to_audio -> _run_text_to_audio_ref()
        - vision_language_generation -> _run_vl_full_generation()
        - speech_to_text -> _run_speech_to_text_ref() (via full_inference)
        """
        task = case.task_strategy
        if task == "text_to_audio":
            return self._run_text_to_audio_ref(case, stage, ctx)
        if task == "vision_language_generation":
            return self._run_vl_full_generation(case, stage, ctx)
        if task == "speech_to_text":
            return self._run_speech_to_text_ref(case, stage, ctx)

        artifacts_dir = ctx.artifacts_dir or tempfile.gettempdir()
        model_dir = _case_artifact_dir(artifacts_dir, case.name) if ctx.artifacts_dir else artifacts_dir
        logits_path = str(Path(model_dir) / "hf_logits.npy")
        text_path = str(Path(model_dir) / "hf_text.txt")

        prompt = case.inputs.get("prompt", "The capital of France is")
        max_new_tokens = case.inputs.get("max_new_tokens", 30)
        trust_remote_code = case.metadata.get("trust_remote_code", False)
        hf_id = case.hf_id

        script = textwrap.dedent(f"""\
            import sys, numpy as np, torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            hf_id = {hf_id!r}
            prompt = {prompt!r}
            max_new_tokens = {max_new_tokens}
            trust_remote_code = {trust_remote_code!r}
            logits_path = {logits_path!r}
            text_path = {text_path!r}

            tokenizer = AutoTokenizer.from_pretrained(
                hf_id, trust_remote_code=trust_remote_code)
            input_ids = tokenizer.encode(prompt)

            load_kwargs = {{
                "trust_remote_code": trust_remote_code,
                "dtype": torch.float32,
            }}
            try:
                model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)
            except TypeError:
                # Backward compatibility for older transformers versions.
                load_kwargs.pop("dtype", None)
                load_kwargs["torch_dtype"] = torch.float32
                model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)
            model.eval()

            ids_tensor = torch.tensor([input_ids], dtype=torch.long)
            all_logits = []

            with torch.no_grad():
                # Prefill
                outputs = model(ids_tensor)
                prefill_logits = outputs.logits[0].numpy()
                for i in range(len(input_ids)):
                    all_logits.append(prefill_logits[i])

                # Autoregressive generation
                gen_ids = list(input_ids)
                generated_token_ids = []
                for _ in range(max_new_tokens):
                    next_token = int(np.argmax(all_logits[-1]))
                    generated_token_ids.append(next_token)
                    gen_ids.append(next_token)
                    ids_tensor = torch.tensor([gen_ids], dtype=torch.long)
                    outputs = model(ids_tensor)
                    all_logits.append(outputs.logits[0, -1].numpy())

            # Decode only tokens actually generated in the autoregressive loop.
            text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
            with open(text_path, "w") as f:
                f.write(text)

            # Pad and save logits
            max_len = max(l.shape[0] for l in all_logits)
            padded = np.zeros((len(all_logits), max_len), dtype=np.float32)
            for i, l in enumerate(all_logits):
                padded[i, :l.shape[0]] = l
            np.save(logits_path, padded)

            print(f"OK steps={{len(all_logits)}} vocab={{max_len}}")
        """)

        python = ctx.hf_python or sys.executable
        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path
        logger.info("HF reference: running %s", case.name)
        t0 = time.monotonic()
        try:
            result = subprocess.run(
                [python, "-c", script],
                capture_output=True, text=True, timeout=1800,
                env=env,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - t0
            raise RuntimeError(
                f"HF reference timed out for {case.name} after {elapsed:.0f}s"
            )
        except Exception as e:
            raise RuntimeError(f"HF reference failed for {case.name}: {e}") from e
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            truncated, log_path = save_full_stderr(
                result.stderr, ctx.artifacts_dir or "",
                "hf_full_generation", case.name)
            msg = f"HF reference failed for {case.name} (rc={result.returncode}):\n{truncated}"
            if log_path:
                msg += f" (full stderr: {log_path})"
            raise RuntimeError(msg)

        # Read generated text
        text = ""
        if Path(text_path).is_file():
            text = Path(text_path).read_text(encoding="utf-8")

        data = {}
        if Path(logits_path).is_file():
            data["logits_path"] = logits_path

        meta = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "trust_remote_code": trust_remote_code,
        }

        return StageOutput(
            stage_name=stage.name,
            data=data,
            text=text,
            logits=logits_path if Path(logits_path).is_file() else None,
            timing_s=elapsed,
            metadata=meta,
        )


    def _run_full_inference(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run HF model forward pass for non-generative tasks.

        Dispatches based on task_strategy to the appropriate HF Auto class.
        """
        task = case.task_strategy
        if task == "encoder_only_nlp":
            return self._run_encoder_only(case, stage, ctx)
        if task == "segmentation":
            return self._run_segmentation_ref(case, stage, ctx)
        if task == "prompted_segmentation":
            return self._run_prompted_segmentation_ref(case, stage, ctx)
        if task == "embedding":
            return self._run_embedding_ref(case, stage, ctx)
        if task == "reranking":
            return self._run_reranking_ref(case, stage, ctx)
        if task == "speech_to_text":
            return self._run_speech_to_text_ref(case, stage, ctx)
        if task == "object_detection":
            return self._run_object_detection_ref(case, stage, ctx)
        raise ValueError(
            f"full_inference not implemented for task_strategy={task!r}")

    def _run_encoder_only(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run HF encoder-only model (e.g. BERT) and return CLS embedding."""
        artifacts_dir = ctx.artifacts_dir or tempfile.gettempdir()
        model_dir = _case_artifact_dir(artifacts_dir, case.name) if ctx.artifacts_dir else artifacts_dir
        output_path = str(Path(model_dir) / "hf_encoder.json")

        prompt = case.inputs.get("prompt", "Hello world")
        trust_remote_code = case.metadata.get("trust_remote_code", False)
        hf_id = case.hf_id

        script = textwrap.dedent(f"""\
            import json, torch, numpy as np
            from transformers import AutoModel, AutoTokenizer

            hf_id = {hf_id!r}
            prompt = {prompt!r}
            trust_remote_code = {trust_remote_code!r}
            output_path = {output_path!r}

            tokenizer = AutoTokenizer.from_pretrained(
                hf_id, trust_remote_code=trust_remote_code)
            model = AutoModel.from_pretrained(
                hf_id, trust_remote_code=trust_remote_code,
                torch_dtype=torch.float32)
            model.eval()

            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)

            # CLS token embedding (first token of last hidden state)
            cls_embedding = outputs.last_hidden_state[0, 0].numpy().tolist()
            result = {{"cls_embedding": cls_embedding}}
            with open(output_path, "w") as f:
                json.dump(result, f)
            print("OK")
        """)

        python = ctx.hf_python or sys.executable
        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script],
            capture_output=True, text=True, timeout=600, env=env,
        )
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            truncated, log_path = save_full_stderr(
                result.stderr, ctx.artifacts_dir or "",
                "hf_encoder_only", case.name)
            msg = f"HF encoder-only failed for {case.name} (rc={result.returncode}):\n{truncated}"
            if log_path:
                msg += f" (full stderr: {log_path})"
            raise RuntimeError(msg)

        data = {}
        if Path(output_path).is_file():
            data = json.loads(Path(output_path).read_text())

        return StageOutput(
            stage_name=stage.name, data=data, timing_s=elapsed,
            metadata={"returncode": result.returncode})

    @staticmethod
    def _resolve_image_path(image_path: str) -> str:
        """Resolve image path, handling relative paths from manifests."""
        if not image_path:
            return image_path
        if os.path.isabs(image_path):
            return image_path
        # Resolve relative to tests/e2e/ directory
        e2e_dir = Path(__file__).resolve().parents[2] / "e2e"
        resolved = e2e_dir / image_path
        if resolved.exists():
            return str(resolved)
        # Also try relative to project root
        project_dir = Path(__file__).resolve().parents[3]
        resolved2 = project_dir / image_path
        if resolved2.exists():
            return str(resolved2)
        return image_path

    def _run_segmentation_ref(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run HF segmentation model as reference."""
        artifacts_dir = ctx.artifacts_dir or tempfile.gettempdir()
        model_dir = _case_artifact_dir(artifacts_dir, case.name) if ctx.artifacts_dir else artifacts_dir
        output_path = str(Path(model_dir) / "hf_seg.npy")

        image_path = self._resolve_image_path(case.inputs.get("image", ""))
        trust_remote_code = case.metadata.get("trust_remote_code", False)
        hf_id = case.hf_id

        script = textwrap.dedent(f"""\
            import numpy as np, torch
            from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor
            from PIL import Image

            hf_id = {hf_id!r}
            image_path = {image_path!r}
            trust_remote_code = {trust_remote_code!r}
            output_path = {output_path!r}

            processor = AutoImageProcessor.from_pretrained(
                hf_id, trust_remote_code=trust_remote_code)
            model = AutoModelForSemanticSegmentation.from_pretrained(
                hf_id, trust_remote_code=trust_remote_code,
                torch_dtype=torch.float32)
            model.eval()

            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits[0].numpy()
            class_map = np.argmax(logits, axis=0).astype(np.int32)
            np.save(output_path, class_map)
            print(f"OK classes={{class_map.max() + 1}}")
        """)

        python = ctx.hf_python or sys.executable
        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script],
            capture_output=True, text=True, timeout=600, env=env,
        )
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            truncated, log_path = save_full_stderr(
                result.stderr, ctx.artifacts_dir or "",
                "hf_segmentation", case.name)
            msg = f"HF segmentation failed for {case.name} (rc={result.returncode}):\n{truncated}"
            if log_path:
                msg += f" (full stderr: {log_path})"
            raise RuntimeError(msg)

        data = {}
        if Path(output_path).is_file():
            data["class_map_path"] = output_path
            import numpy as np
            data["class_map"] = np.load(output_path)

            # Save colorized PNG for human inspection
            try:
                from PIL import Image
                cmap = data["class_map"]
                num_classes = int(cmap.max()) + 1
                # Simple colormap: class index -> hue
                h, w = cmap.shape
                rgb = np.zeros((h, w, 3), dtype=np.uint8)
                for c in range(num_classes):
                    mask = cmap == c
                    # Distribute hues evenly across classes
                    hue = int(255 * c / max(num_classes, 1))
                    rgb[mask] = [hue, 180, 200 if c > 0 else 40]
                # Convert HSV-like to simple distinguishable colors
                np.random.seed(42)
                palette = np.random.randint(0, 255, (num_classes, 3), dtype=np.uint8)
                palette[0] = [0, 0, 0]  # background black
                colored = palette[cmap]
                viz_path = output_path.replace(".npy", "_viz.png")
                Image.fromarray(colored).save(viz_path)
                data["viz_path"] = viz_path
            except Exception as e:
                logger.warning("Failed to save segmentation viz: %s", e)

        return StageOutput(
            stage_name=stage.name, data=data, timing_s=elapsed,
            metadata={"returncode": result.returncode})

    def _run_embedding_ref(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run HF model for embedding and return normalized vector."""
        artifacts_dir = ctx.artifacts_dir or tempfile.gettempdir()
        model_dir = _case_artifact_dir(artifacts_dir, case.name) if ctx.artifacts_dir else artifacts_dir
        output_path = str(Path(model_dir) / "hf_embed.json")

        prompt = case.inputs.get("prompt", "Hello world")
        trust_remote_code = case.metadata.get("trust_remote_code", False)
        hf_id = case.hf_id

        script = textwrap.dedent(f"""\
            import json, torch, numpy as np
            from transformers import AutoModel, AutoTokenizer

            hf_id = {hf_id!r}
            prompt = {prompt!r}
            trust_remote_code = {trust_remote_code!r}
            output_path = {output_path!r}

            tokenizer = AutoTokenizer.from_pretrained(
                hf_id, trust_remote_code=trust_remote_code)
            model = AutoModel.from_pretrained(
                hf_id, trust_remote_code=trust_remote_code,
                torch_dtype=torch.float32)
            model.eval()

            inputs = tokenizer(prompt, return_tensors="pt", padding=True,
                               truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
            # Mean pooling + L2 normalize
            emb = outputs.last_hidden_state.mean(dim=1)[0]
            emb = emb / emb.norm()
            result = {{"embedding": emb.numpy().tolist()}}
            with open(output_path, "w") as f:
                json.dump(result, f)
            print("OK")
        """)

        python = ctx.hf_python or sys.executable
        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script],
            capture_output=True, text=True, timeout=600, env=env,
        )
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            truncated, log_path = save_full_stderr(
                result.stderr, ctx.artifacts_dir or "",
                "hf_embedding", case.name)
            msg = f"HF embedding failed for {case.name} (rc={result.returncode}):\n{truncated}"
            if log_path:
                msg += f" (full stderr: {log_path})"
            raise RuntimeError(msg)

        data = {}
        if Path(output_path).is_file():
            data = json.loads(Path(output_path).read_text())

        return StageOutput(
            stage_name=stage.name, data=data, timing_s=elapsed,
            metadata={"returncode": result.returncode})

    def _run_reranking_ref(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run HF model for reranking and return scores."""
        artifacts_dir = ctx.artifacts_dir or tempfile.gettempdir()
        model_dir = _case_artifact_dir(artifacts_dir, case.name) if ctx.artifacts_dir else artifacts_dir
        output_path = str(Path(model_dir) / "hf_rerank.json")

        prompt = case.inputs.get("prompt", "query: test")
        trust_remote_code = case.metadata.get("trust_remote_code", False)
        hf_id = case.hf_id

        script = textwrap.dedent(f"""\
            import json, torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            hf_id = {hf_id!r}
            prompt = {prompt!r}
            trust_remote_code = {trust_remote_code!r}
            output_path = {output_path!r}

            tokenizer = AutoTokenizer.from_pretrained(
                hf_id, trust_remote_code=trust_remote_code)
            model = AutoModelForSequenceClassification.from_pretrained(
                hf_id, trust_remote_code=trust_remote_code,
                torch_dtype=torch.float32)
            model.eval()

            inputs = tokenizer(prompt, return_tensors="pt", padding=True,
                               truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
            scores = outputs.logits[0].tolist()
            result = {{"scores": scores}}
            with open(output_path, "w") as f:
                json.dump(result, f)
            print("OK")
        """)

        python = ctx.hf_python or sys.executable
        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script],
            capture_output=True, text=True, timeout=600, env=env,
        )
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            truncated, log_path = save_full_stderr(
                result.stderr, ctx.artifacts_dir or "",
                "hf_reranking", case.name)
            msg = f"HF reranking failed for {case.name} (rc={result.returncode}):\n{truncated}"
            if log_path:
                msg += f" (full stderr: {log_path})"
            raise RuntimeError(msg)

        data = {}
        if Path(output_path).is_file():
            data = json.loads(Path(output_path).read_text())

        return StageOutput(
            stage_name=stage.name, data=data, timing_s=elapsed,
            metadata={"returncode": result.returncode})

    def _run_speech_to_text_ref(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run HF Whisper model for speech-to-text reference."""
        artifacts_dir = ctx.artifacts_dir or tempfile.gettempdir()
        model_dir = _case_artifact_dir(artifacts_dir, case.name) if ctx.artifacts_dir else artifacts_dir
        output_path = str(Path(model_dir) / "hf_stt.json")

        audio_path = self._resolve_image_path(case.inputs.get("audio", ""))
        trust_remote_code = case.metadata.get("trust_remote_code", False)
        hf_id = case.hf_id

        script = textwrap.dedent(f"""\
            import json, torch, numpy as np
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
            import scipy.io.wavfile as wav

            hf_id = {hf_id!r}
            audio_path = {audio_path!r}
            trust_remote_code = {trust_remote_code!r}
            output_path = {output_path!r}

            processor = AutoProcessor.from_pretrained(
                hf_id, trust_remote_code=trust_remote_code)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                hf_id, trust_remote_code=trust_remote_code,
                torch_dtype=torch.float32)
            model.eval()

            sr, audio = wav.read(audio_path)
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            # Resample to model's expected sample rate (e.g. 16kHz for Whisper)
            target_sr = getattr(processor.feature_extractor, "sampling_rate", sr)
            if sr != target_sr:
                from scipy.signal import resample
                num_samples = int(len(audio) * target_sr / sr)
                audio = resample(audio, num_samples).astype(np.float32)
                sr = target_sr

            inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=100)
            text = processor.batch_decode(
                generated_ids, skip_special_tokens=True)[0]

            result = {{"text": text}}
            with open(output_path, "w") as f:
                json.dump(result, f)
            print(f"OK text={{text[:100]!r}}")
        """)

        python = ctx.hf_python or sys.executable
        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script],
            capture_output=True, text=True, timeout=600, env=env,
        )
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            truncated, log_path = save_full_stderr(
                result.stderr, ctx.artifacts_dir or "",
                "hf_speech_to_text", case.name)
            msg = f"HF speech-to-text failed for {case.name} (rc={result.returncode}):\n{truncated}"
            if log_path:
                msg += f" (full stderr: {log_path})"
            raise RuntimeError(msg)

        data = {}
        text = ""
        if Path(output_path).is_file():
            data = json.loads(Path(output_path).read_text())
            text = data.get("text", "")

        return StageOutput(
            stage_name=stage.name, data=data, text=text, timing_s=elapsed,
            metadata={"returncode": result.returncode})

    def _run_object_detection_ref(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run HF object detection model as reference."""
        artifacts_dir = ctx.artifacts_dir or tempfile.gettempdir()
        model_dir = _case_artifact_dir(artifacts_dir, case.name) if ctx.artifacts_dir else artifacts_dir
        output_path = str(Path(model_dir) / "hf_det.json")

        image_path = self._resolve_image_path(case.inputs.get("image", ""))
        trust_remote_code = case.metadata.get("trust_remote_code", False)
        hf_id = case.hf_id

        script = textwrap.dedent(f"""\
            import json, torch
            from transformers import AutoModelForObjectDetection, AutoImageProcessor
            from PIL import Image

            hf_id = {hf_id!r}
            image_path = {image_path!r}
            trust_remote_code = {trust_remote_code!r}
            output_path = {output_path!r}

            processor = AutoImageProcessor.from_pretrained(
                hf_id, trust_remote_code=trust_remote_code)
            model = AutoModelForObjectDetection.from_pretrained(
                hf_id, trust_remote_code=trust_remote_code,
                torch_dtype=torch.float32)
            model.eval()

            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            # Post-process to get boxes + scores
            target_sizes = torch.tensor([image.size[::-1]])
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.5)[0]
            detections = []
            for score, label, box in zip(
                results["scores"], results["labels"], results["boxes"]
            ):
                detections.append({{
                    "score": score.item(),
                    "label": label.item(),
                    "box": box.tolist(),
                }})
            with open(output_path, "w") as f:
                json.dump({{"detections": detections}}, f)
            print(f"OK detections={{len(detections)}}")
        """)

        python = ctx.hf_python or sys.executable
        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script],
            capture_output=True, text=True, timeout=600, env=env,
        )
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            truncated, log_path = save_full_stderr(
                result.stderr, ctx.artifacts_dir or "",
                "hf_object_detection", case.name)
            msg = f"HF object detection failed for {case.name} (rc={result.returncode}):\n{truncated}"
            if log_path:
                msg += f" (full stderr: {log_path})"
            raise RuntimeError(msg)

        data = {}
        if Path(output_path).is_file():
            data = json.loads(Path(output_path).read_text())

        return StageOutput(
            stage_name=stage.name, data=data, timing_s=elapsed,
            metadata={"returncode": result.returncode})


    def _run_text_to_audio_ref(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run HF Bark model for text-to-audio reference."""
        artifacts_dir = ctx.artifacts_dir or tempfile.gettempdir()
        model_dir = _case_artifact_dir(artifacts_dir, case.name) if ctx.artifacts_dir else artifacts_dir
        output_path = str(Path(model_dir) / "hf_audio.json")
        wav_path = str(Path(model_dir) / "hf_audio.wav")

        prompt = case.inputs.get("prompt", "Hello, this is a test.")
        trust_remote_code = case.metadata.get("trust_remote_code", False)
        hf_id = case.hf_id

        seed = int(case.determinism.get("seed", 42))
        voice_preset = case.inputs.get("voice_preset", "")

        script = textwrap.dedent(f"""\
            import json, random, struct
            import numpy as np
            import torch
            from transformers import AutoProcessor, BarkModel, set_seed

            hf_id = {hf_id!r}
            prompt = {prompt!r}
            trust_remote_code = {trust_remote_code!r}
            seed = {seed!r}
            voice_preset = {voice_preset!r}
            output_path = {output_path!r}
            wav_path = {wav_path!r}

            # Make Bark reference generation deterministic across runs.
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            set_seed(seed)
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass

            processor = AutoProcessor.from_pretrained(
                hf_id, trust_remote_code=trust_remote_code)
            model = BarkModel.from_pretrained(
                hf_id, trust_remote_code=trust_remote_code,
                torch_dtype=torch.float32)
            model.eval()

            if voice_preset:
                inputs = processor(
                    prompt, voice_preset=voice_preset, return_tensors="pt")
            else:
                inputs = processor(prompt, return_tensors="pt")
            with torch.no_grad():
                audio_values = model.generate(**inputs)

            audio = audio_values.cpu().numpy().squeeze()
            sample_rate = model.generation_config.sample_rate

            # Write WAV
            audio_f32 = audio.astype(np.float32)
            data_bytes = audio_f32.tobytes()
            with open(wav_path, "wb") as f:
                f.write(b"RIFF")
                f.write(struct.pack("<I", 36 + len(data_bytes)))
                f.write(b"WAVE")
                f.write(b"fmt ")
                f.write(struct.pack("<IHHIIHH", 16, 3, 1, sample_rate,
                        sample_rate * 4, 4, 32))
                f.write(b"data")
                f.write(struct.pack("<I", len(data_bytes)))
                f.write(data_bytes)

            rms = float(np.sqrt(np.mean(audio_f32 ** 2)))
            duration = len(audio_f32) / sample_rate
            result = {{"rms": rms, "duration_s": duration,
                      "sample_rate": sample_rate, "num_samples": len(audio_f32),
                      "seed": seed, "voice_preset": voice_preset}}
            with open(output_path, "w") as f:
                json.dump(result, f)
            print(f"OK seed={{seed}} rms={{rms:.4f}} duration={{duration:.2f}}s")
        """)

        python = ctx.hf_python or sys.executable
        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script],
            capture_output=True, text=True, timeout=600, env=env,
        )
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            truncated, log_path = save_full_stderr(
                result.stderr, ctx.artifacts_dir or "",
                "hf_text_to_audio", case.name)
            msg = (f"HF text-to-audio failed for {case.name} "
                   f"(rc={result.returncode}):\n{truncated}")
            if log_path:
                msg += f" (full stderr: {log_path})"
            raise RuntimeError(msg)

        data = {}
        if Path(output_path).is_file():
            data = json.loads(Path(output_path).read_text())
        if Path(wav_path).is_file():
            data["wav_path"] = wav_path

        return StageOutput(
            stage_name=stage.name, data=data, timing_s=elapsed,
            metadata={"returncode": result.returncode})

    def _run_vl_full_generation(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run HF vision-language model for reference generation."""
        artifacts_dir = ctx.artifacts_dir or tempfile.gettempdir()
        model_dir = _case_artifact_dir(artifacts_dir, case.name) if ctx.artifacts_dir else artifacts_dir
        text_path = str(Path(model_dir) / "hf_vl_text.txt")

        prompt = case.inputs.get("prompt", "Describe this image.")
        max_new_tokens = case.inputs.get("max_new_tokens", 30)
        trust_remote_code = case.metadata.get("trust_remote_code", False)
        image_path = self._resolve_image_path(case.inputs.get("image", ""))
        hf_id = case.hf_id

        script = textwrap.dedent(f"""\
            import sys, torch
            from transformers import AutoProcessor
            from PIL import Image

            hf_id = {hf_id!r}
            prompt = {prompt!r}
            max_new_tokens = {max_new_tokens}
            trust_remote_code = {trust_remote_code!r}
            image_path = {image_path!r}
            text_path = {text_path!r}

            processor = AutoProcessor.from_pretrained(
                hf_id, trust_remote_code=trust_remote_code)

            # Try VL-specific auto classes in preference order
            import transformers
            model = None
            for cls_name in ["AutoModelForImageTextToText",
                             "AutoModelForVision2Seq"]:
                try:
                    cls = getattr(transformers, cls_name)
                    model = cls.from_pretrained(
                        hf_id, trust_remote_code=trust_remote_code,
                        torch_dtype=torch.float32)
                    break
                except (AttributeError, ImportError, ValueError, KeyError):
                    continue
            # Fallback for models registered as causal LM with multimodal
            # inputs (e.g. Phi-4-multimodal)
            if model is None:
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    hf_id, trust_remote_code=True,
                    torch_dtype=torch.float32)
            model.eval()

            image = Image.open(image_path).convert("RGB")

            # Build conversation for chat-template models
            messages = [
                {{"role": "user", "content": [
                    {{"type": "image"}},
                    {{"type": "text", "text": prompt}},
                ]}}
            ]
            try:
                text_input = processor.apply_chat_template(
                    messages, add_generation_prompt=True)
                inputs = processor(
                    text=text_input, images=image, return_tensors="pt")
            except Exception:
                # Fallback for models without chat template
                inputs = processor(
                    text=prompt, images=image, return_tensors="pt")

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, max_new_tokens=max_new_tokens)

            # Decode only the generated portion (after input)
            input_len = inputs.get("input_ids", torch.tensor([])).shape[-1]
            gen_ids = generated_ids[0][input_len:]
            text = processor.decode(gen_ids, skip_special_tokens=True)

            with open(text_path, "w") as f:
                f.write(text)
            print(f"OK text={{text[:100]!r}}")
        """)

        python = ctx.hf_python or sys.executable
        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script],
            capture_output=True, text=True, timeout=1800, env=env,
        )
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            truncated, log_path = save_full_stderr(
                result.stderr, ctx.artifacts_dir or "",
                "hf_vl_generation", case.name)
            msg = (f"HF VL generation failed for {case.name} "
                   f"(rc={result.returncode}):\n{truncated}")
            if log_path:
                msg += f" (full stderr: {log_path})"
            raise RuntimeError(msg)

        text = ""
        if Path(text_path).is_file():
            text = Path(text_path).read_text(encoding="utf-8")

        return StageOutput(
            stage_name=stage.name,
            data={"text": text},
            text=text,
            timing_s=elapsed,
            metadata={"returncode": result.returncode,
                       "trust_remote_code": trust_remote_code},
        )

    def _run_prompted_segmentation_ref(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run HF SAM model for prompted segmentation reference."""
        artifacts_dir = ctx.artifacts_dir or tempfile.gettempdir()
        model_dir = _case_artifact_dir(artifacts_dir, case.name) if ctx.artifacts_dir else artifacts_dir
        output_path = str(Path(model_dir) / "hf_sam.json")
        masks_path = str(Path(model_dir) / "hf_sam_masks.npy")

        image_path = self._resolve_image_path(case.inputs.get("image", ""))
        trust_remote_code = case.metadata.get("trust_remote_code", False)
        point_x = case.inputs.get("point_x", 0.5)
        point_y = case.inputs.get("point_y", 0.5)
        hf_id = case.hf_id

        script = textwrap.dedent(f"""\
            import json, torch, numpy as np
            from transformers import SamModel, SamProcessor
            from PIL import Image

            hf_id = {hf_id!r}
            image_path = {image_path!r}
            trust_remote_code = {trust_remote_code!r}
            output_path = {output_path!r}
            masks_path = {masks_path!r}
            point_x_frac = {point_x!r}
            point_y_frac = {point_y!r}

            processor = SamProcessor.from_pretrained(hf_id)
            model = SamModel.from_pretrained(
                hf_id, torch_dtype=torch.float32)
            model.eval()

            image = Image.open(image_path).convert("RGB")
            w, h = image.size

            # Convert fractional coords to pixel coords
            px = int(point_x_frac * w)
            py = int(point_y_frac * h)
            input_points = [[[px, py]]]

            inputs = processor(
                image, input_points=input_points, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)

            masks = processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu()
            )[0]

            iou_scores = outputs.iou_scores[0, 0].cpu().numpy().tolist()
            mask_np = masks[0].cpu().numpy().astype(np.uint8)
            np.save(masks_path, mask_np)

            result = {{
                "iou_scores": iou_scores,
                "num_masks": mask_np.shape[0],
                "mask_shape": list(mask_np.shape),
            }}
            with open(output_path, "w") as f:
                json.dump(result, f)
            print(f"OK masks={{mask_np.shape[0]}} iou={{iou_scores}}")
        """)

        python = ctx.hf_python or sys.executable
        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script],
            capture_output=True, text=True, timeout=600, env=env,
        )
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            truncated, log_path = save_full_stderr(
                result.stderr, ctx.artifacts_dir or "",
                "hf_prompted_segmentation", case.name)
            msg = (f"HF prompted segmentation failed for {case.name} "
                   f"(rc={result.returncode}):\n{truncated}")
            if log_path:
                msg += f" (full stderr: {log_path})"
            raise RuntimeError(msg)

        data = {}
        if Path(output_path).is_file():
            data = json.loads(Path(output_path).read_text())
        if Path(masks_path).is_file():
            data["masks_path"] = masks_path

        return StageOutput(
            stage_name=stage.name, data=data, timing_s=elapsed,
            metadata={"returncode": result.returncode})


plugin = HfTransformersReference()
