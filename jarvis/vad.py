"""Voice activity detection via Silero VAD, run directly through
onnxruntime — no torch dependency.

The upstream `silero-vad` PyPI package pulls in torch + torchaudio (100MB+)
just to shell out to this exact ONNX model underneath (verified: even its
onnx=True code path unconditionally imports torch at module load). The
model itself only needs onnxruntime, already a dependency here via
openwakeword. The .onnx file was extracted once from that package and is
vendored at jarvis/models/silero_vad.onnx (a couple MB, checked into this
repo) so nothing needs installing at runtime.

This exists to replace the RMS-threshold silence/speech gate in
voice_loop.py, which requires constant recalibration and still produces
false negatives — observed live repeatedly: heard_speech=True (RMS just
over threshold) but Whisper transcribes nothing, because crossing an
amplitude threshold isn't the same as containing actual speech.
"""
from pathlib import Path

import numpy as np
import onnxruntime

MODEL_PATH = Path(__file__).parent / "models" / "silero_vad.onnx"
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512  # the model requires exactly this many samples per call at 16kHz
_CONTEXT_SAMPLES = 64  # sliding context prefix the model expects each call


class SileroVAD:
    def __init__(self, model_path: Path = MODEL_PATH):
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = onnxruntime.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"], sess_options=opts
        )
        self._sr = np.array(SAMPLE_RATE, dtype="int64")
        self.reset()

    def reset(self) -> None:
        """Clear recurrent state — call between unrelated recordings so one
        utterance's context doesn't bleed into the next's probabilities."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT_SAMPLES), dtype=np.float32)

    def speech_probability(self, chunk: np.ndarray) -> float:
        """chunk: exactly CHUNK_SAMPLES float32 samples in [-1, 1] at 16kHz.
        Returns P(speech) in [0, 1]. Stateful — call in order, don't skip
        chunks, and call reset() between unrelated audio streams."""
        if chunk.shape[-1] != CHUNK_SAMPLES:
            raise ValueError(f"expected {CHUNK_SAMPLES} samples, got {chunk.shape[-1]}")
        x = np.concatenate([self._context, chunk.reshape(1, -1).astype(np.float32)], axis=1)
        out, state = self._session.run(None, {"input": x, "state": self._state, "sr": self._sr})
        self._state = state
        self._context = x[:, -_CONTEXT_SAMPLES:]
        return float(out[0][0])
