"""Minimal Bhashini / Udyat client for Assamese ASR and TTS.

Two-step flow (see dibd-bhashini.gitbook.io/bhashini-apis):
  1. Pipeline config — userID + ulcaApiKey → service IDs + inference endpoint
  2. Pipeline compute — inference key in Authorization → transcript or audio

Keys live in bella/.env (gitignored):
  BHASHINI_USER_ID          from Udyat → My Profile
  BHASHINI_ULCA_API_KEY     from Udyat → My Profile → Generate
  BHASHINI_INFERENCE_KEY    from Udyat → your app → inference API key

Quick test (after keys are set):
  bella/.venv/bin/python3 -m bella.bhashini_client tts "নমস্কাৰ"
  bella/.venv/bin/python3 -m bella.bhashini_client asr /path/to.wav
"""
from __future__ import annotations

import base64
import json
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from bella.env import load_env

CONFIG_URL = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline"
DEFAULT_INFERENCE_URL = "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"

# Documented Assamese-capable models — used only if the config call fails.
_FALLBACK_ASR = "bhashini/ai4bharat/conformer-multilingual-asr"
_FALLBACK_TTS = "ai4bharat/indic-tts-coqui-indo_aryan-gpu--t4"

_config_cache: dict | None = None


class BhashiniError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def configured() -> bool:
    """Inference key is enough for compute; ULCA + user ID are optional."""
    return bool(load_env().get("BHASHINI_INFERENCE_KEY", "").strip())


def _credentials() -> tuple[str, str, str]:
    env = load_env()
    user_id = env.get("BHASHINI_USER_ID", "").strip()
    ulca_key = env.get("BHASHINI_ULCA_API_KEY", "").strip()
    inference_key = env.get("BHASHINI_INFERENCE_KEY", "").strip()
    if not inference_key:
        raise BhashiniError(
            "Missing BHASHINI_INFERENCE_KEY in bella/.env — "
            "Udyat → app maxi → inference API key."
        )
    return user_id, ulca_key, inference_key


def _fallback_config(tasks: list[str], inference_key: str) -> dict:
    """Udyat inference keys often work without the config call."""
    service_ids: dict[str, str] = {}
    for task in tasks:
        if task == "asr":
            service_ids[task] = _FALLBACK_ASR
        elif task == "tts":
            service_ids[task] = _FALLBACK_TTS
        else:
            raise BhashiniError(f"No fallback serviceId for {task}")
    return {
        "callback_url": DEFAULT_INFERENCE_URL,
        "auth_header": "Authorization",
        "auth_value": inference_key,
        "service_ids": service_ids,
    }


def _post(url: str, body: dict, headers: dict, timeout: int = 30) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **headers},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise BhashiniError(f"HTTP {e.code}: {detail}", status=e.code) from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise BhashiniError(f"Could not reach Bhashini ({e})") from e
    except json.JSONDecodeError as e:
        raise BhashiniError(f"Bhashini returned malformed JSON ({e})") from e


def _pick_service_id(configs: list[dict], task: str, language: str,
                     target_language: str | None = None) -> str:
    for entry in configs or []:
        if entry.get("taskType") != task:
            continue
        for item in entry.get("config") or []:
            lang = item.get("language") or {}
            if task == "translation":
                if (lang.get("sourceLanguage") == language
                        and lang.get("targetLanguage") == target_language):
                    sid = item.get("serviceId")
                    if sid:
                        return sid
            elif lang.get("sourceLanguage") == language:
                sid = item.get("serviceId")
                if sid:
                    return sid
    raise BhashiniError(
        f"No Bhashini serviceId for {task} language={language!r}"
        + (f" → {target_language!r}" if target_language else "")
    )


def _load_config(tasks: list[str], language: str = "as") -> dict:
    global _config_cache
    cache_key = (tuple(tasks), language)
    if _config_cache and _config_cache.get("_key") == cache_key:
        return _config_cache

    user_id, ulca_key, inference_key = _credentials()
    result: dict | None = None
    if user_id and ulca_key:
        pipeline_tasks = []
        for task in tasks:
            item: dict = {"taskType": task}
            if task == "asr":
                item["config"] = {"language": {"sourceLanguage": language}}
            elif task == "tts":
                item["config"] = {"language": {"sourceLanguage": language}}
            elif task == "translation":
                item["config"] = {
                    "language": {"sourceLanguage": language, "targetLanguage": "en"}
                }
            pipeline_tasks.append(item)
        try:
            data = _post(
                CONFIG_URL,
                {"pipelineTasks": pipeline_tasks},
                headers={"userID": user_id, "ulcaApiKey": ulca_key},
            )
            endpoint = data.get("pipelineInferenceAPIEndPoint") or {}
            callback = endpoint.get("callbackUrl") or DEFAULT_INFERENCE_URL
            auth = endpoint.get("inferenceApiKey") or {}
            auth_value = auth.get("value") or inference_key
            result = {
                "_key": cache_key,
                "callback_url": callback,
                "auth_header": auth.get("name") or "Authorization",
                "auth_value": auth_value,
                "service_ids": {},
            }
            configs = data.get("pipelineResponseConfig") or []
            for task in tasks:
                try:
                    if task == "translation":
                        result["service_ids"][task] = _pick_service_id(
                            configs, task, language, "en"
                        )
                    else:
                        result["service_ids"][task] = _pick_service_id(
                            configs, task, language
                        )
                except BhashiniError:
                    if task == "asr":
                        result["service_ids"][task] = _FALLBACK_ASR
                    elif task == "tts":
                        result["service_ids"][task] = _FALLBACK_TTS
                    else:
                        raise
        except BhashiniError as e:
            print(f"(Bhashini config call failed ({e}) — using inference key directly)",
                  flush=True)

    if result is None:
        result = _fallback_config(tasks, inference_key)
        result["_key"] = cache_key

    _config_cache = result
    return result


def _compute(body: dict, cfg: dict) -> dict:
    return _post(
        cfg["callback_url"],
        body,
        headers={
            cfg["auth_header"]: cfg["auth_value"],
            "Accept": "*/*",
        },
    )


def _first_output(data: dict, task: str) -> dict:
    responses = data.get("pipelineResponse")
    if isinstance(responses, list):
        for item in responses:
            if item.get("taskType") == task:
                return item
        if responses:
            return responses[0]
    if data.get("taskType") == task:
        return data
    raise BhashiniError(f"No {task} result in Bhashini response")


def service_id(task: str, language: str = "as") -> str:
    """Resolved Bhashini model id for a task (asr, tts, …)."""
    return _load_config([task], language)["service_ids"][task]


def speech_to_text(
    wav_bytes: bytes,
    language: str = "as",
    sample_rate: int = 16000,
    timeout: int = 45,
) -> str:
    """Transcribe mono WAV (16 kHz preferred) to Assamese text."""
    cfg = _load_config(["asr"], language)
    body = {
        "pipelineTasks": [{
            "taskType": "asr",
            "config": {
                "language": {"sourceLanguage": language},
                "serviceId": cfg["service_ids"]["asr"],
                "audioFormat": "wav",
                "samplingRate": sample_rate,
            },
        }],
        "inputData": {
            "input": [{"source": None}],
            "audio": [{
                "audioContent": base64.b64encode(wav_bytes).decode("ascii"),
            }],
        },
    }
    item = _first_output(_compute(body, cfg), "asr")
    outputs = item.get("output") or []
    if not outputs or not (outputs[0].get("source") or "").strip():
        raise BhashiniError("Bhashini ASR returned no transcript")
    return outputs[0]["source"].strip()


def text_to_speech(
    text: str,
    language: str = "as",
    gender: str = "female",
    speed: float = 1.0,
    sample_rate: int = 22050,
    timeout: int = 45,
) -> bytes:
    """Return WAV bytes for Assamese text."""
    cleaned = (text or "").strip()
    if not cleaned:
        raise BhashiniError("no text to speak")
    cfg = _load_config(["tts"], language)
    body = {
        "pipelineTasks": [{
            "taskType": "tts",
            "config": {
                "language": {"sourceLanguage": language},
                "serviceId": cfg["service_ids"]["tts"],
                "gender": gender,
                "speed": speed,
                "samplingRate": sample_rate,
            },
        }],
        "inputData": {
            "input": [{"source": cleaned}],
            "audio": [{"audioContent": None}],
        },
    }
    item = _first_output(_compute(body, cfg), "tts")
    audios = item.get("audio") or []
    if not audios or not audios[0].get("audioContent"):
        raise BhashiniError("Bhashini TTS returned no audio")
    try:
        return base64.b64decode(audios[0]["audioContent"])
    except (ValueError, TypeError) as e:
        raise BhashiniError(f"Bhashini TTS audio could not be decoded ({e})") from e


def _play_wav(wav: bytes) -> None:
    import subprocess

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        path = Path(handle.name)
        handle.write(wav)
    try:
        subprocess.run(["afplay", str(path)], check=False)
    finally:
        path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args or args[0] in ("-h", "--help", "help"):
        print(__doc__.strip())
        return 0
    if not configured():
        print("Set BHASHINI_INFERENCE_KEY in bella/.env first.", file=sys.stderr)
        return 1

    cmd = args[0].lower()
    try:
        if cmd == "tts":
            text = " ".join(args[1:]).strip() or "নমস্কাৰ, মই Bella।"
            wav = text_to_speech(text)
            out = Path("/tmp/bella_bhashini_tts.wav")
            out.write_bytes(wav)
            print(f"TTS ok ({len(wav)} bytes) → {out}")
            _play_wav(wav)
            return 0
        if cmd == "asr":
            if len(args) < 2:
                print("Usage: python -m bella.bhashini_client asr /path/to.wav",
                      file=sys.stderr)
                return 1
            wav_bytes = Path(args[1]).read_bytes()
            text = speech_to_text(wav_bytes)
            print(text)
            return 0
        if cmd == "check":
            cfg = _load_config(["asr", "tts"], "as")
            print("Bhashini configured.")
            print(f"  endpoint: {cfg['callback_url']}")
            print(f"  asr serviceId: {cfg['service_ids'].get('asr')}")
            print(f"  tts serviceId: {cfg['service_ids'].get('tts')}")
            return 0
    except BhashiniError as e:
        print(f"Bhashini error: {e}", file=sys.stderr)
        return 1
    print(f"Unknown command: {cmd}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
