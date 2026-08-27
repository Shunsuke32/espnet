#!/usr/bin/env python3
"""Prepare ESPnet data directories for FLEURS-only LID and FLEURS+CS-FLEURS LID sequences.

This script intentionally writes both task views from one canonical label pipeline:

* `train_fleurs_lid`, `valid_fleurs_lid`, `test_fleurs_lid` for ESPnet `lid1`.
  These contain single-label `utt2lang` / `lang2utt` files and are used to train
  the FLEURS-only closed-set LID baseline.
* `train_lidseq`, `valid_lidseq`, `test_*` for ESPnet `asr1`.
  Their `text` file is a sequence of language labels such as `<eng>` or `<hin> <eng>` by default.

The canonical labels are ISO-639-3-like lower-case codes, e.g. `eng`, `hin`,
`jpn`, `cmn`. The script fails fast when CS-FLEURS language-pair labels cannot
be parsed, because silently mixing incompatible label inventories invalidates the
baseline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger("prepare_fleurs_cs_lid_data")

# Raw FLEURS config / lang_id names as exposed by HF FLEURS/XTREME-S.
# These are NOT the training tokens by default.  We map them into one canonical
# namespace shared with CS-FLEURS metadata, e.g. en_us -> eng and ar_eg -> ara.
OFFICIAL_FLEURS_CONFIGS: Tuple[str, ...] = (
    "af_za",
    "am_et",
    "ar_eg",
    "as_in",
    "ast_es",
    "az_az",
    "be_by",
    "bg_bg",
    "bn_in",
    "bs_ba",
    "ca_es",
    "ceb_ph",
    "ckb_iq",
    "cmn_hans_cn",
    "cs_cz",
    "cy_gb",
    "da_dk",
    "de_de",
    "el_gr",
    "en_us",
    "es_419",
    "et_ee",
    "fa_ir",
    "ff_sn",
    "fi_fi",
    "fil_ph",
    "fr_fr",
    "ga_ie",
    "gl_es",
    "gu_in",
    "ha_ng",
    "he_il",
    "hi_in",
    "hr_hr",
    "hu_hu",
    "hy_am",
    "id_id",
    "ig_ng",
    "is_is",
    "it_it",
    "ja_jp",
    "jv_id",
    "ka_ge",
    "kam_ke",
    "kea_cv",
    "kk_kz",
    "km_kh",
    "kn_in",
    "ko_kr",
    "ky_kg",
    "lb_lu",
    "lg_ug",
    "ln_cd",
    "lo_la",
    "lt_lt",
    "luo_ke",
    "lv_lv",
    "mi_nz",
    "mk_mk",
    "ml_in",
    "mn_mn",
    "mr_in",
    "ms_my",
    "mt_mt",
    "my_mm",
    "nb_no",
    "ne_np",
    "nl_nl",
    "nso_za",
    "ny_mw",
    "oc_fr",
    "om_et",
    "or_in",
    "pa_in",
    "pl_pl",
    "ps_af",
    "pt_br",
    "ro_ro",
    "ru_ru",
    "sd_in",
    "sk_sk",
    "sl_si",
    "sn_zw",
    "so_so",
    "sr_rs",
    "sv_se",
    "sw_ke",
    "ta_in",
    "te_in",
    "tg_tj",
    "th_th",
    "tr_tr",
    "uk_ua",
    "umb_ao",
    "ur_pk",
    "uz_uz",
    "vi_vn",
    "wo_sn",
    "xh_za",
    "yo_ng",
    "yue_hant_hk",
    "zu_za",
)

FLEURS_TO_ISO3: Dict[str, str] = {
    "af_za": "afr",
    "am_et": "amh",
    "ar_eg": "ara",
    "as_in": "asm",
    "ast_es": "ast",
    "az_az": "aze",
    "be_by": "bel",
    "bg_bg": "bul",
    "bn_in": "ben",
    "bs_ba": "bos",
    "ca_es": "cat",
    "ceb_ph": "ceb",
    "ckb_iq": "ckb",
    "cmn_hans_cn": "cmn",
    "cs_cz": "ces",
    "cy_gb": "cym",
    "da_dk": "dan",
    "de_de": "deu",
    "el_gr": "ell",
    "en_us": "eng",
    "es_419": "spa",
    "et_ee": "est",
    "fa_ir": "fas",
    "ff_sn": "ful",
    "fi_fi": "fin",
    "fil_ph": "tgl",
    "fr_fr": "fra",
    "ga_ie": "gle",
    "gl_es": "glg",
    "gu_in": "guj",
    "ha_ng": "hau",
    "he_il": "heb",
    "hi_in": "hin",
    "hr_hr": "hrv",
    "hu_hu": "hun",
    "hy_am": "hye",
    "id_id": "ind",
    "ig_ng": "ibo",
    "is_is": "isl",
    "it_it": "ita",
    "ja_jp": "jpn",
    "jv_id": "jav",
    "ka_ge": "kat",
    "kam_ke": "kam",
    "kea_cv": "kea",
    "kk_kz": "kaz",
    "km_kh": "khm",
    "kn_in": "kan",
    "ko_kr": "kor",
    "ky_kg": "kir",
    "lb_lu": "ltz",
    "lg_ug": "lug",
    "ln_cd": "lin",
    "lo_la": "lao",
    "lt_lt": "lit",
    "luo_ke": "luo",
    "lv_lv": "lav",
    "mi_nz": "mri",
    "mk_mk": "mkd",
    "ml_in": "mal",
    "mn_mn": "mon",
    "mr_in": "mar",
    "ms_my": "zlm",
    "mt_mt": "mlt",
    "my_mm": "mya",
    "nb_no": "nob",
    "ne_np": "nep",
    "nl_nl": "nld",
    "nso_za": "nso",
    "ny_mw": "nya",
    "oc_fr": "oci",
    "om_et": "orm",
    "or_in": "ory",
    "pa_in": "pan",
    "pl_pl": "pol",
    "ps_af": "pus",
    "pt_br": "por",
    "ro_ro": "ron",
    "ru_ru": "rus",
    "sd_in": "snd",
    "sk_sk": "slk",
    "sl_si": "slv",
    "sn_zw": "sna",
    "so_so": "som",
    "sr_rs": "srp",
    "sv_se": "swe",
    "sw_ke": "swh",
    "ta_in": "tam",
    "te_in": "tel",
    "tg_tj": "tgk",
    "th_th": "tha",
    "tr_tr": "tur",
    "uk_ua": "ukr",
    "umb_ao": "umb",
    "ur_pk": "urd",
    "uz_uz": "uzb",
    "vi_vn": "vie",
    "wo_sn": "wol",
    "xh_za": "xho",
    "yo_ng": "yor",
    "yue_hant_hk": "yue",
    "zu_za": "zul",
    "zh": "cmn",
    "zh_cn": "cmn",
    "zh_hans": "cmn",
    "zh-hans": "cmn",
    "zh_hant": "yue",
    "zh-hant": "yue",
    "cn": "cmn",
    "chinese": "cmn",
    "mandarin": "cmn",
    "english": "eng",
    "arabic": "ara",
    "hindi": "hin",
    "spanish": "spa",
    "japanese": "jpn",
    "french": "fra",
    "german": "deu",
    "portuguese": "por",
    "russian": "rus",
    "korean": "kor",
    "tamil": "tam",
    "telugu": "tel",
    "urdu": "urd",
    "bengali": "ben",
    "malayalam": "mal",
    "malay": "zlm",
    "indonesian": "ind",
    "vietnamese": "vie",
    "tagalog": "tgl",
    "zlm": "zlm",
    "tgl": "tgl",
    "ory": "ory",
}

ALPHA2_TO_ISO3: Dict[str, str] = {
    "af": "afr",
    "am": "amh",
    "ar": "ara",
    "as": "asm",
    "az": "aze",
    "be": "bel",
    "bg": "bul",
    "bn": "ben",
    "bs": "bos",
    "ca": "cat",
    "cs": "ces",
    "cy": "cym",
    "da": "dan",
    "de": "deu",
    "el": "ell",
    "en": "eng",
    "es": "spa",
    "et": "est",
    "fa": "fas",
    "fi": "fin",
    "fr": "fra",
    "ga": "gle",
    "gl": "glg",
    "gu": "guj",
    "ha": "hau",
    "he": "heb",
    "hi": "hin",
    "hr": "hrv",
    "hu": "hun",
    "hy": "hye",
    "id": "ind",
    "ig": "ibo",
    "is": "isl",
    "it": "ita",
    "ja": "jpn",
    "jv": "jav",
    "ka": "kat",
    "kk": "kaz",
    "km": "khm",
    "kn": "kan",
    "ko": "kor",
    "ky": "kir",
    "lb": "ltz",
    "ln": "lin",
    "lo": "lao",
    "lt": "lit",
    "lv": "lav",
    "mi": "mri",
    "mk": "mkd",
    "ml": "mal",
    "mn": "mon",
    "mr": "mar",
    "ms": "zlm",
    "mt": "mlt",
    "my": "mya",
    "ne": "nep",
    "nl": "nld",
    "no": "nob",
    "om": "orm",
    "or": "ory",
    "pa": "pan",
    "pl": "pol",
    "ps": "pus",
    "pt": "por",
    "ro": "ron",
    "ru": "rus",
    "sd": "snd",
    "sk": "slk",
    "sl": "slv",
    "sn": "sna",
    "so": "som",
    "sr": "srp",
    "sv": "swe",
    "sw": "swh",
    "ta": "tam",
    "te": "tel",
    "tg": "tgk",
    "th": "tha",
    "tr": "tur",
    "uk": "ukr",
    "ur": "urd",
    "uz": "uzb",
    "vi": "vie",
    "wo": "wol",
    "xh": "xho",
    "yo": "yor",
    "zh": "cmn",
    "zu": "zul",
}

ISO3_ALIASES: Dict[str, str] = {
    "alb": "sqi",
    "arm": "hye",
    "baq": "eus",
    "bur": "mya",
    "chi": "zho",
    "cze": "ces",
    "dut": "nld",
    "fre": "fra",
    "geo": "kat",
    "ger": "deu",
    "gre": "ell",
    "ice": "isl",
    "mac": "mkd",
    "mao": "mri",
    "may": "zlm",
    "per": "fas",
    "rum": "ron",
    "slo": "slk",
    "tib": "bod",
    "wel": "cym",
    "ara_arb": "ara",
    "arb": "ara",
    "zho": "cmn",
    "cmn_hans": "cmn",
    "msa": "zlm",
    "zsm": "zlm",
    "fil": "tgl",
    "tl": "tgl",
    "ori": "ory",
    "nor": "nob",
}

_TOKEN_RE = re.compile(r"^[a-z][a-z0-9_\-]*$")


@dataclass(frozen=True)
class Example:
    uttid: str
    wav: str
    labels: Tuple[str, ...]
    speaker: str
    source: str
    subset: str
    raw_label_value: str = ""
    raw_label_parts: Tuple[str, ...] = ()
    duration_sec: Optional[float] = None
    num_samples: Optional[int] = None


def sanitize_id(text: object) -> str:
    s = re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(text))
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "utt"


def stable_fraction(key: str) -> float:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def _to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or x < 0:
        return None
    return x


def _to_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        x = int(value)
    except (TypeError, ValueError):
        return None
    return x if x >= 0 else None


def audio_duration_from_file(path: object) -> Optional[Tuple[float, int]]:
    """Return (duration_sec, num_samples) for ordinary audio paths when possible.

    Kaldi-style wav.scp can contain shell pipes.  Those are intentionally not
    opened here; ESPnet's format_wav_scp.sh will handle them later.  For FLEURS
    and cloned CS-FLEURS we expect ordinary local files, so this audit normally
    succeeds.
    """
    if path is None:
        return None
    p = str(path).strip()
    if not p or p.endswith("|") or " |" in p or p.startswith("|"):
        return None
    pp = Path(p)
    if not pp.exists():
        return None
    try:
        import soundfile as sf  # type: ignore

        info = sf.info(str(pp))
        if info.samplerate and info.frames is not None:
            return float(info.frames) / float(info.samplerate), int(info.frames)
    except Exception:
        pass
    try:
        import wave

        with wave.open(str(pp), "rb") as wf:
            frames = wf.getnframes()
            sr = wf.getframerate()
            if sr:
                return float(frames) / float(sr), int(frames)
    except Exception:
        pass
    return None


def infer_duration_from_row(
    row: dict, wav: object, default_sample_rate: int
) -> Tuple[Optional[float], Optional[int]]:
    """Infer duration from dataset metadata, falling back to audio header."""
    duration = (
        _to_float(row.get("duration"))
        or _to_float(row.get("duration_sec"))
        or _to_float(row.get("audio_duration"))
        or _to_float(row.get("seconds"))
    )
    audio = row.get("audio")
    sr = (
        _to_int(row.get("sampling_rate"))
        or _to_int(row.get("sample_rate"))
        or default_sample_rate
    )
    num_samples = _to_int(row.get("num_samples")) or _to_int(row.get("n_samples"))
    if isinstance(audio, dict):
        sr = _to_int(audio.get("sampling_rate")) or sr
        num_samples = num_samples or _to_int(audio.get("num_samples"))
        if num_samples is None and isinstance(audio.get("array"), (list, tuple)):
            num_samples = len(audio["array"])
    if duration is None and num_samples is not None and sr:
        duration = float(num_samples) / float(sr)
    if duration is None:
        info = audio_duration_from_file(wav)
        if info is not None:
            duration, ns = info
            num_samples = num_samples or ns
    return duration, num_samples


def str2bool(x: object) -> bool:
    return str(x).lower() in {"1", "true", "t", "yes", "y"}


def read_label_map(path: Optional[Path]) -> Dict[str, str]:
    mapping = dict(FLEURS_TO_ISO3)
    if path is None:
        return mapping
    if not path.exists():
        raise FileNotFoundError(f"label map not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < 2:
                raise ValueError(
                    f"{path}:{lineno}: expected '<source_label> <canonical_label>'"
                )
            mapping[parts[0].lower()] = ISO3_ALIASES.get(
                parts[1].lower(), parts[1].lower()
            )
    return mapping


def verify_fleurs_config_mapping(mapping: Dict[str, str]) -> None:
    """Fail fast if the raw FLEURS label namespace is not fully auditable."""
    missing = [cfg for cfg in OFFICIAL_FLEURS_CONFIGS if cfg not in mapping]
    if missing:
        raise ValueError(
            "FLEURS config labels missing from label map: " + ",".join(missing)
        )
    canonical = [
        ISO3_ALIASES.get(mapping[cfg], mapping[cfg]) for cfg in OFFICIAL_FLEURS_CONFIGS
    ]
    if len(set(canonical)) != len(canonical):
        inv: Dict[str, List[str]] = defaultdict(list)
        for cfg, lab in zip(OFFICIAL_FLEURS_CONFIGS, canonical):
            inv[lab].append(cfg)
        duplicates = {lab: cfgs for lab, cfgs in inv.items() if len(cfgs) > 1}
        raise ValueError(
            "FLEURS config labels collapse into duplicate canonical labels: "
            + json.dumps(duplicates, ensure_ascii=False, sort_keys=True)
        )


def try_pycountry(token: str) -> Optional[str]:
    try:
        import pycountry  # type: ignore
    except Exception:
        return None
    norm = token.replace("-", "_").lower()
    if len(norm) == 2 and norm.isalpha():
        lang = pycountry.languages.get(alpha_2=norm)
        return getattr(lang, "alpha_3", None) if lang is not None else None
    if len(norm) == 3 and norm.isalpha():
        lang = pycountry.languages.get(alpha_3=norm)
        return getattr(lang, "alpha_3", None) if lang is not None else None
    try:
        lang = pycountry.languages.lookup(norm.replace("_", " "))
        return getattr(lang, "alpha_3", None)
    except LookupError:
        return None


def canonicalize_label(
    raw: object, mapping: Dict[str, str], *, strict: bool = True
) -> str:
    token = str(raw).strip().lower().strip("<>[](){}'\"")
    token = token.replace(".", "")
    token_us = token.replace("-", "_")
    for cand in (token, token_us):
        if cand in mapping:
            return ISO3_ALIASES.get(mapping[cand], mapping[cand])
        if cand in ISO3_ALIASES:
            return ISO3_ALIASES[cand]
    if len(token) == 3 and token.isalpha():
        return token
    if len(token) == 2 and token.isalpha() and token in ALPHA2_TO_ISO3:
        return ALPHA2_TO_ISO3[token]
    pc = try_pycountry(token_us) or try_pycountry(token)
    if pc:
        pc = pc.lower()
        return ISO3_ALIASES.get(pc, pc)
    if strict:
        raise ValueError(f"Cannot canonicalize language label: {raw!r}")
    return token_us


def split_candidate_labels(value: object, mapping: Dict[str, str]) -> List[str]:
    """Extract candidate language tokens from values such as `ara-eng` or `eng+hin`."""
    if value is None:
        return []
    s = str(value).strip().lower()
    if not s:
        return []
    s = s.replace("–", "-").replace("—", "-").replace("→", "-").replace("/", "-")
    s = s.replace("<", " ").replace(">", " ")
    raw_parts = re.split(r"[+,;:|\s]+", s)
    parts: List[str] = []
    for p in raw_parts:
        p = p.strip("()[]{}'\"")
        if not p:
            continue
        p_us = p.replace("-", "_")
        if p in mapping or p_us in mapping:
            parts.append(p)
            continue
        if re.match(r"^[a-z]{2,3}[-_][a-z]{2,3}$", p):
            parts.extend([q for q in re.split(r"[-_]", p) if q])
            continue
        if "-" in p:
            parts.extend([q for q in p.split("-") if q])
            continue
        parts.append(p)
    return [p for p in parts if _TOKEN_RE.match(p)]


def parse_label_sequence(
    value: object, mapping: Dict[str, str], *, strict: bool = True
) -> Tuple[str, ...]:
    labels: List[str] = []
    for part in split_candidate_labels(value, mapping):
        try:
            label = canonicalize_label(part, mapping, strict=strict)
        except ValueError:
            if strict:
                raise
            continue
        if not labels or labels[-1] != label:
            labels.append(label)
    seen = set()
    unique: List[str] = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            unique.append(label)
    return tuple(unique)


def token_text(labels: Sequence[str], token_format: str) -> str:
    if token_format == "angle":
        return " ".join(f"<{x}>" for x in labels)
    if token_format == "plain":
        return " ".join(labels)
    raise ValueError(f"unsupported token_format={token_format}")


def write_spk2utt(utt2spk: Dict[str, str], path: Path) -> None:
    spk2utts: Dict[str, List[str]] = defaultdict(list)
    for utt, spk in sorted(utt2spk.items()):
        spk2utts[spk].append(utt)
    with path.open("w", encoding="utf-8") as f:
        for spk, utts in sorted(spk2utts.items()):
            f.write(f"{spk} {' '.join(sorted(utts))}\n")


def write_lang2utt(examples: Sequence[Example], path: Path) -> None:
    lang2utts: Dict[str, List[str]] = defaultdict(list)
    for ex in examples:
        # ESPnet lid1 is single-label classification; multi-label references are
        # kept in utt2langs and scored separately.
        lang2utts[ex.labels[0]].append(ex.uttid)
    with path.open("w", encoding="utf-8") as f:
        for lang, utts in sorted(lang2utts.items()):
            f.write(f"{lang} {' '.join(sorted(set(utts)))}\n")


def write_data_dir(
    name: str, examples: Sequence[Example], outdir: Path, token_format: str
) -> None:
    d = outdir / name
    d.mkdir(parents=True, exist_ok=True)
    examples = sorted(
        examples, key=lambda e: (str(e.uttid), str(e.wav), str(e.speaker))
    )
    # Use utterance-level speakers.  Kaldi/ESPnet fix_data_dir.sh regenerates
    # utt2spk through spk2utt; shared speaker ids can regroup utterances and
    # break the required global uttid sort order for these LID manifests.
    utt2spk = {ex.uttid: ex.uttid for ex in examples}
    with (
        (d / "wav.scp").open("w", encoding="utf-8") as wav_f,
        (d / "text").open("w", encoding="utf-8") as text_f,
        (d / "utt2spk").open("w", encoding="utf-8") as u2s_f,
        (d / "utt2lang").open("w", encoding="utf-8") as u2l_f,
        (d / "utt2langs").open("w", encoding="utf-8") as u2ls_f,
        (d / "utt2category").open("w", encoding="utf-8") as u2c_f,
        (d / "labels.jsonl").open("w", encoding="utf-8") as js_f,
    ):
        for ex in examples:
            if not ex.labels:
                raise ValueError(f"empty label sequence for {ex.uttid}")
            wav_f.write(f"{ex.uttid} {ex.wav}\n")
            text_f.write(f"{ex.uttid} {token_text(ex.labels, token_format)}\n")
            u2s_f.write(f"{ex.uttid} {utt2spk[ex.uttid]}\n")
            u2l_f.write(f"{ex.uttid} {ex.labels[0]}\n")
            u2ls_f.write(f"{ex.uttid} {' '.join(ex.labels)}\n")
            u2c_f.write(f"{ex.uttid} {ex.source}\n")
            js_f.write(
                json.dumps(
                    {
                        "uttid": ex.uttid,
                        "wav": ex.wav,
                        "labels": list(ex.labels),
                        "speaker": ex.speaker,
                        "source": ex.source,
                        "subset": ex.subset,
                        "raw_label_value": ex.raw_label_value,
                        "raw_label_parts": list(ex.raw_label_parts),
                        "duration_sec": ex.duration_sec,
                        "num_samples": ex.num_samples,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    write_spk2utt(utt2spk, d / "spk2utt")
    write_lang2utt(examples, d / "lang2utt")
    LOGGER.info("wrote %s with %d utterances", d, len(examples))


RAW_FLEURS_SYMBOL_RE = re.compile(r"^<?[a-z]+_[a-z0-9_]+>?$")
BRACKET_PROMPT_RE = re.compile(r"^\[[a-z]+_[a-z0-9_]+\]$")


def target_symbol(label: str, token_format: str) -> str:
    if token_format == "angle":
        return f"<{label}>"
    if token_format == "plain":
        return label
    raise ValueError(f"unsupported token_format for nlsyms: {token_format}")


def validate_nlsyms(symbols: Sequence[str]) -> None:
    if not symbols:
        raise ValueError("nlsyms inventory is empty")
    bad = [
        sym
        for sym in symbols
        if RAW_FLEURS_SYMBOL_RE.fullmatch(sym) or BRACKET_PROMPT_RE.fullmatch(sym)
    ]
    if bad:
        raise ValueError(
            "nlsyms contains raw FLEURS labels or multilingual ASR prompts: "
            + ", ".join(bad[:20])
        )


def write_nlsyms(
    data_dirs: Dict[str, Sequence[Example]], token_format: str, nlsyms_txt: Path
) -> None:
    labels = sorted(
        {lab for examples in data_dirs.values() for ex in examples for lab in ex.labels}
    )
    symbols = [target_symbol(label, token_format) for label in labels]
    validate_nlsyms(symbols)
    nlsyms_txt.parent.mkdir(parents=True, exist_ok=True)
    with nlsyms_txt.open("w", encoding="utf-8") as f:
        for sym in symbols:
            f.write(f"{sym}\n")
    LOGGER.info("wrote canonical nlsyms: %s with %d symbols", nlsyms_txt, len(symbols))


def iter_json_objects(path: Path) -> Iterator[dict]:
    """Iterate JSON objects from JSONL or whitespace-separated JSON objects."""
    decoder = json.JSONDecoder()
    text = path.read_text(encoding="utf-8")
    idx = 0
    n = len(text)
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError as e:
            snippet = text[idx : idx + 120].replace("\n", "\\n")
            raise ValueError(
                f"{path}: invalid JSON near offset {idx}: {snippet}"
            ) from e
        if not isinstance(obj, dict):
            raise ValueError(
                f"{path}: expected object near offset {idx}, got {type(obj).__name__}"
            )
        yield obj
        idx = end


def manifest_path(root: Path, split: str) -> Path:
    candidates = [
        root / f"{split}.jsonl",
        root / f"{split}.json",
        root / split / "metadata.jsonl",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No manifest for split={split} under {root}; tried {candidates}"
    )


def fleurs_uttid(raw_lang: object, rid: object, wav: object) -> str:
    """Build a collision-resistant FLEURS utterance id.

    ESPnet's upstream FLEURS data_prep.pl derives uttId from the filepath and
    speaker/id, not from a bare FLEURS sentence id.  That matters because ids
    such as 759 are reused across languages and can also appear in multiple
    recordings.  We follow the same principle: include raw language plus a
    filepath-derived component.  The split is intentionally not part of the id,
    so accidental train/dev/test audio overlap remains detectable by key.
    """
    wav_s = str(wav)
    stem = sanitize_id(Path(wav_s).stem)
    h = hashlib.sha1(wav_s.encode("utf-8")).hexdigest()[:12]
    return sanitize_id(f"fleurs_{raw_lang}_{rid}_{stem}_{h}")


def extract_lang_from_sentence(sentence: object) -> Optional[str]:
    if sentence is None:
        return None
    m = re.match(r"^\s*\[([^\]]+)\]", str(sentence))
    return m.group(1).strip() if m else None


def load_fleurs_manifest_split(
    root: Path,
    split: str,
    mapping: Dict[str, str],
    subsample_per_lang: int,
    strict_labels: bool,
) -> List[Example]:
    path = manifest_path(root, split)
    per_lang_count: Dict[str, int] = defaultdict(int)
    examples: List[Example] = []
    for row in iter_json_objects(path):
        raw_lang = (
            row.get("lang_id_name")
            or extract_lang_from_sentence(row.get("sentence"))
            or row.get("lang_id")
            or row.get("language")
            or row.get("config")
        )
        if raw_lang is None:
            raise ValueError(
                f"{path}: row has no lang_id/lang_id_name/language/config: {row}"
            )
        if isinstance(raw_lang, int):
            raise ValueError(
                f"{path}: integer lang_id needs lang_id_name in manifest mode: {row}"
            )
        lang = canonicalize_label(raw_lang, mapping, strict=strict_labels)
        if subsample_per_lang > 0:
            if per_lang_count[lang] >= subsample_per_lang:
                continue
            per_lang_count[lang] += 1
        wav = row.get("path") or row.get("file_name") or row.get("audio")
        if isinstance(wav, dict):
            wav = wav.get("path")
        if not wav:
            raise ValueError(
                f"{path}: FLEURS manifest row has no path/file_name/audio: {row}"
            )
        rid = row.get("id") or row.get("client_id") or Path(str(wav)).stem
        uttid = fleurs_uttid(raw_lang, rid, wav)
        speaker = (
            row.get("speaker")
            or row.get("client_id")
            or f"{lang}_{row.get('gender', 'spk')}"
        )
        duration_sec, num_samples = infer_duration_from_row(row, wav, 16000)
        examples.append(
            Example(
                uttid,
                str(wav),
                (lang,),
                f"fleurs_{speaker}",
                "fleurs",
                split,
                str(raw_lang),
                (str(raw_lang),),
                duration_sec,
                num_samples,
            )
        )
    LOGGER.info("loaded FLEURS manifest split=%s examples=%d", split, len(examples))
    return examples


def tsv_split_name(split: str) -> str:
    return {"train": "train", "validation": "dev", "dev": "dev", "test": "test"}.get(
        split, split
    )


def load_fleurs_tsv_split(
    root: Path,
    split: str,
    mapping: Dict[str, str],
    subsample_per_lang: int,
    strict_labels: bool,
) -> List[Example]:
    """Read the TSV layout produced by local/create_fleurs_lid_dataset.py.

    This mirrors egs2/fleurs/asr1/local/data_prep.pl in the important part:
    the audio path participates in the utterance key.  We never save audio as
    <id>.wav, so repeated FLEURS ids cannot overwrite each other.
    """
    tsv = root / f"{tsv_split_name(split)}.tsv"
    if not tsv.exists():
        raise FileNotFoundError(f"FLEURS TSV not found: {tsv}")
    per_lang_count: Dict[str, int] = defaultdict(int)
    examples: List[Example] = []
    with tsv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            raw_lang = (
                row.get("lang_id_name")
                or extract_lang_from_sentence(row.get("sentence"))
                or extract_lang_from_sentence(row.get("transcription"))
                or row.get("config")
                or row.get("language")
                or row.get("accent")
            )
            if raw_lang is None or str(raw_lang).strip() == "":
                raise ValueError(
                    f"{tsv}: row has no lang_id_name / [lang] sentence prefix: {row}"
                )
            lang = canonicalize_label(raw_lang, mapping, strict=strict_labels)
            if subsample_per_lang > 0:
                if per_lang_count[lang] >= subsample_per_lang:
                    continue
                per_lang_count[lang] += 1
            wav = row.get("path") or row.get("filepath") or row.get("file_name")
            if not wav:
                raise ValueError(f"{tsv}: row has no path/filepath/file_name: {row}")
            rid = row.get("id") or row.get("client_id") or Path(str(wav)).stem
            uttid = fleurs_uttid(raw_lang, rid, wav)
            speaker = (
                row.get("speaker")
                or row.get("client_id")
                or f"{lang}_{row.get('gender', 'spk')}"
            )
            duration_sec, num_samples = infer_duration_from_row(row, wav, 16000)
            examples.append(
                Example(
                    uttid,
                    str(wav),
                    (lang,),
                    f"fleurs_{speaker}",
                    "fleurs",
                    split,
                    str(raw_lang),
                    (str(raw_lang),),
                    duration_sec,
                    num_samples,
                )
            )
    LOGGER.info(
        "loaded FLEURS TSV split=%s path=%s examples=%d", split, tsv, len(examples)
    )
    return examples


def load_fleurs_hf_split(
    config: str,
    split: str,
    mapping: Dict[str, str],
    cache_dir: Optional[str],
    subsample_per_lang: int,
    strict_labels: bool,
) -> List[Example]:
    """Direct loader for official google/fleurs when TSVs are not supplied."""
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Please install Hugging Face datasets: pip install datasets soundfile"
        ) from e

    hf_config = config.split(".", 1)[1] if config.startswith("fleurs.") else config
    if hf_config in {"all", "fleurs.all"}:
        raise ValueError(
            "Direct HF loading with --fleurs_config all is not supported here; run local/create_fleurs_lid_dataset.py first and pass --fleurs_tsv_root"
        )
    LOGGER.info("loading google/fleurs config=%s split=%s", hf_config, split)
    ds = load_dataset(
        "google/fleurs",
        hf_config,
        split=split,
        streaming=subsample_per_lang > 0,
        cache_dir=cache_dir,
    )
    per_lang_count: Dict[str, int] = defaultdict(int)
    examples: List[Example] = []
    for row in ds:
        raw_lang = hf_config
        lang = canonicalize_label(raw_lang, mapping, strict=strict_labels)
        if subsample_per_lang > 0:
            if per_lang_count[lang] >= subsample_per_lang:
                continue
            per_lang_count[lang] += 1
        wav = row.get("path") or row.get("file_name")
        if not wav:
            audio = row.get("audio") or {}
            if isinstance(audio, dict):
                wav = audio.get("path")
        if not wav:
            raise ValueError(
                f"FLEURS example has no audio path. Use local/create_fleurs_lid_dataset.py first. row={row}"
            )
        rid = row.get("id") or Path(str(wav)).stem
        uttid = fleurs_uttid(raw_lang, rid, wav)
        speaker = f"fleurs_{lang}_{row.get('gender', 'spk')}"
        duration_sec, num_samples = infer_duration_from_row(row, wav, 16000)
        examples.append(
            Example(
                uttid,
                str(wav),
                (lang,),
                speaker,
                "fleurs",
                split,
                str(raw_lang),
                (str(raw_lang),),
                duration_sec,
                num_samples,
            )
        )
    LOGGER.info(
        "loaded google/fleurs config=%s split=%s examples=%d",
        hf_config,
        split,
        len(examples),
    )
    return examples


def load_fleurs(
    config: str,
    split: str,
    mapping: Dict[str, str],
    cache_dir: Optional[str],
    subsample_per_lang: int,
    strict_labels: bool,
    manifest_root: Optional[Path],
    tsv_root: Optional[Path],
) -> List[Example]:
    if manifest_root is not None:
        return load_fleurs_manifest_split(
            manifest_root, split, mapping, subsample_per_lang, strict_labels
        )
    if tsv_root is not None:
        return load_fleurs_tsv_split(
            tsv_root, split, mapping, subsample_per_lang, strict_labels
        )
    if config in {"all", "fleurs.all"}:
        configs = list(OFFICIAL_FLEURS_CONFIGS)
    else:
        configs = [c.strip() for c in config.split(",") if c.strip()]
    if not configs:
        raise ValueError("--fleurs_config is empty")
    all_examples: List[Example] = []
    for cfg in configs:
        all_examples.extend(
            load_fleurs_hf_split(
                cfg, split, mapping, cache_dir, subsample_per_lang, strict_labels
            )
        )
    return all_examples


def remove_fleurs_train_overlaps(
    train: List[Example], valid: Sequence[Example], test: Sequence[Example]
) -> Tuple[List[Example], List[Example]]:
    """Remove FLEURS train recordings that also appear in validation/test.

    This mirrors the upstream recipe's leakage guard, which filters the train
    wav.scp using dev/test wav.scp after data_prep.  We compare both uttid and
    wav path so path-level overlap is caught even if a user changes id policy.
    """
    blocked_ids = {ex.uttid for ex in [*valid, *test]}
    blocked_wavs = {ex.wav for ex in [*valid, *test]}
    kept: List[Example] = []
    removed: List[Example] = []
    for ex in train:
        if ex.uttid in blocked_ids or ex.wav in blocked_wavs:
            removed.append(ex)
        else:
            kept.append(ex)
    if removed:
        LOGGER.warning(
            "removed %d FLEURS train examples overlapping dev/test", len(removed)
        )
    return kept, removed


def _norm_subset_name(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", x.lower())


def resolve_cs_subset_dir(cs_root: Path, subset: str) -> Path:
    """Resolve subset names across common CS-FLEURS clone layouts.

    The dataset card/paper names are often written as Read-Test, XTTS-Train,
    XTTS-Test1, XTTS-Test2, and MMS-Test, while a Git LFS clone may use flat
    directories (e.g. XTTS-Train) or nested directories (e.g. xtts/train).
    """
    variants = [
        subset,
        subset.replace("/", "-"),
        subset.replace("-", "/"),
        subset.lower(),
        subset.upper(),
        subset.title(),
        subset.replace("/", "-").lower(),
        subset.replace("/", "-").upper(),
        subset.replace("/", "-").title(),
    ]
    seen = set()
    for v in variants:
        if v in seen:
            continue
        seen.add(v)
        d = cs_root / v
        if (d / "metadata.jsonl").exists():
            return d
    target = _norm_subset_name(subset)
    for meta in cs_root.rglob("metadata.jsonl"):
        rel = str(meta.parent.relative_to(cs_root))
        if _norm_subset_name(rel) == target:
            return meta.parent
    raise FileNotFoundError(
        f"CS-FLEURS subset {subset!r} not found under {cs_root}. "
        "Expected metadata.jsonl in a matching directory."
    )


def find_audio_path(cs_root: Path, subset_dir: Path, file_name: object) -> str:
    p = Path(str(file_name))
    candidates: List[Path] = []
    if p.is_absolute():
        candidates.append(p)
    candidates.extend(
        [
            subset_dir / p,
            subset_dir / "audio" / p,
            cs_root / p,
            subset_dir / "audio" / p.name,
        ]
    )
    for cand in candidates:
        if cand.exists():
            return str(cand)
    return str(candidates[0])


def load_cs_subset(
    cs_root: Path,
    subset: str,
    mapping: Dict[str, str],
    pair_field: str,
    strict_labels: bool,
    allow_single_cs: bool,
) -> List[Example]:
    subset_dir = resolve_cs_subset_dir(cs_root, subset)
    meta = subset_dir / "metadata.jsonl"
    examples: List[Example] = []
    for row in iter_json_objects(meta):
        pair_value = row.get(pair_field)
        if pair_value is None:
            raise KeyError(
                f"{meta}: row is missing pair field {pair_field!r}; keys={list(row.keys())}"
            )
        raw_parts = tuple(split_candidate_labels(pair_value, mapping))
        labels = parse_label_sequence(pair_value, mapping, strict=strict_labels)
        if len(labels) == 0:
            raise ValueError(
                f"Could not parse labels from {pair_field}={pair_value!r} in {meta}"
            )
        if len(labels) < 2 and not allow_single_cs:
            raise ValueError(
                f"CS-FLEURS row parsed as a single label {labels} from {pair_field}={pair_value!r}. "
                "Pass --allow_single_cs true only if this is expected, otherwise fix --cs_pair_field or --label_map."
            )
        file_name = row.get("file_name") or row.get("path") or row.get("audio")
        if isinstance(file_name, dict):
            file_name = file_name.get("path")
        if not file_name:
            raise ValueError(f"CS-FLEURS row has no file_name/path/audio: {row}")
        rid = row.get("id") or Path(str(file_name)).stem
        uttid = sanitize_id(f"cs_{subset.replace('/', '_')}_{rid}")
        speaker = row.get("speaker") or "spk"
        wav_path = find_audio_path(cs_root, subset_dir, file_name)
        duration_sec, num_samples = infer_duration_from_row(row, wav_path, 16000)
        examples.append(
            Example(
                uttid=uttid,
                wav=wav_path,
                labels=labels,
                speaker=f"cs_{sanitize_id(speaker)}",
                source="cs_fleurs",
                subset=subset,
                raw_label_value=str(pair_value),
                raw_label_parts=raw_parts,
                duration_sec=duration_sec,
                num_samples=num_samples,
            )
        )
    LOGGER.info("loaded CS-FLEURS subset=%s examples=%d", subset, len(examples))
    return examples


def load_cs_subsets(
    cs_root: Path,
    subsets: Sequence[str],
    mapping: Dict[str, str],
    pair_field: str,
    strict_labels: bool,
    allow_single_cs: bool,
) -> Dict[str, List[Example]]:
    data: Dict[str, List[Example]] = {}
    for subset in subsets:
        if subset:
            data[subset] = load_cs_subset(
                cs_root, subset, mapping, pair_field, strict_labels, allow_single_cs
            )
    return data


def split_cs_train_valid_by_class(
    cs_train_sets: Dict[str, List[Example]],
    dev_ratio: float,
) -> Tuple[List[Example], List[Example], List[Tuple[str, int, int, int]]]:
    """Split CS-FLEURS train data within each canonical label class.

    The old recipe used one global hash threshold, which yielded an approximate
    source-level split but could leave individual language-pair classes with
    very small or empty validation coverage.  Here each canonical label sequence
    is split independently, e.g. ``ara eng`` gets its own 9:1 train/dev split.
    """
    flat: List[Example] = []
    by_class: Dict[Tuple[str, ...], List[Example]] = defaultdict(list)
    for _subset, exs in cs_train_sets.items():
        for ex in exs:
            flat.append(ex)
            by_class[tuple(ex.labels)].append(ex)

    valid_ids = set()
    audit_rows: List[Tuple[str, int, int, int]] = []
    for labels, exs in sorted(by_class.items(), key=lambda kv: kv[0]):
        n_total = len(exs)
        if dev_ratio <= 0.0 or n_total < 2:
            n_valid = 0
        else:
            n_valid = int(round(n_total * dev_ratio))
            n_valid = max(1, min(n_total - 1, n_valid))
        ranked = sorted(exs, key=lambda ex: (stable_fraction(ex.uttid), ex.uttid))
        valid_ids.update(ex.uttid for ex in ranked[:n_valid])
        audit_rows.append((" ".join(labels), n_total, n_total - n_valid, n_valid))

    train = [ex for ex in flat if ex.uttid not in valid_ids]
    valid = [ex for ex in flat if ex.uttid in valid_ids]
    LOGGER.info(
        "split CS-FLEURS train subsets by canonical label class: train=%d valid=%d classes=%d dev_ratio=%.4f",
        len(train),
        len(valid),
        len(audit_rows),
        dev_ratio,
    )
    return train, valid, audit_rows


def split_cs_train_valid_global_hash(
    cs_train_sets: Dict[str, List[Example]],
    dev_ratio: float,
) -> Tuple[List[Example], List[Example], List[Tuple[str, int, int, int]]]:
    """Reproduce the paper split using one deterministic hash threshold."""
    flat = [ex for examples in cs_train_sets.values() for ex in examples]
    valid_ids = {ex.uttid for ex in flat if stable_fraction(ex.uttid) < dev_ratio}
    train = [ex for ex in flat if ex.uttid not in valid_ids]
    valid = [ex for ex in flat if ex.uttid in valid_ids]

    by_class: Dict[Tuple[str, ...], List[Example]] = defaultdict(list)
    for ex in flat:
        by_class[tuple(ex.labels)].append(ex)
    audit_rows = []
    for labels, examples in sorted(by_class.items()):
        n_valid = sum(ex.uttid in valid_ids for ex in examples)
        audit_rows.append(
            (" ".join(labels), len(examples), len(examples) - n_valid, n_valid)
        )
    LOGGER.info(
        "split CS-FLEURS with global hash: train=%d valid=%d dev_ratio=%.4f",
        len(train),
        len(valid),
        dev_ratio,
    )
    return train, valid, audit_rows


def combine_unique(examples: Iterable[Example]) -> List[Example]:
    seen = set()
    out: List[Example] = []
    for ex in examples:
        if ex.uttid in seen:
            raise ValueError(f"duplicate utterance id: {ex.uttid}")
        seen.add(ex.uttid)
        out.append(ex)
    return out


def canonical_pair_class(labels: Sequence[str]) -> str:
    """Return the order-normalized atomic class used by the LID1 baseline."""
    unique = list(dict.fromkeys(labels))
    if len(unique) == 1:
        return unique[0]
    if len(unique) != 2:
        raise ValueError(
            f"atomic pair classification supports one or two labels, got {labels}"
        )
    if "eng" in unique:
        other = unique[0] if unique[1] == "eng" else unique[1]
        return f"{other}-eng"
    return "-".join(sorted(unique))


def to_pair_class_examples(examples: Sequence[Example]) -> List[Example]:
    """Convert individual language sets into one atomic LID1 class."""
    converted = []
    for ex in examples:
        pair_class = canonical_pair_class(ex.labels)
        converted.append(
            Example(
                uttid=ex.uttid,
                wav=ex.wav,
                labels=(pair_class,),
                speaker=ex.speaker,
                source=ex.source,
                subset=ex.subset,
                raw_label_value=ex.raw_label_value,
                raw_label_parts=ex.raw_label_parts,
                duration_sec=ex.duration_sec,
                num_samples=ex.num_samples,
            )
        )
    return converted


def validate_inventories(
    train: Sequence[Example],
    eval_sets: Dict[str, Sequence[Example]],
    allow_unseen: bool,
) -> None:
    train_labels = {lab for ex in train for lab in ex.labels}
    if not train_labels:
        raise ValueError("training label inventory is empty")
    bad: Dict[str, List[str]] = {}
    for name, exs in eval_sets.items():
        labels = {lab for ex in exs for lab in ex.labels}
        unseen = sorted(labels - train_labels)
        if unseen:
            bad[name] = unseen
    if bad and not allow_unseen:
        msg = "Evaluation labels absent from training token inventory: " + json.dumps(
            bad, ensure_ascii=False
        )
        raise ValueError(
            msg
            + ". Use --fleurs_config all or pass --allow_unseen_eval_labels true for diagnostics only."
        )
    if bad:
        LOGGER.warning("unseen eval labels: %s", bad)


def write_label_audits(
    outdir: Path, data_dirs: Dict[str, Sequence[Example]], mapping: Dict[str, str]
) -> None:
    """Write raw-to-canonical audit tables for Codex/manual inspection."""
    local = outdir / "local"
    local.mkdir(parents=True, exist_ok=True)
    with (local / "fleurs_official_label_map.tsv").open("w", encoding="utf-8") as f:
        f.write("raw_fleurs_config\tcanonical_label\ttarget_token\n")
        for cfg in OFFICIAL_FLEURS_CONFIGS:
            canonical = ISO3_ALIASES.get(mapping[cfg], mapping[cfg])
            f.write(f"{cfg}\t{canonical}\t<{canonical}>\n")

    fleurs_counts: Dict[Tuple[str, str, str, str], int] = defaultdict(int)
    cs_counts: Dict[Tuple[str, str, str, str], int] = defaultdict(int)
    for set_name, examples in data_dirs.items():
        for ex in examples:
            if ex.source == "fleurs":
                raw = ex.raw_label_value or (
                    ex.raw_label_parts[0] if ex.raw_label_parts else ""
                )
                fleurs_counts[(set_name, ex.subset, raw, " ".join(ex.labels))] += 1
            elif ex.source == "cs_fleurs":
                cs_counts[
                    (set_name, ex.subset, ex.raw_label_value, " ".join(ex.labels))
                ] += 1

    with (local / "fleurs_observed_label_audit.tsv").open("w", encoding="utf-8") as f:
        f.write("set\tsplit\traw_fleurs_label\tcanonical_label\tnum_utts\n")
        for key, count in sorted(fleurs_counts.items()):
            f.write("\t".join([*key, str(count)]) + "\n")

    with (local / "cs_observed_label_audit.tsv").open("w", encoding="utf-8") as f:
        f.write("set\tsubset\traw_cs_pair\tcanonical_label_sequence\tnum_utts\n")
        for key, count in sorted(cs_counts.items()):
            f.write("\t".join([*key, str(count)]) + "\n")


def write_cs_split_audit(
    outdir: Path,
    rows: Sequence[Tuple[str, int, int, int]],
    split_mode: str,
) -> None:
    local = outdir / "local"
    local.mkdir(parents=True, exist_ok=True)
    with (local / "cs_train_valid_split.tsv").open("w", encoding="utf-8") as f:
        f.write(
            "split_mode\tcanonical_label_sequence\ttotal\ttrain\tvalid\tvalid_ratio\n"
        )
        for label, total, train, valid in rows:
            ratio = 0.0 if total == 0 else valid / total
            f.write(f"{split_mode}\t{label}\t{total}\t{train}\t{valid}\t{ratio:.6f}\n")


def duration_filter_for_set(
    name: str,
    examples: Sequence[Example],
    *,
    min_train_sec: float,
    max_train_sec: float,
    min_eval_sec: float,
    max_eval_sec: float,
    missing_policy: str,
) -> Tuple[List[Example], List[Tuple[str, Example, str]]]:
    """Filter examples by duration before ESPnet sees the data.

    ESPnet ASR's stage-4 duration filter only applies to train/valid; ESPnet LID
    does not have the same train/valid filter in the template.  This explicit
    filter keeps all three planned systems on the same utterance inventory.
    """
    is_eval = name.startswith("test_")
    min_sec = min_eval_sec if is_eval else min_train_sec
    max_sec = max_eval_sec if is_eval else max_train_sec
    kept: List[Example] = []
    removed: List[Tuple[str, Example, str]] = []
    for ex in examples:
        dur = ex.duration_sec
        if dur is None:
            msg = f"missing duration for {name}/{ex.uttid}: {ex.wav}"
            if missing_policy == "error":
                raise ValueError(msg)
            if missing_policy == "warn":
                LOGGER.warning(msg)
            kept.append(ex)
            continue
        if min_sec >= 0 and dur < min_sec:
            removed.append((name, ex, f"too_short<{min_sec}"))
        elif max_sec > 0 and dur >= max_sec:
            removed.append((name, ex, f"too_long>={max_sec}"))
        else:
            kept.append(ex)
    return kept, removed


def apply_duration_filter(
    data_dirs: Dict[str, List[Example]],
    *,
    min_train_sec: float,
    max_train_sec: float,
    min_eval_sec: float,
    max_eval_sec: float,
    missing_policy: str,
) -> Tuple[Dict[str, List[Example]], List[Tuple[str, Example, str]]]:
    out: Dict[str, List[Example]] = {}
    removed_all: List[Tuple[str, Example, str]] = []
    for name, examples in data_dirs.items():
        kept, removed = duration_filter_for_set(
            name,
            examples,
            min_train_sec=min_train_sec,
            max_train_sec=max_train_sec,
            min_eval_sec=min_eval_sec,
            max_eval_sec=max_eval_sec,
            missing_policy=missing_policy,
        )
        out[name] = kept
        removed_all.extend(removed)
        if removed:
            LOGGER.warning(
                "duration filter removed %d/%d utterances from %s",
                len(removed),
                len(examples),
                name,
            )
    return out, removed_all


def _percentile(values: Sequence[float], pct: float) -> Optional[float]:
    xs = sorted(v for v in values if math.isfinite(v))
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * pct / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def write_duration_audit(
    outdir: Path,
    raw_data_dirs: Dict[str, Sequence[Example]],
    kept_data_dirs: Dict[str, Sequence[Example]],
    removed: Sequence[Tuple[str, Example, str]],
) -> None:
    local = outdir / "local"
    local.mkdir(parents=True, exist_ok=True)
    removed_by_set: Dict[str, int] = defaultdict(int)
    for set_name, _ex, _reason in removed:
        removed_by_set[set_name] += 1
    with (local / "duration_summary.tsv").open("w", encoding="utf-8") as f:
        f.write(
            "set\tphase\tnum_utts\tnum_missing_duration\tnum_removed\tmin_sec\tp50_sec\tp95_sec\tp99_sec\tmax_sec\n"
        )
        for phase, dirs in (("raw", raw_data_dirs), ("kept", kept_data_dirs)):
            for name, examples in sorted(dirs.items()):
                durations = [
                    ex.duration_sec for ex in examples if ex.duration_sec is not None
                ]
                missing = sum(1 for ex in examples if ex.duration_sec is None)
                vals = [_percentile(durations, p) for p in (0, 50, 95, 99, 100)]
                vals_s = ["" if v is None else f"{v:.6f}" for v in vals]
                f.write(
                    "\t".join(
                        [
                            name,
                            phase,
                            str(len(examples)),
                            str(missing),
                            str(removed_by_set.get(name, 0) if phase == "raw" else 0),
                            *vals_s,
                        ]
                    )
                    + "\n"
                )
    with (local / "duration_excluded.tsv").open("w", encoding="utf-8") as f:
        f.write(
            "set\tuttid\tsource\tsubset\tduration_sec\tnum_samples\treason\twav\tlabels\traw_label_value\n"
        )
        for set_name, ex, reason in sorted(removed, key=lambda x: (x[0], x[1].uttid)):
            f.write(
                "\t".join(
                    [
                        set_name,
                        ex.uttid,
                        ex.source,
                        ex.subset,
                        "" if ex.duration_sec is None else f"{ex.duration_sec:.6f}",
                        "" if ex.num_samples is None else str(ex.num_samples),
                        reason,
                        ex.wav,
                        " ".join(ex.labels),
                        ex.raw_label_value,
                    ]
                )
                + "\n"
            )


def write_fleurs_overlap_audit(outdir: Path, removed: Sequence[Example]) -> None:
    local = outdir / "local"
    local.mkdir(parents=True, exist_ok=True)
    with (local / "fleurs_train_overlap_removed.tsv").open("w", encoding="utf-8") as f:
        f.write("uttid\twav\tlabels\traw_label_value\tduration_sec\n")
        for ex in sorted(removed, key=lambda e: e.uttid):
            f.write(
                "\t".join(
                    [
                        ex.uttid,
                        ex.wav,
                        " ".join(ex.labels),
                        ex.raw_label_value,
                        "" if ex.duration_sec is None else f"{ex.duration_sec:.6f}",
                    ]
                )
                + "\n"
            )


def write_inventory(
    outdir: Path, data_dirs: Dict[str, Sequence[Example]], mapping: Dict[str, str]
) -> None:
    local = outdir / "local"
    local.mkdir(parents=True, exist_ok=True)
    with (local / "label_inventory.tsv").open("w", encoding="utf-8") as f:
        f.write("set\tlabel\tnum_utts\n")
        for name, examples in sorted(data_dirs.items()):
            counts: Dict[str, int] = defaultdict(int)
            for ex in examples:
                for lab in ex.labels:
                    counts[lab] += 1
            for lab, n in sorted(counts.items()):
                f.write(f"{name}\t{lab}\t{n}\n")
    with (local / "label_map.used.tsv").open("w", encoding="utf-8") as f:
        for k, v in sorted(mapping.items()):
            f.write(f"{k}\t{v}\n")


def split_csv(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def remove_managed_data_dirs(outdir: Path, current_names: Iterable[str]) -> None:
    """Remove only data dirs owned by this recipe before rewriting them."""
    managed = set(current_names)
    managed.update(
        {
            "train_fleurs_lid",
            "valid_fleurs_lid",
            "test_fleurs_lid",
            "train_fleurs_lidseq",
            "valid_fleurs_lidseq",
            "train_lidseq",
            "valid_lidseq",
        }
    )
    if not outdir.exists():
        return
    for path in outdir.iterdir():
        if not path.is_dir():
            continue
        if path.name in managed or path.name.startswith("test_cs_"):
            shutil.rmtree(path)


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--outdir", type=Path, default=Path("data"))
    p.add_argument(
        "--fleurs_config", default="all", help="FLEURS config, e.g. all or en_us,hi_in"
    )
    p.add_argument("--fleurs_cache_dir", default=None)
    p.add_argument(
        "--fleurs_tsv_root",
        type=Path,
        default=None,
        help="TSV dir produced by local/create_fleurs_lid_dataset.py, e.g. ${FLEURS}/all",
    )
    p.add_argument(
        "--fleurs_manifest_root",
        type=Path,
        default=None,
        help="optional local JSONL manifests for tests/offline debug",
    )
    p.add_argument(
        "--fleurs_subsample_per_lang",
        type=int,
        default=0,
        help="debug: keep at most N per language per split",
    )
    p.add_argument("--skip_fleurs", type=str2bool, default=False)
    p.add_argument(
        "--cs_root",
        type=Path,
        default=None,
        help="local git-lfs clone of byan/cs-fleurs",
    )
    p.add_argument("--cs_train_subsets", default="xtts/train")
    p.add_argument(
        "--cs_eval_subsets", default="read/test,xtts/test1,xtts/test2,mms/test"
    )
    p.add_argument("--cs_dev_ratio", type=float, default=0.02)
    p.add_argument(
        "--cs_split_mode",
        choices=("global_hash", "classwise_hash"),
        default="global_hash",
        help="global_hash reproduces the paper split; classwise_hash improves per-class validation coverage",
    )
    p.add_argument("--cs_pair_field", default="language")
    p.add_argument("--allow_single_cs", type=str2bool, default=False)
    p.add_argument("--strict_labels", type=str2bool, default=True)
    p.add_argument("--allow_unseen_eval_labels", type=str2bool, default=False)
    p.add_argument(
        "--require_eval_labels_in_fleurs_lid",
        type=str2bool,
        default=True,
        help="fail if any eval label is absent from FLEURS-only lid1 training inventory",
    )
    p.add_argument(
        "--label_map",
        type=Path,
        default=None,
        help="optional TSV: source_label canonical_label",
    )
    p.add_argument(
        "--nlsyms_txt",
        type=Path,
        default=None,
        help="output canonical target symbols for ESPnet ASR-style LID sequence; default: <outdir>/nlsyms.txt",
    )
    p.add_argument(
        "--verify_fleurs_config_mapping",
        type=str2bool,
        default=True,
        help="verify all 102 raw FLEURS labels have unique canonical labels",
    )
    p.add_argument("--token_format", choices=["plain", "angle"], default="plain")
    p.add_argument(
        "--min_train_duration_sec",
        type=float,
        default=1.0,
        help="filter train/valid utterances shorter than this before ESPnet",
    )
    p.add_argument(
        "--max_train_duration_sec",
        type=float,
        default=30.0,
        help="filter train/valid utterances with duration >= this before ESPnet; <=0 disables",
    )
    p.add_argument(
        "--min_eval_duration_sec",
        type=float,
        default=-1.0,
        help="filter test utterances shorter than this before ESPnet; <0 disables so test is kept unchanged",
    )
    p.add_argument(
        "--max_eval_duration_sec",
        type=float,
        default=0.0,
        help="filter test utterances with duration >= this before ESPnet; <=0 disables so test is kept unchanged",
    )
    p.add_argument(
        "--exclude_fleurs_train_overlaps",
        type=str2bool,
        default=True,
        help="remove FLEURS train recordings also present in validation/test, matching the upstream leakage guard",
    )
    p.add_argument(
        "--duration_missing_policy",
        choices=["warn", "error", "ignore"],
        default="warn",
        help="what to do when duration cannot be inferred",
    )
    p.add_argument("--log_level", default="INFO")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = get_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    if not 0.0 <= args.cs_dev_ratio < 1.0:
        raise ValueError("--cs_dev_ratio must be in [0, 1)")
    mapping = read_label_map(args.label_map)
    if args.verify_fleurs_config_mapping:
        verify_fleurs_config_mapping(mapping)
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.nlsyms_txt is None:
        args.nlsyms_txt = args.outdir / "nlsyms.txt"

    fleurs_train: List[Example] = []
    fleurs_valid: List[Example] = []
    fleurs_test: List[Example] = []
    if not args.skip_fleurs:
        fleurs_train = load_fleurs(
            args.fleurs_config,
            "train",
            mapping,
            args.fleurs_cache_dir,
            args.fleurs_subsample_per_lang,
            args.strict_labels,
            args.fleurs_manifest_root,
            args.fleurs_tsv_root,
        )
        fleurs_valid = load_fleurs(
            args.fleurs_config,
            "validation",
            mapping,
            args.fleurs_cache_dir,
            args.fleurs_subsample_per_lang,
            args.strict_labels,
            args.fleurs_manifest_root,
            args.fleurs_tsv_root,
        )
        fleurs_test = load_fleurs(
            args.fleurs_config,
            "test",
            mapping,
            args.fleurs_cache_dir,
            args.fleurs_subsample_per_lang,
            args.strict_labels,
            args.fleurs_manifest_root,
            args.fleurs_tsv_root,
        )
        if args.exclude_fleurs_train_overlaps:
            fleurs_train, fleurs_overlap_removed = remove_fleurs_train_overlaps(
                fleurs_train, fleurs_valid, fleurs_test
            )
        else:
            fleurs_overlap_removed = []

    cs_train: List[Example] = []
    cs_valid: List[Example] = []
    cs_split_audit_rows: List[Tuple[str, int, int, int]] = []
    cs_eval_sets: Dict[str, List[Example]] = {}
    if args.cs_root is not None:
        cs_train_sets = load_cs_subsets(
            args.cs_root,
            split_csv(args.cs_train_subsets),
            mapping,
            args.cs_pair_field,
            args.strict_labels,
            args.allow_single_cs,
        )
        if args.cs_split_mode == "global_hash":
            cs_train, cs_valid, cs_split_audit_rows = split_cs_train_valid_global_hash(
                cs_train_sets, args.cs_dev_ratio
            )
        else:
            cs_train, cs_valid, cs_split_audit_rows = split_cs_train_valid_by_class(
                cs_train_sets, args.cs_dev_ratio
            )
        cs_eval_sets = load_cs_subsets(
            args.cs_root,
            split_csv(args.cs_eval_subsets),
            mapping,
            args.cs_pair_field,
            args.strict_labels,
            args.allow_single_cs,
        )
    else:
        LOGGER.warning("--cs_root not set: preparing FLEURS-only data dirs")

    data_dirs: Dict[str, List[Example]] = {
        # Closed-set classifier baseline for ESPnet lid1.
        "train_fleurs_lid": combine_unique(fleurs_train),
        "valid_fleurs_lid": combine_unique(fleurs_valid),
        "test_fleurs_lid": combine_unique(fleurs_test),
        # ASR-style encoder-decoder baseline trained on FLEURS only.  These are
        # intentionally separate names, even though the contents are identical
        # to the *_fleurs_lid directories, so experiment logs cannot confuse the
        # classifier baseline with the encoder-decoder baseline.
        "train_fleurs_lidseq": combine_unique(fleurs_train),
        "valid_fleurs_lidseq": combine_unique(fleurs_valid),
        # ASR-style encoder-decoder model trained on FLEURS + CS-FLEURS.
        "train_lidseq": combine_unique([*fleurs_train, *cs_train]),
        "valid_lidseq": combine_unique([*fleurs_valid, *cs_valid]),
    }

    cs_all_eval: List[Example] = []
    for subset, exs in cs_eval_sets.items():
        name = "test_cs_" + sanitize_id(subset.replace("/", "_"))
        data_dirs[name] = combine_unique(exs)
        cs_all_eval.extend(exs)
    if cs_all_eval:
        data_dirs["test_cs_all"] = combine_unique(cs_all_eval)

    raw_data_dirs = {k: list(v) for k, v in data_dirs.items()}
    data_dirs, duration_removed = apply_duration_filter(
        data_dirs,
        min_train_sec=args.min_train_duration_sec,
        max_train_sec=args.max_train_duration_sec,
        min_eval_sec=args.min_eval_duration_sec,
        max_eval_sec=args.max_eval_duration_sec,
        missing_policy=args.duration_missing_policy,
    )

    eval_sets = {k: v for k, v in data_dirs.items() if k.startswith("test_")}
    validate_inventories(
        data_dirs["train_lidseq"], eval_sets, args.allow_unseen_eval_labels
    )
    if args.require_eval_labels_in_fleurs_lid:
        if data_dirs["train_fleurs_lid"]:
            validate_inventories(
                data_dirs["train_fleurs_lid"], eval_sets, args.allow_unseen_eval_labels
            )
        elif eval_sets and not args.skip_fleurs:
            raise ValueError("FLEURS-only lid1 inventory is empty but eval sets exist")
        elif eval_sets:
            LOGGER.warning(
                "skip FLEURS-only lid1 inventory validation because --skip_fleurs true"
            )

    # LID1 treats each two-language set as one atomic class.  Build this view
    # after the shared source-level duration filter, so all systems consume the
    # same utterance inventory.  These pair labels must not enter ASR nlsyms.
    pair_data_dirs = {
        "train_lid_pair_cs": to_pair_class_examples(data_dirs["train_lidseq"]),
        "valid_lid_pair_cs": to_pair_class_examples(data_dirs["valid_lidseq"]),
        "train_lid_pair_fleurs": to_pair_class_examples(data_dirs["train_fleurs_lid"]),
        "valid_lid_pair_fleurs": to_pair_class_examples(data_dirs["valid_fleurs_lid"]),
    }
    pair_test_names = {
        "test_fleurs_lid": "test_lid_pair_fleurs",
        "test_cs_read_test": "test_lid_pair_cs_read_test",
        "test_cs_xtts_test1": "test_lid_pair_cs_xtts_test1",
        "test_cs_xtts_test2": "test_lid_pair_cs_xtts_test2",
        "test_cs_mms_test": "test_lid_pair_cs_mms_test",
        "test_cs_all": "test_lid_pair_cs_all",
    }
    for source_name, pair_name in pair_test_names.items():
        if source_name in data_dirs:
            pair_data_dirs[pair_name] = to_pair_class_examples(data_dirs[source_name])
    data_dirs.update(pair_data_dirs)

    remove_managed_data_dirs(args.outdir, data_dirs.keys())
    nonempty = {k: v for k, v in data_dirs.items() if v}
    for name, exs in nonempty.items():
        write_data_dir(name, exs, args.outdir, args.token_format)
    for name in sorted(set(data_dirs) - set(nonempty)):
        LOGGER.warning("skip empty data dir: %s", name)
    sequence_dirs = {
        name: examples
        for name, examples in nonempty.items()
        if "_lid_pair_" not in name
    }
    write_nlsyms(sequence_dirs, args.token_format, args.nlsyms_txt)
    write_inventory(args.outdir, nonempty, mapping)
    write_label_audits(args.outdir, nonempty, mapping)
    write_cs_split_audit(args.outdir, cs_split_audit_rows, args.cs_split_mode)
    write_duration_audit(args.outdir, raw_data_dirs, nonempty, duration_removed)
    write_fleurs_overlap_audit(args.outdir, locals().get("fleurs_overlap_removed", []))
    LOGGER.info(
        "done. label inventory: %s", args.outdir / "local" / "label_inventory.tsv"
    )
    LOGGER.info(
        "done. label audit: %s", args.outdir / "local" / "fleurs_official_label_map.tsv"
    )
    LOGGER.info(
        "done. duration audit: %s", args.outdir / "local" / "duration_summary.tsv"
    )


if __name__ == "__main__":
    main()
