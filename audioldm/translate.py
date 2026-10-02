"""日本語プロンプトの英訳。

AudioLDM は英語プロンプトを受け取るが、ユーザスタディの被験者は日本語話者である。
被験者に英語で書かせると、測っているものが「言語化の困難」ではなく
**「英作文の困難」**になってしまう（本研究の問いと無関係な交絡）。

対処: 日本語で入力させ、自動翻訳した英文を生成に用いる。
**かつ訳文を画面に表示する。** 表示することで、被験者が「意図と違う訳になった」と
気づいて書き直せるようになり、翻訳は交絡ではなく**利用者が制御できる道具**になる。
ログには原文（prompt_ja）と訳文（prompt_en）の両方を残し、事後に翻訳品質を検証できるようにする。

バックエンド:
  - "google" : Google Cloud Translation API v2（**ユーザスタディ推奨**）。
               環境変数 `GOOGLE_TRANSLATE_API_KEY` が必要。ネットワークが必要。
               逆翻訳（en→ja）にも対応するので、英語が読めない被験者でも訳の妥当性を確認できる。
  - "marian" : ローカルの Helsinki-NLP/opus-mt-ja-en。ネットワーク不要だが**品質が不十分**。
               「緊張感のある弦楽器とブラスのファンファーレ」を誤訳する実例を確認済み。
               `pip install sentencepiece sacremoses` が必要。
  - "none"   : 素通し（入力をそのまま返す）。英語で入力する場合や、翻訳が使えない環境用。
  - "auto"   : google → marian → none の順に使えるものを選ぶ（既定）。

使用例:
    from audioldm.translate import get_translator
    tr = get_translator("google")
    en, info = tr.translate("暖かくて少し寂しいピアノ")
    ja = tr.back_translate(en)       # 逆翻訳（確認用。未対応なら None）
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

MARIAN_MODEL = "Helsinki-NLP/opus-mt-ja-en"

# ひらがな・カタカナ・漢字のいずれかを含むか
_JA_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿]")


def looks_japanese(text: str) -> bool:
    """日本語を含むか。英語で書かれた入力は翻訳せずそのまま通すために使う。"""
    return bool(_JA_RE.search(text or ""))


class Translator:
    """翻訳バックエンドの共通インタフェース。"""

    name = "base"
    available = False

    def translate(self, text: str) -> Tuple[str, dict]:
        """(英文, 付帯情報) を返す。失敗しても例外を投げず原文を返すこと。"""
        raise NotImplementedError

    def back_translate(self, text_en: str) -> Optional[str]:
        """英文を日本語に訳し返す（確認用）。未対応なら None を返す。

        訳文が意図と合っているかを、英語が読めない被験者でも判断できるようにするため。
        """
        return None


class PassthroughTranslator(Translator):
    """素通し。英語入力や、翻訳が使えない環境用。"""

    name = "none"
    available = True

    def translate(self, text: str) -> Tuple[str, dict]:
        return text, {"backend": self.name, "translated": False}


class MarianTranslator(Translator):
    """ローカルの MarianMT（Helsinki-NLP/opus-mt-ja-en）による日→英翻訳。"""

    name = "marian"

    def __init__(self, model_name: str = MARIAN_MODEL, device: Optional[str] = None):
        self.model_name = model_name
        self._tok = None
        self._model = None
        self._device = device
        self.available = False
        try:
            from transformers import MarianMTModel, MarianTokenizer  # noqa: F401
            import sentencepiece  # noqa: F401
            self.available = True
        except ImportError as e:
            self._import_error = str(e)

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import MarianMTModel, MarianTokenizer

        print(f"[翻訳] モデルをロード中: {self.model_name}")
        self._tok = MarianTokenizer.from_pretrained(self.model_name)
        self._model = MarianMTModel.from_pretrained(self.model_name)
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device).eval()
        print(f"[翻訳] 準備完了 (device={self._device})")

    def translate(self, text: str) -> Tuple[str, dict]:
        text = (text or "").strip()
        if not text:
            return "", {"backend": self.name, "translated": False}
        # 英語で書かれていればそのまま通す（被験者が英語で書くことを妨げない）
        if not looks_japanese(text):
            return text, {"backend": self.name, "translated": False,
                          "reason": "not_japanese"}
        try:
            import torch

            self._load()
            batch = self._tok([text], return_tensors="pt", padding=True).to(self._device)
            with torch.no_grad():
                out = self._model.generate(**batch, max_new_tokens=128)
            en = self._tok.decode(out[0], skip_special_tokens=True).strip()
            return en, {"backend": self.name, "translated": True}
        except Exception as e:  # 翻訳の失敗でセッションを止めない
            print(f"[翻訳] 失敗のため原文を使用します: {e}")
            return text, {"backend": self.name, "translated": False,
                          "error": str(e)}


class GoogleTranslator(Translator):
    """Google Cloud Translation API v2 による日→英翻訳。

    API キーは引数 `api_key`、無ければ環境変数 `GOOGLE_TRANSLATE_API_KEY` から読む。
    同じ文の再翻訳はキャッシュから返すので、書き換えの繰り返しでも課金は増えない。
    料金は 100 万文字あたり 20 USD 程度で、ユーザスタディの規模では実質無視できる。
    """

    name = "google"
    ENDPOINT = "https://translation.googleapis.com/language/translate/v2"

    def __init__(self, api_key: Optional[str] = None, timeout: float = 10.0):
        import os

        self.api_key = api_key or os.environ.get("GOOGLE_TRANSLATE_API_KEY")
        self.timeout = timeout
        self._cache: dict = {}
        self.available = bool(self.api_key)
        if not self.available:
            self._import_error = "環境変数 GOOGLE_TRANSLATE_API_KEY が未設定"

    def _call(self, text: str, source: str, target: str) -> Optional[str]:
        import requests

        key = (text, source, target)
        if key in self._cache:
            return self._cache[key]
        resp = requests.post(
            self.ENDPOINT,
            params={"key": self.api_key},
            json={"q": text, "source": source, "target": target, "format": "text"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        out = resp.json()["data"]["translations"][0]["translatedText"].strip()
        self._cache[key] = out
        return out

    def translate(self, text: str) -> Tuple[str, dict]:
        text = (text or "").strip()
        if not text:
            return "", {"backend": self.name, "translated": False}
        if not looks_japanese(text):
            return text, {"backend": self.name, "translated": False,
                          "reason": "not_japanese"}
        try:
            en = self._call(text, "ja", "en")
            return en, {"backend": self.name, "translated": True}
        except Exception as e:  # 翻訳の失敗でセッションを止めない
            print(f"[翻訳] Google API 失敗のため原文を使用します: {e}")
            return text, {"backend": self.name, "translated": False, "error": str(e)}

    def back_translate(self, text_en: str) -> Optional[str]:
        text_en = (text_en or "").strip()
        if not text_en:
            return None
        try:
            return self._call(text_en, "en", "ja")
        except Exception as e:
            print(f"[翻訳] 逆翻訳に失敗: {e}")
            return None


def get_translator(backend: str = "auto", **kwargs) -> Translator:
    """バックエンド名から翻訳器を作る。

    "auto" は marian を試し、依存が無ければ素通しにフォールバックする。
    ユーザスタディでは**素通しに落ちていないことを起動ログで必ず確認すること**
    （落ちたまま実施すると、日本語がそのまま AudioLDM に入り生成が破綻する）。
    """
    if backend == "none":
        return PassthroughTranslator()

    if backend in ("google", "auto"):
        tr = GoogleTranslator(**{k: v for k, v in kwargs.items()
                                 if k in ("api_key", "timeout")})
        if tr.available:
            return tr
        msg = f"[翻訳] Google Translation API を使えません（{getattr(tr, '_import_error', '')}）"
        if backend == "google":
            raise RuntimeError(
                msg + "\n  export GOOGLE_TRANSLATE_API_KEY=... を設定してください")
        print(msg + " → marian を試します")

    if backend in ("marian", "auto"):
        tr = MarianTranslator(**{k: v for k, v in kwargs.items()
                                 if k in ("model_name", "device")})
        if tr.available:
            return tr
        msg = f"[翻訳] {MARIAN_MODEL} を使えません（{getattr(tr, '_import_error', '')}）"
        if backend == "marian":
            raise RuntimeError(
                msg + "\n  pip install sentencepiece sacremoses を実行してください")
        print(msg + " → 素通しにフォールバックします")
        return PassthroughTranslator()

    raise ValueError(f"未知の翻訳バックエンド: {backend}")
