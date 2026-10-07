"""生成バックエンド（AudioLDM / MusicLDM）の抽象化。

本手法が生成モデルに要求するのは、条件ベクトル $c$ と初期ノイズ $x_T$ の2入力へ
**独立に介入できること**だけである。その境界を4つの操作に絞って抽象化し、
AudioLDM と MusicLDM を切り替えられるようにする。

    latent_shape      : (C, T, F) — x_T を作るための形
    encode_text       : プロンプト → CLAP text embedding (1, 1, 512)
    get_unconditional : CFG 用の無条件条件（バックエンドが内部で作る場合は None）
    generate          : (c のバッチ, x_T のバッチ) → 波形のリスト

**Phase A の制約（2026-10-07）**
  MusicLDM バックエンドは diffusers の `MusicLDMPipeline.__call__` を使う。
  これは `latents` と `prompt_embeds` を受けるので2入力介入は成立するが、
  生成途中の潜在を取り出せないため次の2つが使えない。
    - 潜在のキャッシュ（`genotype.z0`）
    - 区間インペインティング（バリエーションガチャ）
  どちらも評価対象ではないので Phase A では落とす。
  Phase B でコンポーネント（unet/vae/text_encoder/vocoder/scheduler）を直接使う
  自前 denoise ループに差し替えれば両方戻せる。この抽象化の境界は変わらない。

**`MusicLDMPipeline` は diffusers 0.33.1 で deprecated** であり、バグ修正も機能追加も
入らない。将来削除される可能性があるため `requirements_iec.txt` でバージョンを固定する。
壊れているわけではなく（スパイクで動作確認済み）、非推奨なのはパイプラインという
薄いラッパーだけで、中身の UNet / VAE / CLAP / ボコーダは標準クラスである。
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
import torch

# 各バックエンドの訓練時の潜在時間長（= 既定の音声長）。
# 訓練長の 1/4 などで生成すると分布から外れて品質が落ちる（2026-10-06 に確認）。
TRAINING_DURATION = {
    "audioldm": 10.0,    # latent_t_size=256 / 25.6 フレーム毎秒
    "musicldm": 10.24,   # unet.config.sample_size=256 / 25 フレーム毎秒
}


class GenerationBackend:
    """生成バックエンドの共通インタフェース。"""

    name = "base"
    supports_inpaint = False      # 区間インペインティング（バリエーションガチャ）
    supports_latent_cache = False  # genotype.z0 のキャッシュ

    @property
    def latent_shape(self) -> Tuple[int, int, int]:
        raise NotImplementedError

    def set_duration(self, seconds: float) -> None:
        raise NotImplementedError

    def encode_text(self, prompt: str) -> torch.Tensor:
        """プロンプト → CLAP text embedding (1, 1, 512)。L2 正規化済み。"""
        raise NotImplementedError

    def generate(
        self,
        c_batch: torch.Tensor,            # (N, 1, 512)
        x_T_batch: torch.Tensor,          # (N, C, T, F)
        inpaint_x0: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[List[np.ndarray], Optional[torch.Tensor]]:
        """(波形のリスト, 潜在 or None) を返す。"""
        raise NotImplementedError


class AudioLDMBackend(GenerationBackend):
    """既存の AudioLDM ネイティブ実装をそのまま使うバックエンド。

    `latent_diffusion` を公開するので、conditioning モード以外の経路
    （latent GA / transform GA / スタイル転送 / 長尺生成）は従来どおり動く。
    """

    name = "audioldm"
    supports_inpaint = True
    supports_latent_cache = True

    def __init__(self, model_name: str, device: str, duration: float,
                 guidance_scale: float, ddim_steps: int,
                 ckpt_path: Optional[str] = None):
        from audioldm.pipeline import build_model, duration_to_latent_t_size

        self.model_name = model_name
        self.device = device
        self.guidance_scale = guidance_scale
        self.ddim_steps = ddim_steps
        self._dur_to_T = duration_to_latent_t_size

        print(f"AudioLDM モデルをロード中: {model_name}")
        self.latent_diffusion = build_model(ckpt_path=ckpt_path, model_name=model_name)
        self.latent_diffusion = self.latent_diffusion.to(device)
        self.latent_diffusion.eval()
        self.latent_diffusion.cond_stage_model.embed_mode = "text"
        self.set_duration(duration)

    @property
    def latent_shape(self) -> Tuple[int, int, int]:
        ld = self.latent_diffusion
        return (ld.channels, ld.latent_t_size, ld.latent_f_size)

    def set_duration(self, seconds: float) -> None:
        self.duration = seconds
        self.latent_diffusion.latent_t_size = self._dur_to_T(seconds)

    def encode_text(self, prompt: str) -> torch.Tensor:
        cond = self.latent_diffusion.cond_stage_model
        orig_mode, orig_prob = cond.embed_mode, cond.unconditional_prob
        cond.embed_mode, cond.unconditional_prob = "text", 0.0
        try:
            with torch.no_grad():
                return cond([prompt, prompt])[0:1].clone()   # (1, 1, 512)
        finally:
            cond.embed_mode, cond.unconditional_prob = orig_mode, orig_prob

    def generate(self, c_batch, x_T_batch, inpaint_x0=None, mask=None):
        n = c_batch.shape[0]
        ld = self.latent_diffusion
        with ld.ema_scope("Conditioning Batch"):
            with torch.no_grad():
                uc = ld.cond_stage_model.get_unconditional_condition(n)
                samples, _ = ld.sample_log(
                    cond=c_batch, batch_size=n, ddim=True,
                    ddim_steps=self.ddim_steps, eta=0.0,
                    unconditional_guidance_scale=self.guidance_scale,
                    unconditional_conditioning=uc,
                    x_T=x_T_batch, mask=mask, x0=inpaint_x0,
                )
                if torch.max(torch.abs(samples)) > 1e2:
                    samples = torch.clip(samples, min=-10, max=10)
                mel = ld.decode_first_stage(samples)
                wf = ld.mel_spectrogram_to_waveform(mel)
        return [wf[i:i + 1] for i in range(n)], samples.detach()


class MusicLDMBackend(GenerationBackend):
    """diffusers の MusicLDMPipeline を使うバックエンド（Phase A）。

    `prompt_embeds`（= $c$）と `latents`（= $x_T$）の両方を受け取れるので
    2入力への独立介入が成立する。CFG の無条件条件はパイプラインが内部で作る。
    """

    name = "musicldm"
    supports_inpaint = False        # __call__ では各ステップの潜在に触れない
    supports_latent_cache = False

    DEFAULT_MODEL = "ucsd-reach/musicldm"

    def __init__(self, model_name: str, device: str, duration: float,
                 guidance_scale: float, ddim_steps: int,
                 ckpt_path: Optional[str] = None):
        from diffusers import MusicLDMPipeline

        if model_name in (None, "", "musicldm") or "audioldm" in model_name:
            model_name = self.DEFAULT_MODEL
        self.model_name = model_name
        self.device = device
        self.guidance_scale = guidance_scale
        self.ddim_steps = ddim_steps

        print(f"MusicLDM モデルをロード中: {model_name}")
        self.pipe = MusicLDMPipeline.from_pretrained(
            model_name, torch_dtype=torch.float32)
        self.pipe = self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)

        vc = self.pipe.vocoder.config
        self.sampling_rate = vc.sampling_rate
        # 1 mel フレームあたりの秒数。潜在の時間長は mel フレーム数 / vae_scale_factor
        self._sec_per_mel = float(np.prod(vc.upsample_rates)) / vc.sampling_rate
        self._vae_scale = self.pipe.vae_scale_factor
        self._channels = self.pipe.unet.config.in_channels
        self._f = vc.model_in_dim // self._vae_scale
        self.set_duration(duration)

        train_T = self.pipe.unet.config.sample_size
        print(f"  訓練時の潜在時間長 T={train_T} "
              f"（= {self._T_to_seconds(train_T):.2f} 秒）")
        if self._T != train_T:
            print(f"  ⚠ 現在の設定は T={self._T}（{self.duration} 秒）。"
                  f"訓練長から外れると品質が落ちる")

    # --- 音声長と潜在形状 ---
    def _seconds_to_T(self, seconds: float) -> int:
        return int(round((seconds / self._sec_per_mel) / self._vae_scale))

    def _T_to_seconds(self, T: int) -> float:
        return T * self._vae_scale * self._sec_per_mel

    @property
    def latent_shape(self) -> Tuple[int, int, int]:
        return (self._channels, self._T, self._f)

    def set_duration(self, seconds: float) -> None:
        self.duration = seconds
        self._T = self._seconds_to_T(seconds)

    # --- 2入力 ---
    def encode_text(self, prompt: str) -> torch.Tensor:
        tok = self.pipe.tokenizer(
            [prompt], padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            emb = self.pipe.text_encoder.get_text_features(**tok)   # (1, 512) 正規化済み
        return emb.unsqueeze(1).clone()                              # (1, 1, 512)

    def generate(self, c_batch, x_T_batch, inpaint_x0=None, mask=None):
        if inpaint_x0 is not None or mask is not None:
            raise NotImplementedError(
                "MusicLDM バックエンド（Phase A）は区間インペインティングに未対応です。"
                "バリエーションガチャは AudioLDM バックエンドでのみ使えます。"
            )
        n = c_batch.shape[0]
        # IEC 側は (N,1,512) で持つが、パイプラインは (N,512) を要求する
        prompt_embeds = c_batch.reshape(n, -1).to(self.device)
        with torch.no_grad():
            out = self.pipe(
                prompt_embeds=prompt_embeds,
                latents=x_T_batch.to(self.device),
                num_inference_steps=self.ddim_steps,
                guidance_scale=self.guidance_scale,
                audio_length_in_s=self.duration,
            )
        audios = out.audios                      # (N, samples)
        return [audios[i:i + 1] for i in range(n)], None


def build_backend(backend: str, model_name: str, device: str,
                  duration: Optional[float], guidance_scale: float,
                  ddim_steps: int, ckpt_path: Optional[str] = None
                  ) -> GenerationBackend:
    """バックエンド名から生成バックエンドを作る。

    duration が None の場合は、そのバックエンドの**訓練長**を既定にする。
    """
    backend = (backend or "audioldm").lower()
    if duration is None:
        duration = TRAINING_DURATION[backend]
    cls = {"audioldm": AudioLDMBackend, "musicldm": MusicLDMBackend}.get(backend)
    if cls is None:
        raise ValueError(f"未知のバックエンド: {backend}（audioldm / musicldm）")
    return cls(model_name=model_name, device=device, duration=duration,
               guidance_scale=guidance_scale, ddim_steps=ddim_steps,
               ckpt_path=ckpt_path)
