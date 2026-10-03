# -*- coding: utf-8 -*-
# Google Colab notebook cell: Myanmar Voice Clone with VoxCPM2
# Run this in Google Colab with Runtime > Change runtime type > T4 GPU.

import os, re, sys, shutil, subprocess, tempfile, traceback
import importlib.metadata as md

# 1. Stop immediately if NVIDIA GPU tooling is unavailable.
if shutil.which("nvidia-smi") is None:
    raise SystemExit(
        "❌ GPU မရှိပါ။ Colab မှာ Runtime → Change runtime type → T4 GPU ရွေးပြီး ပြန် Run ပါ။"
    )
print("✅ GPU တွေ့ပါပြီ")

# 2. Preserve the versions of key packages already installed in Colab.
pins = []
for n in ["numpy", "torch", "torchaudio", "torchvision", "scipy", "numba", "pandas", "scikit-learn"]:
    try:
        pins.append(f"{n}=={md.version(n)}")
    except Exception:
        pass

with open("constraints.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(pins))

print("🔧 Package တပ်ဆင်နေပါတယ်…")
r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-c", "constraints.txt",
     "voxcpm", "gradio", "soundfile"],
    capture_output=True, text=True
)
if r.returncode != 0:
    print(r.stderr[-3000:])
    raise SystemExit("❌ Package တပ်ဆင်မှု မအောင်မြင်ပါ။ အပေါ်က error ကို စစ်ပါ။")

import numpy as np
import soundfile as sf
import torch
import gradio as gr
from voxcpm import VoxCPM

print("🚀 Device:", "CUDA" if torch.cuda.is_available() else "CPU")
if not torch.cuda.is_available():
    raise SystemExit("❌ torch က GPU မမြင်ပါ။ Colab မှာ T4 GPU ရွေးထားကြောင်း စစ်ပါ။")

print("⏳ VoxCPM2 model load နေပါတယ်…")
model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
SR = getattr(model.tts_model, "sample_rate", 48000)
print("✅ Model အဆင်သင့်")

def split(text, n=250):
    out, cur = [], ""
    for p in re.split(r"(?<=[။?!.\n])", text):
        if len(cur) + len(p) > n and cur:
            out.append(cur.strip())
            cur = ""
        if len(p) > n:
            for i in range(0, len(p), n):
                out.append(p[i:i+n].strip())
        else:
            cur += p
    if cur.strip():
        out.append(cur.strip())
    return [x for x in out if x]

NAMES = [
    "နမူနာအသံ စစ်ဆေးခြင်း",
    "စာကို အပိုင်းခွဲခြင်း",
    "Clone လုပ်ပြီး အသံထုတ်ခြင်း",
    "MP3 ပြောင်းခြင်း",
    "ပြီးစီးပါပြီ",
]
ICON = {"wait": "○", "run": "⏳", "done": "✅", "err": "❌"}

def view(st, d):
    return "\n".join(
        f"{ICON[s]} {i+1}. {name}" + (f" — {d[i]}" if i in d else "")
        for i, (s, name) in enumerate(zip(st, NAMES))
    )

def run(sample, text):
    st, d = ["wait"] * 5, {}
    if not sample or not text or not text.strip():
        yield "⚠️ နမူနာအသံနဲ့ စာ နှစ်ခုလုံး ထည့်ပါ", None, None
        return

    w = tempfile.mkdtemp(prefix="myanmar_voice_clone_")
    try:
        st[0] = "run"
        yield view(st, d), None, None

        ref = os.path.join(w, "ref.wav")
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", sample, "-t", "30", "-ac", "1", "-ar", "44100", ref],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            raise RuntimeError("ffmpeg audio conversion failed: " + r.stderr[-500:])

        st[0] = "done"
        st[1] = "run"
        yield view(st, d), None, None

        parts = split(text)
        st[1] = "done"
        d[1] = f"{len(parts)} အပိုင်း"
        st[2] = "run"
        out = []

        for i, part in enumerate(parts):
            d[2] = f"{i+1}/{len(parts)}"
            yield view(st, d), None, None
            wav = model.generate(
                text=part,
                reference_wav_path=ref,
                cfg_value=2.0,
                inference_timesteps=10
            )
            out.append(np.asarray(wav, dtype=np.float32).reshape(-1))
            out.append(np.zeros(int(SR * 0.15), dtype=np.float32))

        st[2] = "done"
        st[3] = "run"
        yield view(st, d), None, None

        wp = os.path.join(w, "out.wav")
        sf.write(wp, np.concatenate(out), SR)

        mp = os.path.join(w, "voice-clone.mp3")
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", wp, "-b:a", "192k", mp],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            raise RuntimeError("MP3 conversion failed: " + r.stderr[-500:])

        st[3] = "done"
        st[4] = "done"
        yield view(st, d), mp, mp

    except Exception:
        if "run" in st:
            st[st.index("run")] = "err"
        yield view(st, d) + "\n\n" + traceback.format_exc()[-1000:], None, None

with gr.Blocks(title="မြန်မာ Voice Clone") as demo:
    gr.Markdown(
        "# 🎙️ မြန်မာ Voice Clone\n"
        "ကိုယ့်အသံ (သို့) ခွင့်ပြုချက်ရထားတဲ့ အသံကိုသာ သုံးပါ။\n\n"
        "**မှတ်ချက်:** Google Colab GPU လိုအပ်ပြီး ပထမဆုံးအကြိမ် model download လုပ်ရပါမယ်။"
    )
    a = gr.Audio(
        sources=["upload", "microphone"],
        type="filepath",
        label="1. Clone ချင်တဲ့ အသံ (နားထောင်ကြည့်လို့ရ)"
    )
    t = gr.Textbox(
        lines=8,
        label="2. ထွက်စေချင်တဲ့ စာ (စာကြောင်းဆုံးတိုင်း ။ ထည့်ပါ)"
    )
    b = gr.Button("▶ 3. စတင်မည်", variant="primary")
    s = gr.Textbox(lines=7, label="4. လုပ်ဆောင်ချက် အဆင့်ဆင့်", interactive=False)
    o = gr.Audio(type="filepath", label="5. ရလဒ် — နားထောင်ရန်")
    f = gr.File(label="⬇ MP3 ဒေါင်းလုဒ်")
    b.click(run, [a, t], [s, o, f])

demo.queue().launch(share=True, debug=True)
