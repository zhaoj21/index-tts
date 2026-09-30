"""Clone a reference WAV voice and synthesize SRT subtitles into one WAV.

Run ``python infer_srt.py --help`` for options. Model imports are deliberately
lazy: help, input validation and unit tests do not load the TTS models.
"""

import argparse
from dataclasses import dataclass
import html
import os
from pathlib import Path
import re
import sys
import tempfile
import wave


@dataclass(frozen=True)
class Cue:
    index: int
    start_ms: int
    end_ms: int
    text: str


_TIME = r"(\d{2,}):([0-5]\d):([0-5]\d)[,.](\d{3})"
_TIMING = re.compile(rf"^{_TIME}\s*-->\s*{_TIME}$")
_TAGS = re.compile(r"</?(?:b|i|u|s|font|span)\b[^>]*>", re.IGNORECASE)


def _milliseconds(parts):
    hours, minutes, seconds, millis = map(int, parts)
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis


def parse_srt(content):
    """Parse numbered SRT cues; reject malformed input instead of reading times aloud."""
    content = content.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not content:
        raise ValueError("SRT 文件为空")
    cues = []
    for block_number, block in enumerate(re.split(r"\n[ \t]*\n+", content), 1):
        lines = block.strip().splitlines()
        if len(lines) < 3 or not lines[0].strip().isdigit():
            raise ValueError(f"SRT 第 {block_number} 块缺少序号、时间轴或正文")
        timing = _TIMING.fullmatch(lines[1].strip())
        if timing is None:
            raise ValueError(f"SRT 第 {block_number} 块时间轴无效: {lines[1]}")
        start_ms = _milliseconds(timing.groups()[:4])
        end_ms = _milliseconds(timing.groups()[4:])
        if end_ms <= start_ms:
            raise ValueError(f"SRT 第 {block_number} 块结束时间必须晚于开始时间")
        text = html.unescape(_TAGS.sub("", " ".join(line.strip() for line in lines[2:])))
        text = text.strip()
        if not text:
            raise ValueError(f"SRT 第 {block_number} 块正文为空")
        cues.append(Cue(int(lines[0]), start_ms, end_ms, text))
    return cues


def validate_timeline(cues):
    previous_end = 0
    for cue in cues:
        if cue.start_ms < previous_end:
            raise ValueError(
                f"字幕 {cue.index} 时间倒序或重叠；请修正 SRT，或使用 --timing concat 忽略时间轴"
            )
        previous_end = cue.end_ms


def build_parser():
    parser = argparse.ArgumentParser(
        description="使用 WAV 音色参考和 SRT 字幕生成配音 WAV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--voice", type=Path, required=True, help="音色参考 WAV（不是待转写音频）")
    parser.add_argument("--srt", type=Path, required=True, help="字幕文件")
    parser.add_argument("--output", type=Path, required=True, help="输出 WAV 路径")
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent / "checkpoints")
    parser.add_argument("--version", choices=("2", "2.5"), default="2.5")
    parser.add_argument("--device", default="auto", help="auto、cpu、cuda:0、mps 或 xpu")
    parser.add_argument("--lang", choices=("ZH", "EN", "JA", "AR", "ES"), default="ZH", help="v2.5 语言")
    parser.add_argument("--encoding", default="utf-8-sig", help="SRT 编码，例如 gb18030")
    parser.add_argument("--timing", choices=("srt", "concat"), default="srt", help="按字幕时间轴对齐，或按文件顺序拼接")
    parser.add_argument("--overflow", choices=("speed-up", "error"), default="speed-up", help="语音超过字幕时长时保音高加速，或报错；仅 srt 模式")
    parser.add_argument("--gap-ms", type=int, default=200, help="concat 模式字幕间静音毫秒数")
    parser.add_argument("--max-text-tokens", type=int, default=120, help="模型内部每段最大文本 token 数")
    parser.add_argument("--num-beams", type=int, default=1, help="解码 beam 数，1 较快")
    parser.add_argument("--half", action="store_true", help="半精度：v2 使用 FP16，v2.5 使用 BF16；CPU 不启用")
    parser.add_argument("--overwrite", action="store_true", help="允许替换已存在的输出文件（不会替换输入文件）")
    parser.add_argument("--verbose", action="store_true", help="显示模型详细日志和错误堆栈")
    return parser


def validate_inputs(args):
    for path, suffix in ((args.voice, ".wav"), (args.srt, ".srt")):
        if not path.is_file():
            raise ValueError(f"输入文件不存在: {path}")
        if path.suffix.lower() != suffix:
            raise ValueError(f"输入必须为 {suffix} 文件: {path}")
    if args.output.suffix.lower() != ".wav":
        raise ValueError("输出文件扩展名必须为 .wav")
    for source in (args.voice, args.srt):
        if args.output.resolve() == source.resolve() or (
            args.output.exists() and os.path.samefile(args.output, source)
        ):
            raise ValueError("输出路径不能覆盖输入文件")
    if args.output.exists() and (not args.overwrite or not args.output.is_file()):
        raise ValueError(f"输出路径已存在（替换文件需 --overwrite）: {args.output}")
    if args.gap_ms < 0 or args.max_text_tokens < 1 or args.num_beams < 1:
        raise ValueError("gap-ms 不能为负，max-text-tokens 和 num-beams 必须大于 0")
    if args.version == "2" and args.lang not in ("ZH", "EN"):
        raise ValueError("v2 仅支持 ZH/EN；其他语言请使用 --version 2.5")
    cues = parse_srt(args.srt.read_text(encoding=args.encoding))
    if args.timing == "srt":
        validate_timeline(cues)
    # Check primary files without triggering a large download on a typo.
    required = ["config.yaml", "gpt.pth", "s2mel.pth", "wav2vec2bert_stats.pt"]
    required += ["codec.pth", "multilingual_zh_ja_yue_char_del.tiktoken"] if args.version == "2.5" else ["bpe.model"]
    missing = [name for name in required if not (args.model_dir / name).is_file()]
    if missing:
        raise ValueError(f"v{args.version} 模型目录不完整: {args.model_dir}；缺少 {', '.join(missing)}")
    return cues


def load_tts(args):
    if args.version == "2.5":
        from indextts.infer_v2_5 import IndexTTS2
    else:
        from indextts.infer_v2 import IndexTTS2

    kwargs = {
        "cfg_path": str(args.model_dir / "config.yaml"),
        "model_dir": str(args.model_dir),
        "device": None if args.device == "auto" else args.device,
        "use_cuda_kernel": False,
        "use_deepspeed": False,
        "use_accel": False,
        "use_torch_compile": False,
        "use_qwen_emo": False,
        "use_bf16" if args.version == "2.5" else "use_fp16": args.half,
    }
    return IndexTTS2(**kwargs)


def read_pcm(path):
    import numpy as np

    with wave.open(str(path), "rb") as reader:
        if reader.getsampwidth() != 2 or reader.getnchannels() != 1 or reader.getcomptype() != "NONE":
            raise ValueError(f"模型输出必须为单声道 PCM16 WAV: {path}")
        rate = reader.getframerate()
        frames = reader.getnframes()
        raw = reader.readframes(frames)
    if not frames or len(raw) != frames * 2:
        raise ValueError(f"模型输出音频为空或不完整: {path}")
    return rate, np.frombuffer(raw, dtype="<i2").copy()


def fit_audio(samples, target_frames, overflow, cue_index):
    """Only speed up overlong speech; shorter speech is padded by the writer."""
    if target_frames < 1:
        raise ValueError(f"字幕 {cue_index} 时长不足一个音频采样点")
    if len(samples) <= target_frames:
        return samples
    factor = len(samples) / target_frames
    if overflow == "error":
        raise ValueError(f"字幕 {cue_index} 语音超时（需要 {factor:.2f} 倍加速）；可使用 --overflow speed-up 或 --timing concat")

    import librosa
    import numpy as np

    print(f">> 字幕 {cue_index}: 保音高加速 {factor:.2f} 倍以匹配时间轴", flush=True)
    stretched = librosa.effects.time_stretch(samples.astype(np.float32) / 32768.0, rate=factor)
    # Account for sample rounding; never truncate the original spoken waveform.
    stretched = librosa.util.fix_length(stretched, size=target_frames)
    return np.clip(np.rint(stretched * 32768.0), -32768, 32767).astype("<i2")


def write_silence(writer, frames):
    # Bounded memory even for large gaps in a subtitle timeline.
    while frames > 0:
        count = min(frames, 65536)
        writer.writeframesraw(b"\x00\x00" * count)
        frames -= count


def synthesize(tts, cues, args):
    """Write incrementally to a temporary WAV; publish only after all cues succeed."""
    if not cues:
        raise ValueError("没有可合成的字幕")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".indextts-srt-", dir=args.output.parent) as temp:
        segment_path = Path(temp) / "segment.wav"
        combined_path = Path(temp) / "combined.wav"
        sample_rate = None
        cursor = 0
        with wave.open(str(combined_path), "wb") as writer:
            # Set a valid header even if the first inference fails; the temp is discarded.
            writer.setparams((1, 2, 22050, 0, "NONE", "not compressed"))
            for position, cue in enumerate(cues):
                print(f">> [{position + 1}/{len(cues)}] 字幕 {cue.index}: {cue.text}", flush=True)
                segment_path.unlink(missing_ok=True)
                kwargs = dict(
                    spk_audio_prompt=str(args.voice), text=cue.text,
                    output_path=str(segment_path), use_emo_text=False,
                    interval_silence=0, verbose=args.verbose,
                    max_text_tokens_per_segment=args.max_text_tokens,
                    num_beams=args.num_beams,
                )
                if args.version == "2.5":
                    kwargs["lang"] = args.lang
                tts.infer(**kwargs)
                if not segment_path.is_file():
                    raise ValueError(f"字幕 {cue.index} 未生成音频")
                rate, samples = read_pcm(segment_path)
                if sample_rate is None:
                    sample_rate = rate
                    writer.setframerate(rate)
                elif rate != sample_rate:
                    raise ValueError(f"字幕 {cue.index} 输出采样率发生变化: {sample_rate} -> {rate}")

                if args.timing == "srt":
                    start = (cue.start_ms * rate + 500) // 1000
                    end = (cue.end_ms * rate + 500) // 1000
                    samples = fit_audio(samples, end - start, args.overflow, cue.index)
                    write_silence(writer, start - cursor)
                    writer.writeframesraw(samples.tobytes())
                    write_silence(writer, end - start - len(samples))
                    cursor = end
                else:
                    if position:
                        gap = (args.gap_ms * rate + 500) // 1000
                        write_silence(writer, gap)
                        cursor += gap
                    writer.writeframesraw(samples.tobytes())
                    cursor += len(samples)
        if args.overwrite:
            os.replace(combined_path, args.output)
        else:
            # Atomic no-clobber publish, including a file created during inference.
            os.link(combined_path, args.output)
    return sample_rate, cursor


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        cues = validate_inputs(args)
        print(f">> 共 {len(cues)} 条字幕，模型 v{args.version}，设备 {args.device}，模式 {args.timing}", flush=True)
        tts = load_tts(args)
        rate, frames = synthesize(tts, cues, args)
        print(f">> 已保存: {args.output.resolve()} ({frames / rate:.3f}s, {rate} Hz, mono PCM16)")
        return 0
    except Exception as exc:
        if args.verbose:
            raise
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())