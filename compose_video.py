#!/usr/bin/env python3
"""Combine video, optional replacement audio and UTF-8 SRT using FFmpeg."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def run(command: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode:
        raise RuntimeError(f"{Path(command[0]).name} 执行失败：\n{result.stderr[-6000:]}")
    return result.stdout


def probe(path: Path) -> dict:
    return json.loads(run([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path),
    ]))


def streams(info: dict, kind: str) -> list[dict]:
    return [stream for stream in info.get("streams", []) if stream.get("codec_type") == kind]


def compose_video(
    video: Path,
    subtitles: Path,
    output: Path,
    audio: Path | None = None,
    subtitle_mode: str = "soft",
    overwrite: bool = False,
) -> Path:
    """Keep the first video stream and exactly one selected audio stream, if present.

    External audio starts at video time zero (no automatic speech alignment).
    Soft mode copies video; burn mode encodes SDR video to H.264.
    """
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"未找到 {tool}，请先安装 FFmpeg 并加入 PATH。")
    video, subtitles, output = (Path(p).expanduser().resolve() for p in (video, subtitles, output))
    audio = Path(audio).expanduser().resolve() if audio is not None else None
    inputs = [video, subtitles] + ([audio] if audio is not None else [])
    for path in inputs:
        if not path.is_file():
            raise ValueError(f"输入文件不存在：{path}")
        if output == path or (output.exists() and output.samefile(path)):
            raise ValueError("输出不能与任何输入文件相同。")
    if output.suffix.lower() != ".mp4":
        raise ValueError("输出文件必须使用 .mp4 扩展名。")
    if subtitles.suffix.lower() != ".srt":
        raise ValueError("字幕必须是 UTF-8 编码的 .srt 文件。")
    if subtitle_mode not in {"soft", "burn"}:
        raise ValueError("字幕模式必须为 soft 或 burn。")
    if output.exists() and not overwrite:
        raise FileExistsError(f"输出已存在：{output}；如需覆盖请添加 --overwrite。")
    text = subtitles.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ValueError("字幕内容为空。")

    video_info = probe(video)
    videos = [s for s in streams(video_info, "video") if not s.get("disposition", {}).get("attached_pic")]
    if not videos:
        raise ValueError("输入视频不包含视频流。")
    selected_video = videos[0]
    try:
        duration = float(selected_video.get("duration") or video_info["format"]["duration"])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("无法确定视频时长。") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("视频时长必须为正数。")
    if subtitle_mode == "burn" and selected_video.get("color_transfer") in {"smpte2084", "arib-std-b67"}:
        raise ValueError("烧录模式暂不支持 HDR 视频，请使用 soft 模式或先转换为 SDR。")
    has_audio = bool(streams(video_info, "audio"))
    if audio is not None:
        if not streams(probe(audio), "audio"):
            raise ValueError("新音频文件不包含音频流。")
        has_audio = True

    output.parent.mkdir(parents=True, exist_ok=True)
    # Safe local subtitle name avoids FFmpeg filter escaping for user paths.
    # A staged MP4 also prevents failed encodes from leaving a partial output.
    with tempfile.TemporaryDirectory(prefix=".compose-", dir=output.parent) as directory:
        work = Path(directory)
        local_srt = work / "captions.srt"
        local_srt.write_text(text, encoding="utf-8")
        if not streams(probe(local_srt), "subtitle"):
            raise ValueError("无法读取 SRT 字幕流。")
        staged = work / "result.mp4"
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-i", str(video)]
        if audio is not None:
            command += ["-i", str(audio)]
        if subtitle_mode == "soft":
            command += ["-i", str(local_srt)]
        command += ["-map", f"0:{selected_video['index']}"]
        if has_audio:
            # Explicit mapping: original audio is never mixed with replacement audio.
            command += ["-map", "1:a:0" if audio is not None else "0:a:0"]
        if subtitle_mode == "soft":
            subtitle_index = 2 if audio is not None else 1
            command += ["-map", f"{subtitle_index}:s:0", "-c:v", "copy", "-c:s", "mov_text",
                        "-disposition:s:0", "default"]
        else:
            command += ["-vf", "subtitles=filename=captions.srt", "-c:v", "libx264",
                        "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p"]
        if has_audio:
            command += ["-c:a", "aac", "-b:a", "192k", "-af", "apad"]
        # Do not use -shortest: short subtitles or audio must not truncate video.
        command += ["-t", f"{duration:.9f}", "-map_chapters", "-1", "-movflags", "+faststart", str(staged)]
        run(command, cwd=work)
        result = probe(staged)
        if len(streams(result, "video")) != 1 or len(streams(result, "audio")) != int(has_audio):
            raise RuntimeError("合成结果的视频/音频流数量不符合预期。")
        if len(streams(result, "subtitle")) != int(subtitle_mode == "soft"):
            raise RuntimeError("合成结果的字幕流数量不符合预期。")
        if overwrite:
            os.replace(staged, output)
        else:
            # Same-filesystem atomic publication, refusing even a concurrent overwrite.
            os.link(staged, output)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="合成视频、新音频和 SRT 字幕，输出 MP4。")
    parser.add_argument("--video", type=Path, required=True, help="输入视频路径")
    parser.add_argument("--audio", type=Path, help="新音频路径；提供则替换原声，不提供则保留原声")
    parser.add_argument("--subtitles", type=Path, required=True, help="UTF-8 SRT 字幕路径")
    parser.add_argument("--output", type=Path, required=True, help="输出 MP4 路径")
    parser.add_argument("--subtitle-mode", choices=("soft", "burn"), default="soft",
                        help="soft：可开关字幕且复制画面（默认）；burn：字幕烧录到画面")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有输出，但绝不覆盖输入")
    args = parser.parse_args(argv)
    try:
        output = compose_video(args.video, args.subtitles, args.output, args.audio,
                               args.subtitle_mode, args.overwrite)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    print(f"合成完成：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())