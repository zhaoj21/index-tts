# WAV 音色参考 + SRT 字幕配音

独立入口：[infer_srt.py](../infer_srt.py)。不依赖 WebUI，不会修改原始 WAV/SRT。
WAV **只用作音色参考**，不会识别其内容，也不会混入生成结果。合成文字来自 SRT 正文。

## 使用

在项目已安装依赖的 Python 环境中，从项目根目录执行：

```bash
python infer_srt.py --voice reference.wav --srt subtitles.srt --output outputs/dubbed.wav
```

没有 GPU 时可以明确指定 CPU：

```bash
python infer_srt.py --voice reference.wav --srt subtitles.srt --output outputs/dubbed.wav --device cpu
```

默认读取脚本所在目录下的 checkpoints，模型版本为 2.5。
通过 `--model-dir /path/to/checkpoints` 指定其他权重目录；使用 v2 权重时添加 `--version 2`。
主模型文件必须提前准备好；底层推理代码可能自动下载缺失的辅助模型，首次运行可能需要联网。
脚本不启用 QwenEmotion、DeepSpeed、Flash Attention、CUDA 自定义 kernel 或 torch.compile。

## 时间轴处理

- 默认 `--timing srt`：逐条推理，把语音放在字幕开始时间，空隙和句尾不足的时长补静音。
- 语音超过字幕时长时，默认 `--overflow speed-up` 使用已有依赖 librosa 保音高加速，并打印加速倍数。
  倍数过大可能影响可懂度和音质；可延长字幕时间，或者改用顺序拼接模式。
- `--overflow error`：遇到超时就报错，不自动加速，也不截断原始语音。
- 时间轴模式不支持重叠或倒序字幕，会在加载模型前报错。
- 最终音频长度到最后一条字幕的结束时间（采样点取整）；不会自动补到参考 WAV 的时长。
- `--timing concat --gap-ms 200`：忽略字幕时间戳，按文件顺序拼接自然时长的语音，条目之间添加 200ms 静音。
  此模式允许时间轴重叠，不会对语音做时长压缩。

## 常用选项

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--device` | `auto` | 自动选择可用设备，也可指定 `cpu`、`cuda:0` 等 |
| `--lang` | `ZH` | v2.5 可选 `ZH/EN/JA/AR/ES`；v2 仅中英文 |
| `--encoding` | `utf-8-sig` | 支持带 BOM 的 UTF-8；旧字幕可指定 `gb18030` |
| `--num-beams` | `1` | 解码 beam 数，越大通常越慢 |
| `--max-text-tokens` | `120` | 模型内部文本分段大小 |
| `--half` | 关闭 | v2 为 FP16，v2.5 为 BF16；仅在设备支持时使用，CPU 会关闭 |
| `--overwrite` | 关闭 | 明确允许替换已有输出，不允许覆盖输入 |
| `--verbose` | 关闭 | 显示详细日志和错误堆栈 |

接受标准的“序号 + 时间轴 + 正文”SRT，支持多行正文、CRLF、常见 HTML 字幕格式标签。
不支持高级定位指令、ASS 样式或自动区分说话人；所有条目使用同一个参考音色。

输出为模型采样率的单声道 PCM16 WAV。每条音频临时落盘后立即拼接，模型只加载一次。
全部条目成功后才发布输出；失败会清理临时文件，不留下半成品，也不替换已有输出。
CPU 可用但较慢；请先用短字幕验证速度与效果。