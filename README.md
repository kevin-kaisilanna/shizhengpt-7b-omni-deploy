# ShizhenGPT-7B-Omni 本地部署测试

从 `ShizhenGPT-7B-Omni` 工程中整理出的**自行编写的部署 / 测试代码**：一行命令确认 CUDA 环境，再用一个封装类加载本地模型跑中医问答冒烟测试。

> **本仓库不含模型。** 解压后约 18 GB 的模型权重（原始 zip 约 14 GB）、官方建模代码、LM Studio 安装包、conda 环境、官方 demo 一律未上传；来源、许可证与获取方式见文末。

## 这个工程做什么

| 文件 | 原名 | 作用 |
| --- | --- | --- |
| `check_cuda.py` | `chack_cuda.py` | 打印 PyTorch 能否看到 CUDA、GPU 数量、第一块卡的型号，用来在加载 7B 模型前先确认环境 |
| `shizhengpt_deploy_test.py` | `Employment(1).py` | `ShizhengGPTDeployer` 类：从本地目录以 `trust_remote_code=True` 加载模型与 processor；`generate_response()` 支持文本 / 图像 / 音频混合输入，自动插入 `<\|vision_start\|>`、`<\|audio_bos\|>` 等占位 token、套 chat template、生成后裁剪解码。 `__main__` 跑 4 个中医文本问答冒烟测试 |
| `.vscode/settings.json` | 同名 | 原工程的 VS Code / conda 环境设置，仅作环境参考 |

脚本内容保持原样，只统一了换行符和文件名。模型路径用相对路径 `./ShizhenGPT-7B-Omni`，可自行修改 `model_path`。

## 部署：从零跑通一次推理

**1. 先把模型放到脚本旁边**（本仓库不含模型，也不能只拷权重）

从 HuggingFace 下载**完整目录**，不要只下 `*.safetensors` —— `trust_remote_code=True` 会加载目录里的 `modeling_shizhen.py`、`processing_qwen2_5_vl_audio.py`，还缺不得 `config.json`、`tokenizer*`、`model.safetensors.index.json`：

```bash
pip install -U huggingface_hub
# 国内网络可先设镜像（脚本里也已默认设置）
export HF_ENDPOINT=https://hf-mirror.com     # Windows: set HF_ENDPOINT=https://hf-mirror.com
hf download FreedomIntelligence/ShizhenGPT-7B-Omni --local-dir ./ShizhenGPT-7B-Omni
```

下载完目录里应有 4 个 `model-0000*-of-00004.safetensors`（解压后合计约 17.9 GB）、自定义建模与处理代码、tokenizer 全套文件。放到与 `shizhengpt_deploy_test.py` 同级的 `./ShizhenGPT-7B-Omni/`，或把 `model_path` 改成自己的绝对路径。

**2. 建 Python 环境并装依赖**

```bash
conda create -n shizhen python=3.10 -y
conda activate shizhen
pip install -r requirements.txt
```

两个硬约束：

- 官方明确要求 **`transformers==4.51.0`**，更高 / 更低版本可能无法识别 `Qwen2_5_VL_Audio_ForConditionalGeneration` 架构；`requirements.txt` 已锁定。
- `torch` 要按本机 CUDA 版本单独装（例如 `pip install torch --index-url https://download.pytorch.org/whl/cu121`），`requirements.txt` 里只给下限，不锁死 CUDA 版本。

**3. 先验 GPU**

```bash
python check_cuda.py
```

期望输出 `True`、GPU 数量（单卡为 `1`）、显卡型号。若第一行为 `False`，先别往下跑 —— 说明装的是 CPU 版 torch。

**4. 跑部署测试**

```bash
python shizhengpt_deploy_test.py
```

期望依次看到 `正在加载模型...` → `从本地路径加载模型: ./ShizhenGPT-7B-Omni` → `模型加载完成!` → `=== 文本问答 ===` 下 4 组「问题 / 回答」及分隔线。首次加载 7B 权重需要几十秒到数分钟。

**5. 换成图像 / 音频输入**

`generate_response(text, images=[...], signals=[...])` 已封装好：`images` 传图片路径或 PIL 对象，`signals` 传音频文件路径（用 `processor.feature_extractor.sampling_rate` 重采样）。在当前 `__main__` 里把某个问题改成带这两个参数即可。

### 常见失败

| 现象 | 处理 |
| --- | --- |
| `FileNotFoundError: 模型路径不存在` | 检查 `./ShizhenGPT-7B-Omni/` 是否在本脚本同级，或改 `model_path` |
| `KeyError` / 架构不认识 | `transformers` 不是 4.51.0，`pip install transformers==4.51.0` |
| 显存不足 / OOM | 7B fp16 约需 15 GB 以上显存；脚本已用 `device_map="auto"`，可加 CPU offload 或换 7B-LLM / VL 变体 |
| `No module named 'qwen_vl_utils'` / `accelerate` / `librosa` | 重新 `pip install -r requirements.txt` |
| 音频读取报错 | 补装 `soundfile` 与系统 ffmpeg；`librosa.load` 依赖它们 |

## 第三方内容（均未上传）

| 类别 | 来源 / 许可证 / 获取方式 |
| --- | --- |
| **ShizhenGPT-7B-Omni 模型权重**（4 个 safetensors，约 14 GB）、`modeling_shizhen.py`、`processing_qwen2_5_vl_audio.py`、tokenizer / config / 模型 README | [FreedomIntelligence/ShizhenGPT-7B-Omni](https://huggingface.co/FreedomIntelligence/ShizhenGPT-7B-Omni)，**Apache-2.0**，基座为 `Qwen/Qwen2.5-7B`；按上文第 1 步自行下载 |
| 官方 demo `app_shizhengpt.py`、`requirements.txt`、`README.md`、`assets/image.png` | [github.com/FreedomIntelligence/ShizhenGPT](https://github.com/FreedomIntelligence/ShizhenGPT)；本仓库未复制其代码，仅参考其依赖版本说明 |
| `LM-Studio-0.3.31-7-x64.exe`（567 MB）、conda 环境 `shizhen`、演示图片与测试音频 | 各自官网 / 本机环境，未分发 |
| `transformers` / `torch` / `Pillow` / `librosa` / `soundfile` / `accelerate` / `qwen_vl_utils` | 用 `requirements.txt` 从 PyPI 安装，权利归各自权利人 |

本仓库只包含自行编写的 2 个部署测试脚本（及原 VS Code 配置），**没有复制任何官方示例、模型文件、权重或第三方库代码**。

## 许可证

仅自研部署测试代码，尚未指定开源许可证（默认保留所有权利）。ShizhenGPT 模型及官方代码权利归 FreedomIntelligence，依其 Apache-2.0 许可使用，与本仓库无关。
