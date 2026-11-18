import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import librosa
import soundfile as sf
from accelerate import init_empty_weights

class ShizhenGPTDeployer:
    def __init__(self, model_path='./ShizhenGPT-7B-Omni'):  # 修改为本地路径
        print("正在加载模型...")

        # 检查本地路径是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在: {model_path}")

        print(f"从本地路径加载模型: {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print("模型加载完成!")

    def process_image(self, image_path):
        """处理图像输入"""
        if isinstance(image_path, str):
            return Image.open(image_path).convert('RGB')
        return image_path

    def process_audio(self, audio_path):
        """处理音频输入"""
        audio, sr = librosa.load(audio_path, sr=self.processor.feature_extractor.sampling_rate)
        return audio

    def generate_response(self, text, images=None, signals=None):
        """生成回复"""
        # 处理图像
        processed_images = []
        if images is not None and images:
            text = ''.join(['<|vision_start|><|image_pad|><|vision_end|>'] * len(images)) + text
            processed_images = [self.process_image(img) for img in images if img is not None]

        # 处理音频
        processed_signals = []
        if signals is not None and signals:
            text = ''.join(['<|audio_bos|><|AUDIO|><|audio_eos|>'] * len(signals)) + text
            processed_signals = [self.process_audio(signal) for signal in signals if signal is not None]

        # 准备消息
        messages = [{'role': 'user', 'content': text}]
        text_processed = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 处理输入数据
        input_data = self.processor(
            text=[text_processed],
            audios=processed_signals if processed_signals else None,
            images=processed_images if processed_images else None,
            return_tensors="pt",
            padding=True
        )
        input_data = input_data.to(self.model.device)

        # 生成回复
        with torch.no_grad():
            generated_ids = self.model.generate(
                **input_data,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(input_data.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0]


# 使用示例
if __name__ == "__main__":
    # 初始化部署器，指定本地路径
    deployer = ShizhenGPTDeployer('./ShizhenGPT-7B-Omni')

    # 1. 纯文本问答
    print("=== 文本问答 ===")
    questions = [
        '为什么我总是手脚冰凉，是阳虚吗？',
        '中医如何调理失眠？',
        '什么是气虚，有什么症状？',
        '请解释一下中医的阴阳平衡理论'
    ]

    for question in questions:
        print(f"问题: {question}")
        response = deployer.generate_response(question)
        print(f"回答: {response}")
        print("-" * 50)