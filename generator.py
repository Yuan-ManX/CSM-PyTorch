from dataclasses import dataclass
from typing import List, Tuple

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from csm_model import Model
from moshi.models import loaders
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer
from watermarking import CSM_1B_GH_WATERMARK, load_watermarker, watermark


@dataclass
class Segment:
    """
    Segment 数据类用于表示一个音频片段，包括说话人ID、文本内容和音频张量。
    
    属性:
        speaker (int): 说话人的ID，用于区分不同的说话人。
        text (str): 对应的文本内容。
        audio (torch.Tensor): 音频张量，形状为 (num_samples,)，采样率为24,000 Hz。
    """
    speaker: int
    text: str
    # (num_samples,), sample_rate = 24_000
    audio: torch.Tensor


def load_llama3_tokenizer():
    """
    加载LLaMA 3.2-1B的分词器，并进行后处理以适应特定的任务需求。

    返回:
        AutoTokenizer: 配置好的LLaMA 3.2-1B分词器实例。
    
    流程:
        1. 定义分词器的名称，这里使用Hugging Face Hub上的"meta-llama/Llama-3.2-1B"模型。
        2. 使用AutoTokenizer.from_pretrained方法加载分词器。
        3. 获取分词器的开始标记（bos_token）和结束标记（eos_token）。
        4. 配置分词器的后处理器（post_processor），使用TemplateProcessing来定义模板：
            - single: 单个输入的模板，格式为 "<bos>:0 <文本>:0 <eos>:0"。
            - pair: 一对输入的模板，格式为 "<bos>:0 <文本A>:0 <eos>:0 <bos>:1 <文本B>:1 <eos>:1"。
            - special_tokens: 特殊标记及其对应的ID，包括开始标记和结束标记。
        5. 返回配置好的分词器实例。
    """
    tokenizer_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # 获取开始标记
    bos = tokenizer.bos_token
    # 获取结束标记
    eos = tokenizer.eos_token

    # 配置分词器的后处理器，使用TemplateProcessing定义模板
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",  # 单个输入的模板
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",  # 一对输入的模板
        special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],  # 特殊标记及其ID
    )

    return tokenizer


class Generator:
    """
    Generator 类用于生成音频片段。它结合了文本分词器、音频分词器和水印器，以处理文本输入并生成相应的音频输出。
    """
    def __init__(
        self,
        model: Model,
    ):
        """
        初始化Generator实例。

        参数:
            model (Model): 用于生成音频的预训练模型实例。
        
        流程:
            1. 将模型实例赋值给 self._model，并调用 setup_caches 方法设置缓存，参数为1。
            2. 调用 load_llama3_tokenizer 函数加载文本分词器，并赋值给 self._text_tokenizer。
            3. 获取模型的设备（CPU或GPU），并赋值给 device 变量。
            4. 从Hugging Face Hub下载MIMI权重文件，路径由 loaders.DEFAULT_REPO 和 loaders.MIMI_NAME 确定。
            5. 使用 loaders.get_mimi 函数加载MIMI，并将其移动到指定的设备上。
            6. 设置MIMI的码本数量为32。
            7. 将加载的MIMI赋值给 self._audio_tokenizer。
            8. 调用 load_watermarker 函数加载水印器，并将其移动到指定的设备上，赋值给 self._watermarker。
            9. 从MIMI中获取采样率，并赋值给 self.sample_rate。
            10. 将设备赋值给 self.device。
        """
        self._model = model
        # 设置缓存，参数为1
        self._model.setup_caches(1)

        # 加载文本分词器
        self._text_tokenizer = load_llama3_tokenizer()

        device = next(model.parameters()).device
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        mimi = loaders.get_mimi(mimi_weight, device=device)
        # 设置MIMI的码本数量为32
        mimi.set_num_codebooks(32)
        # 赋值音频分词器
        self._audio_tokenizer = mimi

        self._watermarker = load_watermarker(device=device)

        self.sample_rate = mimi.sample_rate
        self.device = device

    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对文本片段进行分词，并生成相应的帧标记和掩码。

        参数:
            text (str): 要分词的文本内容。
            speaker (int): 说话人的ID。
        
        返回:
            Tuple[torch.Tensor, torch.Tensor]: 包含帧标记和掩码的元组。
        
        流程:
            1. 初始化空的帧标记列表和帧掩码列表。
            2. 将文本和说话人ID组合成格式 "[<speaker>]<text>"，并使用文本分词器进行编码。
            3. 创建一个全零的张量，形状为 (len(text_tokens), 33)，类型为长整型。
            4. 创建一个全零的布尔型掩码张量，形状与文本张量相同。
            5. 将编码后的文本token赋值给文本张量的最后一列。
            6. 将掩码张量的最后一列设置为True，表示这些位置有有效的token。
            7. 将文本张量和掩码张量移动到模型所在的设备上。
            8. 将文本张量和掩码张量添加到各自的列表中。
            9. 将帧标记列表和帧掩码列表中的张量连接起来，返回。
        """
        # 初始化帧标记列表
        frame_tokens = []
        # 初始化帧掩码列表
        frame_masks = []

        # 对文本进行编码，格式为 [<speaker>]<text>
        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        # 创建全零的长整型张量，形状为 (len(text_tokens), 33)
        text_frame = torch.zeros(len(text_tokens), 33).long()
        # 创建全零的布尔型掩码张量，形状与文本张量相同
        text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
        # 将编码后的文本token赋值给文本张量的最后一列
        text_frame[:, -1] = torch.tensor(text_tokens)
        # 将掩码张量的最后一列设置为True
        text_frame_mask[:, -1] = True

        # 将文本张量移动到设备上，并添加到帧标记列表中
        frame_tokens.append(text_frame.to(self.device))
        # 将掩码张量移动到设备上，并添加到帧掩码列表中
        frame_masks.append(text_frame_mask.to(self.device))

        # 连接帧标记和帧掩码张量，并返回
        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对单通道音频张量进行分词，生成音频帧的token和对应的掩码。

        参数:
            audio (torch.Tensor): 单通道音频张量，形状为 (num_samples,)。
        
        返回:
            Tuple[torch.Tensor, torch.Tensor]: 包含音频帧token和对应掩码的元组。
        
        """
        assert audio.ndim == 1, "Audio must be single channel"

        # 初始化音频帧token列表
        frame_tokens = []
        # 初始化音频帧掩码列表
        frame_masks = []

        # (K, T)
        audio = audio.to(self.device)
        # 使用音频分词器对音频张量进行编码
        audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        # 添加EOS帧
        eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        # 创建音频帧张量，形状为 (T, 33)
        audio_frame = torch.zeros(audio_tokens.size(1), 33).long().to(self.device)
        # 创建音频帧掩码张量，形状与音频帧张量相同
        audio_frame_mask = torch.zeros(audio_tokens.size(1), 33).bool().to(self.device)
        # 将编码后的音频token赋值给音频帧张量的前T-1列
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        # 设置音频帧掩码张量的前T-1列
        audio_frame_mask[:, :-1] = True

        # 将音频帧张量和音频帧掩码张量添加到列表中
        frame_tokens.append(audio_frame)
        frame_masks.append(audio_frame_mask)

        # 连接音频帧token和音频帧掩码，并返回
        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对Segment对象进行分词，生成文本和音频的token和对应的掩码。

        参数:
            segment (Segment): 包含文本和音频信息的Segment对象。
        
        返回:
            Tuple[torch.Tensor, torch.Tensor]: 包含文本和音频token及对应掩码的元组。
        
        返回类型:
            (seq_len, 33), (seq_len, 33)
        """
        # 对Segment对象的文本和说话人ID进行分词
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        # 对Segment对象的音频进行分词
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)
        
        # 连接文本和音频的token及对应的掩码，并返回
        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 50,
    ) -> torch.Tensor:
        """
        使用模型生成音频。

        参数:
            text (str): 要生成的文本内容。
            speaker (int): 说话人的ID。
            context (List[Segment]): 上下文Segment对象列表，包含之前的音频片段。
            max_audio_length_ms (float): 生成音频的最大长度，单位为毫秒，默认为90,000毫秒（90秒）。
            temperature (float): 生成温度，默认为0.9。
            topk (int): Top-K值，默认为50。
        
        返回:
            torch.Tensor: 生成的音频张量。
        
        流程:
            1. **重置缓存**: 调用模型的 `reset_caches` 方法重置缓存。
            2. **计算最大生成长度**: 将 `max_audio_length_ms` 转换为生成的最大token数量，假设每个token对应80毫秒。
            3. **初始化token和掩码列表**: 初始化空的token和掩码列表，用于存储上下文和生成的token。
            4. **处理上下文**: 遍历上下文中的每个Segment对象，调用 `_tokenize_segment` 方法对其进行分词，并将结果添加到token和掩码列表中。
            5. **处理生成段**: 对输入的文本和说话人ID进行分词，并将结果添加到token和掩码列表中。
            6. **合并token和掩码**: 将所有token和掩码连接起来，并移动到模型所在的设备上。
            7. **检查输入长度**: 如果输入长度超过最大序列长度减去最大生成长度，则抛出错误。
            8. **生成过程**:
                - 使用一个循环，根据 `max_generation_len` 生成音频token。
                - 在每次迭代中，调用模型的 `generate_frame` 方法生成下一个音频token。
                - 如果生成的token全为零，则停止生成（假设这是EOS标记）。
                - 将生成的token添加到样本列表中，并更新当前token、掩码和位置。
            9. **解码音频**: 将生成的音频token解码为音频张量。
            10. **添加水印**: 使用 `watermark` 函数对音频添加水印，以确保音频的透明性和可追溯性。
            11. **重采样**: 将音频重采样到模型的采样率。
            12. **返回结果**: 返回生成的音频张量。
        """
        # 重置模型缓存
        self._model.reset_caches()

        # 计算最大生成长度，假设每个token对应80毫秒
        max_generation_len = int(max_audio_length_ms / 80)
        # 初始化token和掩码列表
        tokens, tokens_mask = [], []
        # 遍历上下文中的每个Segment对象
        for segment in context:
            # 对Segment对象进行分词
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            # 将token添加到列表中
            tokens.append(segment_tokens)
            # 将掩码添加到列表中
            tokens_mask.append(segment_tokens_mask)

        # 对输入的文本和说话人ID进行分词
        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        # 将生成的token添加到列表中
        tokens.append(gen_segment_tokens)
        # 将生成的掩码添加到列表中
        tokens_mask.append(gen_segment_tokens_mask)

        # 连接所有token并移动到设备上
        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        # 连接所有掩码并移动到设备上
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)

        # 初始化样本列表，用于存储生成的音频token
        samples = []
        # 增加一个维度，形状变为 (1, seq_len)
        curr_tokens = prompt_tokens.unsqueeze(0)
        # 增加一个维度，形状变为 (1, seq_len)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        # 生成当前位置张量
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)

        # 定义最大序列长度
        max_seq_len = 2048
        # 计算最大上下文长度
        max_context_len = max_seq_len - max_generation_len
        # 检查输入长度是否超过限制
        if curr_tokens.size(1) >= max_context_len:
            raise ValueError(
                f"Inputs too long, must be below max_seq_len - max_generation_len: {max_context_len}"
            )

        # 开始生成过程
        for _ in range(max_generation_len):
            # 生成下一个音频token
            sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
            # 如果生成的token全为零，则停止生成
            if torch.all(sample == 0):
                break  # eos
            
            # 将生成的token添加到样本列表中
            samples.append(sample)

            # 更新当前token
            curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
            # 更新当前掩码
            curr_tokens_mask = torch.cat(
                [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
            ).unsqueeze(1)
            # 更新当前位置
            curr_pos = curr_pos[:, -1:] + 1
        
        # 解码音频token为音频张量
        audio = self._audio_tokenizer.decode(torch.stack(samples).permute(1, 2, 0)).squeeze(0).squeeze(0)

        # 添加水印: 这会在音频中嵌入一个不可察觉的水印，以识别音频为AI生成。水印确保透明度，防止滥用，并实现可追溯性。
        audio, wm_sample_rate = watermark(self._watermarker, audio, self.sample_rate, CSM_1B_GH_WATERMARK)
        # 重采样音频
        audio = torchaudio.functional.resample(audio, orig_freq=wm_sample_rate, new_freq=self.sample_rate)

        return audio


def load_csm_1b(device: str = "cuda") -> Generator:
    """
    加载CSM 1B模型并返回Generator实例。

    参数:
        device (str): 设备名称，默认为 "cuda"。可以选择 "cuda", "mps", "cpu" 等。
    
    返回:
        Generator: 初始化好的Generator实例。
    
    流程:
        1. 从指定的Hugging Face Hub路径加载预训练的CSM 1B模型。
        2. 将模型移动到指定的设备上，并设置数据类型为 bfloat16。
        3. 创建Generator实例，并将加载的模型传入。
        4. 返回Generator实例。
    """
    model = Model.from_pretrained("sesame/csm-1b")
    model.to(device=device, dtype=torch.bfloat16)

    # 创建Generator实例
    generator = Generator(model)
    return generator
