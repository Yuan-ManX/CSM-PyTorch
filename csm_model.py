from dataclasses import dataclass
import torch
import torch.nn as nn
import torchtune
from huggingface_hub import PyTorchModelHubMixin
from torchtune.models import llama3_2


def llama3_2_1B() -> torchtune.modules.transformer.TransformerDecoder:
    """
    构建一个名为 llama3_2_1B 的 Transformer 解码器模型，参数如下：
    
    Returns:
        TransformerDecoder: 配置好的 Transformer 解码器模型实例
    """
    return llama3_2.llama3_2(
        vocab_size=128_256,           # 词汇表大小，128256个token
        num_layers=16,                # Transformer 层数，共16层
        num_heads=32,                 # 多头注意力机制的头数，共32个头
        num_kv_heads=8,               # KV 注意力头的数量，共8个
        embed_dim=2048,               # 词嵌入的维度，2048维
        max_seq_len=2048,             # 最大序列长度，2048个token
        intermediate_dim=8192,        # 前馈神经网络中间层的维度，8192维
        attn_dropout=0.0,             # 注意力机制中的dropout比例，0.0表示不使用dropout
        norm_eps=1e-5,                # 归一化层的epsilon值，防止除零错误
        rope_base=500_000,            # ROPE（旋转位置编码）的基数
        scale_factor=32,              # 缩放因子，用于调整某些层的输出
    )


def llama3_2_100M() -> torchtune.modules.transformer.TransformerDecoder:
    """
    构建一个名为 llama3_2_100M 的 Transformer 解码器模型，参数如下：
    
    Returns:
        TransformerDecoder: 配置好的 Transformer 解码器模型实例
    """
    return llama3_2.llama3_2(
        vocab_size=128_256,           # 词汇表大小，128256个token
        num_layers=4,                 # Transformer 层数，共4层
        num_heads=8,                  # 多头注意力机制的头数，共8个头
        num_kv_heads=2,               # KV 注意力头的数量，共2个
        embed_dim=1024,               # 词嵌入的维度，1024维
        max_seq_len=2048,             # 最大序列长度，2048个token
        intermediate_dim=8192,        # 前馈神经网络中间层的维度，8192维
        attn_dropout=0.0,             # 注意力机制中的dropout比例，0.0表示不使用dropout
        norm_eps=1e-5,                # 归一化层的epsilon值，防止除零错误
        rope_base=500_000,            # ROPE（旋转位置编码）的基数
        scale_factor=32,              # 缩放因子，用于调整某些层的输出
    )


# 定义可用的模型版本
FLAVORS = {
    "llama-1B": llama3_2_1B,
    "llama-100M": llama3_2_100M,
}


def _prepare_transformer(model):
    """
    准备 Transformer 模型，去除词嵌入层和输出层，并返回嵌入维度。
    
    Args:
        model (TransformerDecoder): 需要处理的 Transformer 解码器模型
    
    Returns:
        tuple:
            - TransformerDecoder: 修改后的 Transformer 模型
            - int: 词嵌入的维度
    """
    # 获取词嵌入的维度
    embed_dim = model.tok_embeddings.embedding_dim
    # 用恒等映射替换词嵌入层
    model.tok_embeddings = nn.Identity()
    # 用恒等映射替换输出层
    model.output = nn.Identity()
    # 返回修改后的模型和嵌入维度
    return model, embed_dim


def _create_causal_mask(seq_len: int, device: torch.device):
    """
    创建一个因果掩码矩阵，用于防止模型在预测时看到未来的token。
    
    Args:
        seq_len (int): 序列的长度
        device (torch.device): 设备类型（如CPU或GPU）
    
    Returns:
        torch.Tensor: 因果掩码矩阵，形状为 (seq_len, seq_len)
    """
    # torch.tril 返回下三角矩阵，torch.ones 创建一个全1的矩阵，dtype=torch.bool 表示布尔类型
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def _index_causal_mask(mask: torch.Tensor, input_pos: torch.Tensor):
    """
    根据输入的位置索引因果掩码。
    
    Args:
        mask (torch.Tensor): 因果掩码矩阵，形状为 (max_seq_len, max_seq_len)
        input_pos (torch.Tensor): 输入的位置索引，形状为 (batch_size, seq_len)
    
    Returns:
        torch.Tensor: 索引后的因果掩码，形状为 (batch_size, seq_len, max_seq_len)
    """
    # 根据 input_pos 索引 mask
    r = mask[input_pos, :]
    return r


def _multinomial_sample_one_no_sync(probs):
    """
    对概率分布进行多项式采样，不进行CUDA同步。
    
    Args:
        probs (torch.Tensor): 概率分布，形状为 (..., num_classes)
    
    Returns:
        torch.Tensor: 采样得到的token，形状为 (..., 1)
    """
    # 生成与 probs 形状相同的指数分布张量
    q = torch.empty_like(probs).exponential_(1)
    # 对 (probs / q) 进行 argmax 操作，得到采样结果
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample_topk(logits: torch.Tensor, topk: int, temperature: float):
    """
    对logits进行top-k采样。
    
    Args:
        logits (torch.Tensor): 模型的原始输出，形状为 (batch_size, seq_len, vocab_size)
        topk (int): top-k 采样的k值
        temperature (float): 温度参数，用于调整采样分布的平滑度
    
    Returns:
        torch.Tensor: 采样得到的token，形状为 (batch_size, seq_len, 1)
    """
    # 对logits进行温度缩放
    logits = logits / temperature
    
    # 定义过滤值，用于屏蔽不需要的token
    filter_value: float = -float("Inf")
    # 获取top-k的logits，并生成一个布尔掩码，屏蔽掉非top-k的logits
    indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]

    # 将非top-k的logits替换为 -Inf
    scores_processed = logits.masked_fill(indices_to_remove, filter_value)

    # 对处理后的logits进行log-softmax归一化
    scores_processed = torch.nn.functional.log_softmax(scores_processed, dim=-1)

    # 对log-softmax结果进行softmax，得到概率分布
    probs = torch.nn.functional.softmax(scores_processed, dim=-1)

    # 对概率分布进行多项式采样，得到采样结果
    sample_token = _multinomial_sample_one_no_sync(probs)

    # 返回采样得到的token
    return sample_token


@dataclass
class ModelArgs:
    """
    模型配置参数类，用于初始化 Model 类。

    Attributes:
        backbone_flavor (str): 主干模型的版本名称，例如 "llama-1B"。
        decoder_flavor (str): 解码器模型的版本名称，例如 "llama-100M"。
        text_vocab_size (int): 文本词汇表的大小，即文本token的总数。
        audio_vocab_size (int): 音频词汇表的大小，即音频token的总数。
        audio_num_codebooks (int): 音频编码本的数量，用于离散化音频信号。
    """
    backbone_flavor: str
    decoder_flavor: str
    text_vocab_size: int
    audio_vocab_size: int
    audio_num_codebooks: int


class Model(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="",
    pipeline_tag="text-to-speech",
    license="apache-2.0",
):
    """
    文本到语音（TTS）模型类，继承自 nn.Module 和 PyTorchModelHubMixin。

    Args:
        config (ModelArgs): 模型配置参数。
    
    Attributes:
        config (ModelArgs): 模型配置参数。
        backbone (TransformerDecoder): 主干 Transformer 解码器模型。
        decoder (TransformerDecoder): 音频解码器 Transformer 模型。
        text_embeddings (nn.Embedding): 文本嵌入层。
        audio_embeddings (nn.Embedding): 音频嵌入层。
        projection (nn.Linear): 线性投影层，用于将主干输出投影到解码器输入维度。
        codebook0_head (nn.Linear): 第一个编码本的线性头，用于预测第一个音频token。
        audio_head (nn.Parameter): 其他编码本的参数矩阵，用于预测后续音频token。
    """
    def __init__(self, config: ModelArgs):
        super().__init__()
        # 保存配置参数
        self.config = config

        # 准备主干 Transformer 模型，并获取其嵌入维度
        self.backbone, backbone_dim = _prepare_transformer(FLAVORS[config.backbone_flavor]())
        # 准备音频解码器 Transformer 模型，并获取其嵌入维度
        self.decoder, decoder_dim = _prepare_transformer(FLAVORS[config.decoder_flavor]())

        # 初始化文本嵌入层，嵌入维度为 backbone_dim，词汇表大小为 text_vocab_size
        self.text_embeddings = nn.Embedding(config.text_vocab_size, backbone_dim)
        # 初始化音频嵌入层，嵌入维度为 backbone_dim，词汇表大小为 audio_vocab_size * audio_num_codebooks
        self.audio_embeddings = nn.Embedding(config.audio_vocab_size * config.audio_num_codebooks, backbone_dim)

        # 初始化线性投影层，将 backbone_dim 投影到 decoder_dim，不使用偏置
        self.projection = nn.Linear(backbone_dim, decoder_dim, bias=False)
        # 初始化第一个编码本的线性头，输出维度为 audio_vocab_size，不使用偏置
        self.codebook0_head = nn.Linear(backbone_dim, config.audio_vocab_size, bias=False)
        # 初始化其他编码本的参数矩阵，形状为 (audio_num_codebooks - 1, decoder_dim, audio_vocab_size)
        self.audio_head = nn.Parameter(torch.empty(config.audio_num_codebooks - 1, decoder_dim, config.audio_vocab_size))

    def setup_caches(self, max_batch_size: int) -> torch.Tensor:
        """
        设置 KV 缓存并返回因果掩码。

        Args:
            max_batch_size (int): 最大批量大小，用于初始化缓存。

        Returns:
            torch.Tensor: 因果掩码，形状为 (max_seq_len, max_seq_len)。
        """
        # 获取模型参数的 dtype
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        with device:
            # 设置主干 Transformer 的 KV 缓存
            self.backbone.setup_caches(max_batch_size, dtype)
            # 设置音频解码器 Transformer 的 KV 缓存，decoder_max_seq_len 设置为 audio_num_codebooks
            self.decoder.setup_caches(max_batch_size, dtype, decoder_max_seq_len=self.config.audio_num_codebooks)

        # 创建并注册主干 Transformer 的因果掩码
        self.register_buffer("backbone_causal_mask", _create_causal_mask(self.backbone.max_seq_len, device))
        # 创建并注册音频解码器的因果掩码
        self.register_buffer("decoder_causal_mask", _create_causal_mask(self.config.audio_num_codebooks, device))

    def generate_frame(
        self,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        input_pos: torch.Tensor,
        temperature: float,
        topk: int,
    ) -> torch.Tensor:
        """
        生成一个音频帧的采样结果。

        Args:
            tokens (torch.Tensor): 输入的token，形状为 (batch_size, seq_len, audio_num_codebooks+1)。
            tokens_mask (torch.Tensor): token的掩码，形状为 (batch_size, seq_len, audio_num_codebooks+1)。
            input_pos (torch.Tensor): 每个token的位置索引，形状为 (batch_size, seq_len)。
            temperature (float): 采样温度，用于控制采样分布的平滑度。
            topk (int): top-k 采样的k值。

        Returns:
            torch.Tensor: 采样得到的音频token，形状为 (batch_size, audio_num_codebooks)。
        """
        # 获取模型参数的 dtype
        dtype = next(self.parameters()).dtype
        # 获取批量大小和序列长度
        b, s, _ = tokens.size()

        # 确保主干 Transformer 的缓存已启用
        assert self.backbone.caches_are_enabled(), "backbone caches are not enabled"
        # 获取当前主干 Transformer 的因果掩码
        curr_backbone_mask = _index_causal_mask(self.backbone_causal_mask, input_pos)
        # 对输入 tokens 进行嵌入
        embeds = self._embed_tokens(tokens)
        # 应用掩码
        masked_embeds = embeds * tokens_mask.unsqueeze(-1)
        # 对嵌入后的 tokens 进行求和
        h = masked_embeds.sum(dim=2)
        # 通过主干 Transformer 进行处理
        h = self.backbone(h, input_pos=input_pos, mask=curr_backbone_mask).to(dtype=dtype)

        # 获取最后一个隐藏状态
        last_h = h[:, -1, :]
        # 通过第一个编码本的线性头进行预测
        c0_logits = self.codebook0_head(last_h)
        # 进行 top-k 采样
        c0_sample = sample_topk(c0_logits, topk, temperature)
        # 对第一个编码本的采样结果进行嵌入
        c0_embed = self._embed_audio(0, c0_sample)

        # 初始化当前隐藏状态和采样结果
        curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
        curr_sample = c0_sample.clone()
        curr_pos = torch.arange(0, curr_h.size(1), device=curr_h.device).unsqueeze(0).repeat(curr_h.size(0), 1)

        # 重置音频解码器的缓存
        self.decoder.reset_caches()
        # 对后续的音频编码本进行迭代处理
        for i in range(1, self.config.audio_num_codebooks):
            # 获取当前解码器的因果掩码
            curr_decoder_mask = _index_causal_mask(self.decoder_causal_mask, curr_pos)
            # 通过解码器处理当前隐藏状态
            decoder_h = self.decoder(self.projection(curr_h), input_pos=curr_pos, mask=curr_decoder_mask).to(
                dtype=dtype
            )
            # 通过第 i 个编码本的参数矩阵进行预测
            ci_logits = torch.mm(decoder_h[:, -1, :], self.audio_head[i - 1])
            # 进行 top-k 采样
            ci_sample = sample_topk(ci_logits, topk, temperature)
            # 对第 i 个编码本的采样结果进行嵌入
            ci_embed = self._embed_audio(i, ci_sample)

            # 更新当前隐藏状态和采样结果
            curr_h = ci_embed
            curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
            curr_pos = curr_pos[:, -1:] + 1

        # 返回最终的采样结果
        return curr_sample

    def reset_caches(self):
        """
        重置主干 Transformer 和音频解码器的缓存。
        """
        self.backbone.reset_caches()
        self.decoder.reset_caches()

    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        """
        对音频token进行嵌入。

        Args:
            codebook (int): 当前编码本的索引。
            tokens (torch.Tensor): 音频token，形状为 (batch_size, seq_len)。

        Returns:
            torch.Tensor: 嵌入后的音频token，形状为 (batch_size, seq_len, embed_dim)。
        """
        return self.audio_embeddings(tokens + codebook * self.config.audio_vocab_size)

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        对输入的文本和音频token进行嵌入。

        Args:
            tokens (torch.Tensor): 输入的token，形状为 (batch_size, seq_len, audio_num_codebooks + 1)。

        Returns:
            torch.Tensor: 嵌入后的token，形状为 (batch_size, seq_len, audio_num_codebooks + 1, embed_dim)。
        """
        # 对最后一个token（假设为文本token）进行嵌入
        text_embeds = self.text_embeddings(tokens[:, :, -1]).unsqueeze(-2)
        
        # 对前 audio_num_codebooks 个token（假设为音频token）进行嵌入
        audio_tokens = tokens[:, :, :-1] + (
            self.config.audio_vocab_size * torch.arange(self.config.audio_num_codebooks, device=tokens.device)
        )
        audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(
            tokens.size(0), tokens.size(1), self.config.audio_num_codebooks, -1
        )

        # 将音频和文本嵌入结果拼接起来
        return torch.cat([audio_embeds, text_embeds], dim=-2)
