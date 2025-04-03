import torch
import torchaudio

from generator import load_csm_1b, Segment


# Generate a sentence
# 检查可用的硬件设备，并设置生成器使用的设备
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# 加载预训练的生成模型，并将其移动到指定的设备上
generator = load_csm_1b(device=device)

# 使用生成器生成音频
# 参数说明：
# - text (str): 要合成的文本内容，这里是 "Hello World!"
# - speaker (int): 指定说话人的ID，这里为0。不同的ID可能对应不同的声音或说话人特征
# - context (list): 上下文信息列表，这里为空列表[]，表示没有上下文
# - max_audio_length_ms (int): 生成音频的最大长度，单位为毫秒，这里设置为10,000毫秒（10秒）
audio = generator.generate(
    text="Hello World!",
    speaker=0,
    context=[],
    max_audio_length_ms=10_000,
)

# 将生成的音频保存为 WAV 文件
torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)


# Generate with context
# 生成带有上下文的音频
# 定义说话人列表，不同的说话人ID对应不同的声音或说话人特征
speakers = [0, 1, 0, 0]

# 定义文本转录列表，每个元素对应一个说话人的文本内容
transcripts = [
    "Hey how are you doing.",
    "Pretty good, pretty good.",
    "I'm great.",
    "So happy to be speaking to you.",
]

# 定义音频文件路径列表，每个元素对应一个音频文件的保存路径
audio_paths = [
    "utterance_0.wav",
    "utterance_1.wav",
    "utterance_2.wav",
    "utterance_3.wav",
]

# 定义一个函数，用于加载音频文件并调整其采样率以匹配生成器的采样率
def load_audio(audio_path):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    # 将原始采样率 orig_freq 调整为生成器的采样率 new_freq
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
    )
    return audio_tensor

# 创建 Segment 对象列表，每个对象包含文本、说话人ID和音频张量
segments = [
    Segment(text=transcript, speaker=speaker, audio=load_audio(audio_path))
    for transcript, speaker, audio_path in zip(transcripts, speakers, audio_paths)
]

# 使用生成器生成带有上下文的音频
# 参数说明：
# - text (str): 要合成的文本内容，这里是 "Hello World!"
# - speaker (int): 指定说话人的ID，这里为1
# - context (list): 上下文信息列表，这里包含之前生成的音频片段 segments
# - max_audio_length_ms (int): 生成音频的最大长度，单位为毫秒，这里设置为10,000毫秒（10秒）
audio = generator.generate(
    text="Hello World!",
    speaker=1,
    context=segments,
    max_audio_length_ms=10_000,
)

# 将生成的音频保存为 WAV 文件
torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
