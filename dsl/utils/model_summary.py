import torch
from torchinfo import summary
from dsl.model.cogvideox.cogvideox_transformer_3d import CogVideoXTransformer3DModel

# 如果你的模型文件命名为 cogvideo_transformer.py，你需要先导入其中的类
# from cogvideo_transformer import CogVideoXTransformer3DModel

# 1. 实例化模型（此处仅示例，实际使用时需要传入对应的配置参数）
model = CogVideoXTransformer3DModel()

# 准备输入
batch_size = 1
sample_frames = 16
in_channels = 16
height = 60
width = 90
max_text_seq_length = 226
text_embed_dim = 4096

dummy_hidden_states = torch.randn(
    batch_size, sample_frames, in_channels, height, width
)
dummy_encoder_hidden_states = torch.randn(
    batch_size, max_text_seq_length, text_embed_dim
)
dummy_timestep = torch.tensor([10] * batch_size)

# 保存到文件
summary_stats = summary(
    model,
    input_data=(
        dummy_hidden_states,
        dummy_encoder_hidden_states,
        dummy_timestep,
    ),
    col_names=["input_size", "output_size", "num_params", "params_percent"],
    dtypes=[torch.float32, torch.float32, torch.long],
    device="cpu",
    verbose=2,  # 确保打印详细信息
)

# 将结果保存到文件
with open("model_summary.txt", "w") as f:
    f.write(str(summary_stats))  # 使用 str() 将其转换为字符串