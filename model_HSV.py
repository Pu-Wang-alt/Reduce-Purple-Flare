import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia  # 确保已安装: pip install kornia


# =====================================================================================
# 1. 基础模块 (无需改动)
# =====================================================================================

class VectorQuantizer(nn.Module):
    """
    基于 VQ-VAE 的向量量化器模块。
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.embedding = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim), requires_grad=False)
        self.embedding.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', torch.zeros(num_embeddings, self.embedding_dim))

    def forward(self, inputs):
        input_shape = inputs.shape
        flat_input = inputs.reshape(-1, self.embedding_dim)
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.t()))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self.embedding).reshape(input_shape)
        if self.training:
            self.ema_cluster_size = self.ema_cluster_size * self.decay + \
                                    (1 - self.decay) * torch.sum(encodings, 0)
            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size.data = (self.ema_cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
            dw = torch.matmul(encodings.t(), flat_input.detach())
            self.ema_w = self.ema_w * self.decay + (1 - self.decay) * dw
            self.embedding.data = self.ema_w / self.ema_cluster_size.unsqueeze(1)
        loss = self.commitment_cost * F.mse_loss(quantized.detach(), inputs)
        quantized = inputs + (quantized - inputs).detach()
        final_encoding_indices = encoding_indices.reshape(input_shape[0], input_shape[1])
        return {
            "quantized": quantized,
            "loss": loss,
            "encoding_indices": final_encoding_indices
        }


class ResidualBlock(nn.Module):
    """
    残差块，用于增加模型深度而不增加过拟合风险。
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = self.activation(x)
        return x


# =====================================================================================
# 2. HSV 通道分词器 (无需改动)
# =====================================================================================
class CAST(nn.Module):
    """
    色差感知光谱分词器 (CAST)
    处理 HSV 颜色空间中的 H 和 V 通道。
    """

    def __init__(self, image_size, vocab_size, embedding_dim, sequence_length, commitment_cost=0.25, decay=0.99):
        super(CAST, self).__init__()
        self.embedding_dim = embedding_dim
        self.quantizer = VectorQuantizer(vocab_size, embedding_dim, commitment_cost, decay)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            ResidualBlock(64),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            ResidualBlock(128),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            ResidualBlock(256),
            nn.Conv2d(256, self.embedding_dim, kernel_size=4, stride=2, padding=1)
        )
        self.post_quant_conv = nn.Conv2d(self.embedding_dim, 256, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            ResidualBlock(256),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            ResidualBlock(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            ResidualBlock(64),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1), nn.Sigmoid()
        )

    def encode(self, img):
        b, _, _, _ = img.shape
        hsv_img = kornia.color.rgb_to_hsv(img)
        h_channel, s_channel, v_channel = hsv_img[:, 0:1], hsv_img[:, 1:2], hsv_img[:, 2:3]
        h_features = self.encoder(h_channel)
        v_features = self.encoder(v_channel)
        _, _, h_enc, w_enc = h_features.shape
        spatial_shape = (h_enc, w_enc)
        h_flat = h_features.permute(0, 2, 3, 1).contiguous().view(b, h_enc * w_enc, self.embedding_dim)
        v_flat = v_features.permute(0, 2, 3, 1).contiguous().view(b, h_enc * w_enc, self.embedding_dim)
        h_vq_output = self.quantizer(h_flat)
        v_vq_output = self.quantizer(v_flat)
        quantization_loss = h_vq_output["loss"] + v_vq_output["loss"]
        return {
            "h": h_vq_output["encoding_indices"], "v": v_vq_output["encoding_indices"],
            "h_quantized": h_vq_output["quantized"], "v_quantized": v_vq_output["quantized"],
            "quantization_loss": quantization_loss, "spatial_shape": spatial_shape,
            "s_channel": s_channel
        }

    def decode_channels(self, features_dict, spatial_shape, s_channel):
        recon_channels = {}
        for name in ['h', 'v']:
            quantized_features = features_dict[name]
            b, h, w = quantized_features.size(0), spatial_shape[0], spatial_shape[1]
            feature_map = quantized_features.view(b, h, w, self.embedding_dim).permute(0, 3, 1, 2).contiguous()
            post_quant_map = self.post_quant_conv(feature_map)
            recon_channels[name] = self.decoder(post_quant_map)
        reconstructed_hsv = torch.cat([recon_channels['h'], s_channel, recon_channels['v']], dim=1)
        return kornia.color.hsv_to_rgb(reconstructed_hsv)

    def forward(self, img):
        encode_output = self.encode(img)
        tokens_dict = {"h_quantized": encode_output["h_quantized"], "v_quantized": encode_output["v_quantized"]}
        recon_image = self.decode_channels(tokens_dict, encode_output["spatial_shape"], encode_output["s_channel"])
        return {
            "recon_image": recon_image,
            "quantization_loss": encode_output["quantization_loss"],
            "h": encode_output["h"], "v": encode_output["v"]
        }


# =====================================================================================
# 3. 优化后的高效 LUT 模块 (来自你的 V2 版本)
# =====================================================================================
class LUTModule(nn.Module):
    """
    高效 LUT 模块，使用 F.grid_sample 加速并融合多个动态权重的 LUT。
    适配拼接后的 H、V token 序列。
    """

    def __init__(self, vocab_size, embedding_dim, num_luts=8, hidden_dim=256):
        super(LUTModule, self).__init__()
        self.num_luts = num_luts
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

        # Token 聚合器，输入维度为 embedding_dim，因为 embedding 层已处理完 token
        self.token_aggregator = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # LUT 参数生成器
        self.lut_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_luts * 3 * 33 * 33 * 33)
        )

        # 动态权重生成器
        self.weight_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, num_luts)
        )

        # 残差分支处理器
        self.residual_processor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.GELU(),
            ResidualBlock(32),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

        # 最终融合层
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
            nn.GELU(),
            ResidualBlock(32),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

    def apply_3dlut(self, image, lut):
        """
        使用 F.grid_sample 高效应用 3D LUT。
        Args:
            image (torch.Tensor): 输入图像 [B, 3, H, W]，范围 [0, 1]
            lut (torch.Tensor): 3D LUT [B, 3, 33, 33, 33]
        """
        B, C, H, W = image.shape
        # 将图像坐标从 [0, 1] 映射到 [-1, 1]，并调整通道顺序以匹配 grid_sample
        grid = image.permute(0, 2, 3, 1)[:, :, :, [2, 1, 0]] * 2 - 1

        # F.grid_sample 需要 LUT 在 D, H, W 维度上，所以 lut 需要是 [B, 3, 33, 33, 33]
        # grid 需要是 [B, 1, H, W, 3]
        output = F.grid_sample(
            lut,
            grid.unsqueeze(1),  # 增加一个深度维度
            mode='bilinear',
            padding_mode='border',
            align_corners=True  # LUT 坐标通常是对齐角落的
        )
        return output.squeeze(2)  # 移除深度维度

    def forward(self, tokens, img, recon_image):
        """
        前向传播
        Args:
            tokens (torch.Tensor): 拼接后的 H、V token 序列 [B, 2 * seq_len]
            img (torch.Tensor): 原始输入图像 [B, 3, H, W]
            recon_image (torch.Tensor): 来自 CAST 的重建图像 [B, 3, H, W]
        """
        B, _, H, W = img.shape

        # 1. Token embedding & aggregation
        emb = self.token_embedding(tokens)  # [B, 2*seq_len, D]
        agg_features = torch.mean(emb, dim=1)  # [B, D]
        token_features = self.token_aggregator(agg_features)  # [B, hidden_dim]

        # 2. 生成多个 LUT
        lut_params = self.lut_generator(token_features)
        lut_params = lut_params.view(B, self.num_luts, 3, 33, 33, 33)

        # 3. 生成动态融合权重
        lut_weights = self.weight_generator(token_features)
        lut_weights = F.softmax(lut_weights, dim=1)  # [B, num_luts]

        # 4. 应用并加权融合所有 LUT
        fused_lut_output = 0
        for i in range(self.num_luts):
            single_lut = lut_params[:, i]
            weight = lut_weights[:, i].view(B, 1, 1, 1)  # [B, 1, 1, 1] for broadcasting
            lut_output = self.apply_3dlut(recon_image, single_lut)
            fused_lut_output += lut_output * weight

        # 5. 处理残差分支（原始图像分支）
        processed_input = self.residual_processor(img)

        # 6. 最终融合，并添加全局残差连接
        combined = torch.cat([fused_lut_output, processed_input], dim=1)
        final_output = self.fusion_layer(combined) + img

        return final_output


# =====================================================================================
# 4. 最终的整合模型
# =====================================================================================
class CASTWithLUT(nn.Module):
    """
    组合了 CAST 分词器和高效 LUT 模块的最终模型。
    """

    def __init__(self, cast_config, lut_config):
        super(CASTWithLUT, self).__init__()
        self.cast = CAST(**cast_config)
        self.lut = LUTModule(**lut_config)

    def forward(self, img):
        # 1. 通过 CAST 分词器得到重建图像和 token
        cast_output = self.cast(img)
        recon_image = cast_output['recon_image']

        # 2. 关键适配：将 H 和 V 通道的 token 序列在长度维度上拼接
        h_tokens = cast_output['h']  # [B, seq_len]
        v_tokens = cast_output['v']  # [B, seq_len]
        combined_tokens = torch.cat([h_tokens, v_tokens], dim=1)  # [B, 2 * seq_len]

        # 3. 将拼接后的 token 和图像传入 LUT 模块
        lut_output = self.lut(combined_tokens, img, recon_image)

        return {
            'recon_image': recon_image,
            'lut_output': lut_output,
            'quantization_loss': cast_output['quantization_loss']
        }


# =====================================================================================
# 5. 使用示例
# =====================================================================================
if __name__ == "__main__":
    # --- 配置参数 ---
    IMAGE_SIZE = 256
    VOCAB_SIZE = 8192
    EMBEDDING_DIM = 256
    # 编码器将 256x256 图像下采样 4 次 (256 -> 128 -> 64 -> 32 -> 16)
    # 所以特征图大小为 16x16，序列长度为 16 * 16 = 256
    SEQUENCE_LENGTH = (IMAGE_SIZE // 16) ** 2

    cast_config = {
        'image_size': IMAGE_SIZE,
        'vocab_size': VOCAB_SIZE,
        'embedding_dim': EMBEDDING_DIM,
        'sequence_length': SEQUENCE_LENGTH,
        'commitment_cost': 0.25,
        'decay': 0.99
    }

    lut_config = {
        'vocab_size': VOCAB_SIZE,
        'embedding_dim': EMBEDDING_DIM,
        'num_luts': 8,
        'hidden_dim': 512
    }

    # --- 模型初始化 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CASTWithLUT(cast_config, lut_config).to(device)

    # --- 打印模型信息 ---
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型已加载到: {device}")
    print(f"总可训练参数量: {total_params:,}")
    print("-" * 30)

    # --- 创建测试输入 ---
    batch_size = 4
    dummy_img = torch.rand(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)

    # --- 前向传播测试 ---
    model.train()  # 确保 VQ-VAE 的 EMA 更新开启
    output = model(dummy_img)

    # --- 打印输出形状 ---
    print(f"输入图像形状: {dummy_img.shape}")
    print(f"CAST 重建图像形状: {output['recon_image'].shape}")
    print(f"LUT 最终输出形状: {output['lut_output'].shape}")
    print(f"量化损失: {output['quantization_loss'].item():.6f}")

    # --- 检查损失计算 ---
    target_img = torch.rand(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)  # 假设这是目标真值图像
    recon_loss = F.l1_loss(output['recon_image'], target_img)
    final_loss = F.l1_loss(output['lut_output'], target_img)
    quantization_loss = output['quantization_loss']
    total_loss = final_loss + recon_loss + quantization_loss

    print("-" * 30)
    print("损失计算示例:")
    print(f"  重建损失 (L1): {recon_loss.item():.4f}")
    print(f"  最终输出损失 (L1): {final_loss.item():.4f}")
    print(f"  量化损失: {quantization_loss.item():.4f}")
    print(f"  总损失 (示例): {total_loss.item():.4f}")