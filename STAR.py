import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize
from layers.ChebyKANLayer import ChebyKANLinear
from layers.TimeDART_EncDec import Diffusion
from utils.RevIN import RevIN


class EmbeddingSTAR(nn.Module):
    """在embedding空间应用的STAR模块 - 简化随机池化版本"""

    def __init__(self, d_model, d_core=None):
        super().__init__()
        self.d_model = d_model
        self.d_core = d_core if d_core is not None else d_model // 2

        # 核心表示生成网络 - 参考SOFTS的两步设计
        self.gen1 = nn.Linear(d_model, d_model)
        self.gen2 = nn.Linear(d_model, self.d_core)

        # 融合网络 - 参考SOFTS风格
        self.gen3 = nn.Linear(d_model + self.d_core, d_model)
        self.gen4 = nn.Linear(d_model, d_model)

    def stochastic_pooling(self, x):
        """SOFTS风格的简化随机池化"""
        # x: [B, T, d_core]
        batch_size, seq_len, core_dim = x.shape

        if self.training:
            # 训练时：按概率随机采样 - 参考SOFTS的简洁实现
            ratio = F.softmax(x, dim=1)  # [B, T, d_core] - 在时间维度计算概率
            ratio = ratio.permute(0, 2, 1)  # [B, d_core, T]
            ratio = ratio.reshape(-1, seq_len)  # [B*d_core, T]

            # 为每个(batch, feature)对采样一个时间点
            indices = torch.multinomial(ratio, 1)  # [B*d_core, 1]
            indices = indices.view(batch_size, core_dim, 1)  # [B, d_core, 1]
            indices = indices.permute(0, 2, 1)  # [B, 1, d_core]

            # 收集采样结果
            core = torch.gather(x, 1, indices)  # [B, 1, d_core]
        else:
            # 测试时：加权平均
            weight = F.softmax(x, dim=1)  # [B, T, d_core]
            core = torch.sum(x * weight, dim=1, keepdim=True)  # [B, 1, d_core]

        return core

    def forward(self, x):
        """
        x: [B, T, d_model] - embedding后的特征
        输出: [B, T, d_model] - 增强后的特征
        """
        B, T, D = x.shape

        # 第一步：生成中间表示 - 参考SOFTS
        combined_mean = F.gelu(self.gen1(x))  # [B, T, d_model]

        # 第二步：生成核心表示候选
        combined_mean = self.gen2(combined_mean)  # [B, T, d_core]

        # 随机池化生成全局核心
        global_core = self.stochastic_pooling(combined_mean)  # [B, 1, d_core]

        # 将全局核心分发到每个时间步
        global_core_expanded = global_core.repeat(1, T, 1)  # [B, T, d_core]

        # 融合原始特征和全局核心 - 参考SOFTS的fusion
        fused_input = torch.cat([x, global_core_expanded], dim=-1)  # [B, T, d_model + d_core]
        fused_output = F.gelu(self.gen3(fused_input))  # [B, T, d_model]
        fused_output = self.gen4(fused_output)  # [B, T, d_model]

        # 残差连接
        return fused_output


class SimpleFrequencyProcessor(nn.Module):
    """简单的频域处理器"""

    def __init__(self, seq_len, pred_len, d_model):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model

        # 计算有效频率点
        self.valid_fre_points_in = int((seq_len + 1) / 2 + 0.5)
        self.valid_fre_points_out = int((pred_len + 1) / 2 + 0.5)

        # 频域特征处理网络 - 分别处理实部和虚部
        self.freq_processor_real = nn.Sequential(
            nn.Linear(self.valid_fre_points_in, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.valid_fre_points_out)
        )

        self.freq_processor_imag = nn.Sequential(
            nn.Linear(self.valid_fre_points_in, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.valid_fre_points_out)
        )

    def forward(self, x):
        """
        x: [B, T, N] - 时域输入
        输出: [B, pred_len, N] - 频域预测结果
        """
        B, T, N = x.shape

        # 转换到频域 - 在时间维度做FFT
        x_freq = torch.fft.rfft(x, dim=1, norm='ortho')  # [B, valid_fre_points_in, N]

        # 分离实部和虚部
        real_part = x_freq.real  # [B, valid_fre_points_in, N]
        imag_part = x_freq.imag  # [B, valid_fre_points_in, N]

        # 处理每个通道的频域特征
        real_out = []
        imag_out = []

        for i in range(N):
            # 处理第i个通道
            real_i = self.freq_processor_real(real_part[:, :, i])  # [B, valid_fre_points_out]
            imag_i = self.freq_processor_imag(imag_part[:, :, i])  # [B, valid_fre_points_out]
            real_out.append(real_i)
            imag_out.append(imag_i)

        # 重新组合
        real_output = torch.stack(real_out, dim=-1)  # [B, valid_fre_points_out, N]
        imag_output = torch.stack(imag_out, dim=-1)  # [B, valid_fre_points_out, N]

        # 构建复数输出
        freq_output = torch.complex(real_output, imag_output)

        # 转换回时域
        time_output = torch.fft.irfft(freq_output, n=self.pred_len, dim=1, norm='ortho')  # [B, pred_len, N]

        return time_output


class TimeFrquencyFusion(nn.Module):
    """时频域融合模块 - 修复维度问题"""

    def __init__(self, pred_len, n_channels):
        super().__init__()
        self.pred_len = pred_len
        self.n_channels = n_channels

        # 学习时频权重
        self.time_freq_weights = nn.Parameter(torch.tensor([0.7, 0.3]))  # [time, freq]

        # 🔥 修复：门控机制使用正确的维度
        self.gate = nn.Sequential(
            nn.Linear(2, 1),  # 输入是2个标量（时域值和频域值），输出1个权重
            nn.Sigmoid()
        )

    def forward(self, time_output, freq_output):
        """
        time_output: [B, pred_len, N] - 时域预测
        freq_output: [B, pred_len, N] - 频域预测
        """
        # 方法1：简单加权融合
        weights = F.softmax(self.time_freq_weights, dim=0)
        simple_fusion = weights[0] * time_output + weights[1] * freq_output

        # 方法2：逐点自适应门控融合
        B, T, N = time_output.shape
        adaptive_outputs = []

        for i in range(N):  # 对每个通道分别处理
            time_channel = time_output[:, :, i:i + 1]  # [B, T, 1]
            freq_channel = freq_output[:, :, i:i + 1]  # [B, T, 1]

            # 逐时间点计算门控权重
            channel_outputs = []
            for t in range(T):
                # 取当前时间点的值作为门控输入
                gate_input = torch.stack([
                    time_channel[:, t, 0],  # [B]
                    freq_channel[:, t, 0]  # [B]
                ], dim=-1)  # [B, 2]

                gate_weight = self.gate(gate_input)  # [B, 1]

                # 融合当前时间点
                fused_point = gate_weight * time_channel[:, t:t + 1, :] + (1 - gate_weight) * freq_channel[:, t:t + 1,
                                                                                              :]
                channel_outputs.append(fused_point)

            channel_output = torch.cat(channel_outputs, dim=1)  # [B, T, 1]
            adaptive_outputs.append(channel_output)

        adaptive_fusion = torch.cat(adaptive_outputs, dim=-1)  # [B, T, N]

        # 最终输出：结合两种融合方式
        final_output = 0.5 * simple_fusion + 0.5 * adaptive_fusion

        return final_output


class LightweightDiffusion(nn.Module):
    """轻量级扩散模块"""

    def __init__(self, time_steps=20, device='cuda', scheduler='linear'):
        super().__init__()
        self.diffusion = Diffusion(time_steps=time_steps, device=device, scheduler=scheduler)

    def forward(self, x, apply_noise=True):
        if apply_noise and self.training:
            return self.diffusion(x)
        else:
            return x, None, None


class AdaptiveKANMixer(nn.Module):
    """自适应KAN混合器"""

    def __init__(self, d_model, component_type='trend'):
        super().__init__()
        # 根据分量类型选择KAN阶数
        order_map = {'trend': 3, 'seasonal': 5, 'residual': 4}
        order = order_map.get(component_type, 4)

        self.kan_layer = ChebyKANLinear(d_model, d_model, order)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, C = x.shape
        x_kan = self.kan_layer(x.reshape(B * T, C)).reshape(B, T, C)
        x_conv = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x + x_kan + x_conv)


class TCNResidualProcessor(nn.Module):
    """TCN残差处理器 - 专门用于处理残差分量"""

    def __init__(self, configs, num_levels=4):
        super().__init__()
        self.num_levels = num_levels

        # TCN参数
        input_channels = configs.d_model
        hidden_channels = configs.d_model
        kernel_size = 3

        # 构建多层膨胀卷积
        self.tcn_layers = nn.ModuleList()
        self.residual_layers = nn.ModuleList()

        for i in range(num_levels):
            dilation = 2 ** i  # 膨胀率：1, 2, 4, 8
            padding = (kernel_size - 1) * dilation

            # 因果卷积层
            conv_layer = nn.Conv1d(
                input_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation
            )

            # 层归一化和激活
            layer_norm = nn.LayerNorm(hidden_channels)
            dropout = nn.Dropout(configs.dropout)

            # 残差连接的1x1卷积（如果维度不匹配）
            residual_conv = nn.Conv1d(input_channels, hidden_channels, 1) if input_channels != hidden_channels else None

            self.tcn_layers.append(nn.ModuleDict({
                'conv': conv_layer,
                'norm': layer_norm,
                'dropout': dropout
            }))
            self.residual_layers.append(residual_conv)

            input_channels = hidden_channels

        # 最终输出层
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_channels, configs.d_ff),
            nn.GELU(),
            nn.Linear(configs.d_ff, configs.d_model),
            nn.Dropout(configs.dropout)
        )

    def forward(self, x):
        """
        x: [B, T, d_model] - 输入特征
        输出: [B, T, d_model] - TCN处理后的特征
        """
        B, T, C = x.shape

        # 转换为卷积格式 [B, C, T]
        x_conv = x.transpose(1, 2)  # [B, d_model, T]

        # 逐层TCN处理
        for i, (tcn_layer, residual_layer) in enumerate(zip(self.tcn_layers, self.residual_layers)):
            # 保存输入用于残差连接
            residual = x_conv

            # 因果卷积
            out = tcn_layer['conv'](x_conv)

            # 因果性：移除未来信息（右侧padding）
            if out.shape[2] > T:
                out = out[:, :, :T]

            # 转换回时序格式进行归一化
            out = out.transpose(1, 2)  # [B, T, C]
            out = tcn_layer['norm'](out)
            out = F.gelu(out)
            out = tcn_layer['dropout'](out)
            out = out.transpose(1, 2)  # [B, C, T]

            # 残差连接
            if residual_layer is not None:
                residual = residual_layer(residual)

            # 确保维度匹配
            if residual.shape[2] != out.shape[2]:
                min_len = min(residual.shape[2], out.shape[2])
                residual = residual[:, :, :min_len]
                out = out[:, :, :min_len]

            x_conv = out + residual

        # 转换回时序格式
        x_out = x_conv.transpose(1, 2)  # [B, T, d_model]

        # 最终投影
        output = self.output_projection(x_out)

        return output


class ComponentProcessor(nn.Module):
    """分量处理器 - 残差分量使用TCN"""

    def __init__(self, configs, component_type):
        super().__init__()
        self.component_type = component_type

        if component_type == 'trend':
            self.processor = nn.Sequential(
                AdaptiveKANMixer(configs.d_model, 'trend'),
                nn.Linear(configs.d_model, configs.d_model),
                nn.GELU(),
                nn.Dropout(configs.dropout)
            )
        elif component_type == 'seasonal':
            # 为seasonal分量添加简化的STAR模块
            self.embedding_star = EmbeddingSTAR(configs.d_model, configs.d_model // 2)
            self.diffusion = LightweightDiffusion(time_steps=20, device=configs.device)
            self.processor = AdaptiveKANMixer(configs.d_model, 'seasonal')
        else:  # residual - 🔥 使用TCN处理器
            self.processor = TCNResidualProcessor(configs, num_levels=4)

    def forward(self, x):
        if self.component_type == 'seasonal':
            # 先应用简化的embedding级别STAR模块
            x_star = self.embedding_star(x)

            # 然后应用扩散和处理
            if self.training:
                x_noise, noise, t = self.diffusion(x_star, apply_noise=True)
                return self.processor(x_noise)
            else:
                return self.processor(x_star)
        else:
            return self.processor(x)


class Model(nn.Module):
    """简单并行时频双分支STAR模型"""

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # 🔥 时域分支：原有的STAR处理流程
        self.decomposition = series_decomp(configs.moving_avg)

        # 嵌入层
        if configs.channel_independence == 1:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq, configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)

        # 时域分量处理器
        self.trend_processor = ComponentProcessor(configs, 'trend')
        self.seasonal_processor = ComponentProcessor(configs, 'seasonal')
        self.residual_processor = ComponentProcessor(configs, 'residual')

        # 时域预测层
        self.trend_predictor = nn.Linear(configs.seq_len, configs.pred_len)
        self.seasonal_predictor = nn.Linear(configs.seq_len, configs.pred_len)
        self.residual_predictor = nn.Linear(configs.seq_len, configs.pred_len)

        # 时域投影
        if configs.channel_independence == 1:
            self.projection_layer = nn.Linear(configs.d_model, 1, bias=True)
        else:
            self.projection_layer = nn.Linear(configs.d_model, configs.c_out, bias=True)

        # 时域融合权重
        self.time_fusion_weights = nn.Parameter(torch.tensor([0.25, 0.5, 0.25]))

        # 🔥 频域分支：简单的频域处理
        self.frequency_processor = SimpleFrequencyProcessor(
            seq_len=configs.seq_len,
            pred_len=configs.pred_len,
            d_model=configs.d_model
        )

        # 🔥 时频融合模块
        self.time_freq_fusion = TimeFrquencyFusion(
            pred_len=configs.pred_len,
            n_channels=configs.c_out
        )

        # 归一化
        self.revin_layer = RevIN(configs.enc_in, affine=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            return self.forecast(x_enc, x_mark_enc)
        else:
            raise ValueError('Only long_term_forecast implemented')

    def forecast(self, x_enc, x_mark_enc=None):
        B, T, N = x_enc.size()

        # 归一化
        x_enc = self.revin_layer(x_enc, 'norm')

        # 🔥 时域分支处理
        time_output = self.time_domain_branch(x_enc, x_mark_enc, B, T, N)

        # 🔥 频域分支处理
        freq_output = self.frequency_processor(x_enc)  # [B, pred_len, N]

        # 🔥 时频融合
        fused_output = self.time_freq_fusion(time_output, freq_output)

        # 反归一化
        fused_output = self.revin_layer(fused_output, 'denorm')
        return fused_output

    def time_domain_branch(self, x_enc, x_mark_enc, B, T, N):
        """时域分支处理"""
        # 分解
        seasonal, trend = self.decomposition(x_enc)
        residual = x_enc - seasonal - trend

        # 通道独立性处理
        if self.configs.channel_independence == 1:
            seasonal = seasonal.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            trend = trend.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            residual = residual.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        # 嵌入
        if self.configs.channel_independence == 1 and x_mark_enc is not None:
            x_mark_enc_expanded = x_mark_enc.repeat(N, 1, 1)
        else:
            x_mark_enc_expanded = x_mark_enc

        seasonal_emb = self.enc_embedding(seasonal, x_mark_enc_expanded)
        trend_emb = self.enc_embedding(trend, x_mark_enc_expanded)
        residual_emb = self.enc_embedding(residual, x_mark_enc_expanded)

        # 分量处理（seasonal包含STAR模块）
        seasonal_out = self.seasonal_processor(seasonal_emb)
        trend_out = self.trend_processor(trend_emb)
        residual_out = self.residual_processor(residual_emb)

        # 时序预测
        seasonal_pred = self.seasonal_predictor(seasonal_out.permute(0, 2, 1)).permute(0, 2, 1)
        trend_pred = self.trend_predictor(trend_out.permute(0, 2, 1)).permute(0, 2, 1)
        residual_pred = self.residual_predictor(residual_out.permute(0, 2, 1)).permute(0, 2, 1)

        # 投影
        seasonal_pred = self.projection_layer(seasonal_pred)
        trend_pred = self.projection_layer(trend_pred)
        residual_pred = self.projection_layer(residual_pred)

        # 时域加权融合
        weights = F.softmax(self.time_fusion_weights, dim=0)
        time_output = (weights[0] * trend_pred +
                       weights[1] * seasonal_pred +
                       weights[2] * residual_pred)

        # 输出重塑
        if self.configs.channel_independence == 1:
            time_output = time_output.reshape(B, N, self.pred_len, -1)
            if time_output.shape[-1] == 1:
                time_output = time_output.squeeze(-1)
            time_output = time_output.permute(0, 2, 1).contiguous()

        if time_output.shape[-1] > self.configs.c_out:
            time_output = time_output[..., :self.configs.c_out]

        return time_output