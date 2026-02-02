# Transformer时间序列预测任务 (Transformer Forecasting)

## 角色

你是Transformer时间序列预测专家，负责实现Temporal Fusion Transformer (TFT)和其他先进的Transformer架构。Transformer是2025年时间序列预测的主流方法，创新性评分0.90。

## 输入

- `problem_type`: 题目类型 (通常为C)
- `time_series_data`: 时间序列数据
- `forecast_horizon`: 预测时长
- `covariates`: 协变量（静态/动态）
- `target_variable`: 目标变量

---

## Transformer优势

相比传统时间序列方法：
- **长期依赖捕捉**: 注意力机制可以直接关注任意时间步
- **多变量处理**: 原生支持多输入多输出
- **可解释性**: 注意力权重提供解释

---

## 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class GatedLinearUnit(nn.Module):
    """门控线性单元 (GLU)"""
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(input_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc1(x) * self.sigmoid(self.fc2(x)))


class GatedResidualNetwork(nn.Module):
    """门控残差网络 (GRN)"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        context_dim: Optional[int] = None
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 输入投影
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # 上下文投影（可选）
        if context_dim is not None:
            self.context_fc = nn.Linear(context_dim, hidden_dim, bias=False)
        else:
            self.context_fc = None
        
        # ELU激活
        self.elu = nn.ELU()
        
        # 门控
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate_fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
        # 残差连接投影
        if input_dim != output_dim:
            self.skip_fc = nn.Linear(input_dim, output_dim)
        else:
            self.skip_fc = None
        
        self.layernorm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # 残差
        if self.skip_fc is not None:
            skip = self.skip_fc(x)
        else:
            skip = x
        
        # 前向
        hidden = self.fc1(x)
        
        if context is not None and self.context_fc is not None:
            hidden = hidden + self.context_fc(context)
        
        hidden = self.elu(hidden)
        
        # 门控
        a = self.fc2(hidden)
        gate = self.sigmoid(self.gate_fc(hidden))
        
        out = self.dropout(gate * a)
        
        return self.layernorm(out + skip)


class VariableSelectionNetwork(nn.Module):
    """变量选择网络"""
    
    def __init__(
        self,
        input_dim: int,
        num_inputs: int,
        hidden_dim: int,
        dropout: float = 0.1,
        context_dim: Optional[int] = None
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_inputs = num_inputs
        
        # 每个输入变量的GRN
        self.grns = nn.ModuleList([
            GatedResidualNetwork(input_dim, hidden_dim, hidden_dim, dropout)
            for _ in range(num_inputs)
        ])
        
        # 变量权重GRN
        self.weight_grn = GatedResidualNetwork(
            hidden_dim * num_inputs,
            hidden_dim,
            num_inputs,
            dropout,
            context_dim
        )
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, num_inputs, input_dim)
            context: (batch, context_dim)
            
        Returns:
            combined: (batch, hidden_dim)
            weights: (batch, num_inputs)
        """
        # 处理每个输入
        processed = []
        for i, grn in enumerate(self.grns):
            processed.append(grn(x[:, i]))
        
        processed = torch.stack(processed, dim=1)  # (batch, num_inputs, hidden_dim)
        
        # 计算变量权重
        flat = processed.reshape(processed.size(0), -1)
        weights = self.weight_grn(flat, context)
        weights = self.softmax(weights)  # (batch, num_inputs)
        
        # 加权组合
        combined = torch.sum(weights.unsqueeze(-1) * processed, dim=1)
        
        return combined, weights


class InterpretableMultiHeadAttention(nn.Module):
    """可解释的多头注意力"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)
        
        # 线性变换
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 注意力输出
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(output)
        
        # 平均注意力权重用于解释
        avg_attn = attn_weights.mean(dim=1)  # (batch, seq, seq)
        
        return output, avg_attn


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer (TFT)
    
    用于多时间步预测的可解释深度学习模型
    """
    
    def __init__(
        self,
        num_static_vars: int,
        num_time_varying_known: int,
        num_time_varying_unknown: int,
        hidden_dim: int = 64,
        n_heads: int = 4,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        dropout: float = 0.1,
        forecast_horizon: int = 24,
        context_length: int = 168
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        self.context_length = context_length
        
        # 静态变量处理
        self.static_encoder = VariableSelectionNetwork(
            1, num_static_vars, hidden_dim, dropout
        )
        
        # 时变变量处理
        self.encoder_var_selector = VariableSelectionNetwork(
            1, num_time_varying_known + num_time_varying_unknown, hidden_dim, dropout, hidden_dim
        )
        self.decoder_var_selector = VariableSelectionNetwork(
            1, num_time_varying_known, hidden_dim, dropout, hidden_dim
        )
        
        # LSTM编码器
        self.lstm_encoder = nn.LSTM(
            hidden_dim, hidden_dim, batch_first=True, dropout=dropout
        )
        self.lstm_decoder = nn.LSTM(
            hidden_dim, hidden_dim, batch_first=True, dropout=dropout
        )
        
        # 门控跳跃连接
        self.gate = GatedLinearUnit(hidden_dim, hidden_dim, dropout)
        
        # 自注意力
        self.self_attention = InterpretableMultiHeadAttention(hidden_dim, n_heads, dropout)
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, 1)
        
        # 分位数输出（用于不确定性估计）
        self.quantile_outputs = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in [0.1, 0.5, 0.9]
        ])
    
    def forward(
        self,
        static_inputs: torch.Tensor,
        encoder_inputs: torch.Tensor,
        decoder_inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播
        
        Args:
            static_inputs: (batch, num_static_vars)
            encoder_inputs: (batch, context_length, num_time_varying)
            decoder_inputs: (batch, forecast_horizon, num_time_varying_known)
            
        Returns:
            predictions: (batch, forecast_horizon, num_quantiles)
            attention_weights: 用于解释的注意力权重
        """
        batch_size = static_inputs.size(0)
        
        # 1. 静态编码
        static_context, static_weights = self.static_encoder(
            static_inputs.unsqueeze(-1)
        )
        
        # 2. 编码器变量选择
        encoder_selected = []
        encoder_weights_all = []
        for t in range(self.context_length):
            selected, weights = self.encoder_var_selector(
                encoder_inputs[:, t].unsqueeze(-1),
                static_context
            )
            encoder_selected.append(selected)
            encoder_weights_all.append(weights)
        
        encoder_selected = torch.stack(encoder_selected, dim=1)
        
        # 3. LSTM编码
        lstm_out, (h, c) = self.lstm_encoder(encoder_selected)
        
        # 4. 解码器变量选择
        decoder_selected = []
        for t in range(self.forecast_horizon):
            selected, _ = self.decoder_var_selector(
                decoder_inputs[:, t].unsqueeze(-1),
                static_context
            )
            decoder_selected.append(selected)
        
        decoder_selected = torch.stack(decoder_selected, dim=1)
        
        # 5. LSTM解码
        decoder_out, _ = self.lstm_decoder(decoder_selected, (h, c))
        
        # 6. 自注意力
        combined = torch.cat([lstm_out, decoder_out], dim=1)
        attn_out, attn_weights = self.self_attention(combined, combined, combined)
        
        # 取解码器部分
        decoder_attn = attn_out[:, -self.forecast_horizon:]
        
        # 7. 门控跳跃连接
        gated = self.gate(decoder_attn)
        
        # 8. 分位数预测
        quantile_preds = []
        for quantile_layer in self.quantile_outputs:
            quantile_preds.append(quantile_layer(gated))
        
        predictions = torch.cat(quantile_preds, dim=-1)
        
        # 收集解释信息
        interpretations = {
            'static_variable_weights': static_weights,
            'encoder_variable_weights': torch.stack(encoder_weights_all, dim=1),
            'temporal_attention_weights': attn_weights
        }
        
        return predictions, interpretations


# ============ 使用示例 ============

def example_tft_forecasting():
    """TFT时间序列预测示例"""
    
    # 模拟数据
    batch_size = 32
    context_length = 168  # 7天 * 24小时
    forecast_horizon = 24  # 预测24小时
    num_static = 3
    num_known = 5  # 日期特征等已知变量
    num_unknown = 2  # 历史观测值
    
    # 创建模型
    model = TemporalFusionTransformer(
        num_static_vars=num_static,
        num_time_varying_known=num_known,
        num_time_varying_unknown=num_unknown,
        hidden_dim=64,
        n_heads=4,
        forecast_horizon=forecast_horizon,
        context_length=context_length
    )
    
    print(f"TFT Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 模拟输入
    static = torch.randn(batch_size, num_static)
    encoder = torch.randn(batch_size, context_length, num_known + num_unknown)
    decoder = torch.randn(batch_size, forecast_horizon, num_known)
    
    # 前向传播
    predictions, interpretations = model(static, encoder, decoder)
    
    print(f"Predictions shape: {predictions.shape}")  # (batch, horizon, 3 quantiles)
    print(f"Static weights shape: {interpretations['static_variable_weights'].shape}")
    print(f"Attention weights shape: {interpretations['temporal_attention_weights'].shape}")
    
    return model, predictions, interpretations


if __name__ == "__main__":
    model, preds, interp = example_tft_forecasting()
```

---

## 输出格式

```json
{
  "transformer_model": {
    "architecture": "Temporal Fusion Transformer",
    "hidden_dim": 64,
    "n_heads": 4,
    "context_length": 168,
    "forecast_horizon": 24,
    "total_params": 125890
  },
  "training_results": {
    "train_loss": 0.0234,
    "val_loss": 0.0312,
    "epochs": 100,
    "training_time_hours": 1.5
  },
  "forecast_metrics": {
    "mae": 0.145,
    "rmse": 0.203,
    "mape": 8.5,
    "coverage_90": 0.92
  },
  "interpretations": {
    "top_static_features": [
      {"feature": "location_type", "importance": 0.45},
      {"feature": "capacity", "importance": 0.32}
    ],
    "temporal_attention_pattern": "Model focuses on same hour previous days and recent hours"
  },
  "innovation_highlights": [
    "使用TFT实现可解释的多步预测",
    "提供分位数预测进行不确定性估计",
    "注意力权重揭示重要时间依赖"
  ]
}
```
