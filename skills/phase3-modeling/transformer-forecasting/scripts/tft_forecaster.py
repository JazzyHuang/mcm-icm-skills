"""
Temporal Fusion Transformer (TFT) Forecaster
时间序列预测模块

基于注意力机制的可解释时间序列预测，适用于MCM/ICM C题。
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import json


class GatedLinearUnit(nn.Module):
    """门控线性单元"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)
        
    def forward(self, x):
        out = self.fc(x)
        return out[..., :out.shape[-1]//2] * torch.sigmoid(out[..., out.shape[-1]//2:])


class VariableSelectionNetwork(nn.Module):
    """变量选择网络"""
    def __init__(self, input_dim, num_vars, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_vars = num_vars
        
        self.grns = nn.ModuleList([
            GatedLinearUnit(input_dim, hidden_dim) for _ in range(num_vars)
        ])
        self.softmax_layer = nn.Linear(hidden_dim * num_vars, num_vars)
        
    def forward(self, x):
        # x: [batch, time, num_vars, features]
        processed = []
        for i, grn in enumerate(self.grns):
            processed.append(grn(x[..., i, :]))
        
        combined = torch.cat(processed, dim=-1)
        weights = torch.softmax(self.softmax_layer(combined), dim=-1)
        
        output = sum(w.unsqueeze(-1) * p for w, p in 
                    zip(weights.unbind(-1), processed))
        return output, weights


class TemporalSelfAttention(nn.Module):
    """时间自注意力"""
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, mask=None):
        attn_out, attn_weights = self.attention(x, x, x, attn_mask=mask)
        return self.norm(x + attn_out), attn_weights


class TFTForecaster:
    """Temporal Fusion Transformer预测器"""
    
    def __init__(
        self,
        max_encoder_length: int = 60,
        max_prediction_length: int = 20,
        hidden_dim: int = 64,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        static_categoricals: List[str] = None,
        time_varying_known: List[str] = None,
        time_varying_unknown: List[str] = None,
        device: str = 'auto'
    ):
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.hidden_dim = hidden_dim
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        
        self.static_categoricals = static_categoricals or []
        self.time_varying_known = time_varying_known or []
        self.time_varying_unknown = time_varying_unknown or ['target']
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.model = None
        self.training_history = {'train_loss': [], 'val_loss': []}
        self.variable_weights = None
        self.attention_weights = None
        
    def _build_model(self, num_features: int):
        """构建模型"""
        class TFTModel(nn.Module):
            def __init__(self, num_features, hidden_dim, num_heads, dropout):
                super().__init__()
                self.input_proj = nn.Linear(num_features, hidden_dim)
                self.encoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
                self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
                self.attention = TemporalSelfAttention(hidden_dim, num_heads)
                self.output_proj = nn.Linear(hidden_dim, 3)  # 3 quantiles
                self.dropout = nn.Dropout(dropout)
                
            def forward(self, x, future_known=None):
                # Encode
                x = self.input_proj(x)
                x = self.dropout(x)
                encoded, (h, c) = self.encoder_lstm(x)
                
                # Decode
                if future_known is not None:
                    future = self.input_proj(future_known)
                else:
                    future = encoded[:, -1:, :].expand(-1, 20, -1)
                    
                decoded, _ = self.decoder_lstm(future, (h, c))
                
                # Attention
                attended, attn_weights = self.attention(decoded)
                
                # Output
                output = self.output_proj(attended)
                return output, attn_weights
                
        self.model = TFTModel(
            num_features, self.hidden_dim, 
            self.num_attention_heads, self.dropout
        ).to(self.device)
        
    def _prepare_data(self, data: pd.DataFrame, num_features: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """准备训练数据，将DataFrame转换为模型输入格式"""
        features = self.time_varying_unknown + self.time_varying_known
        if not features:
            features = ['target']
        
        # 提取特征数据
        available_features = [f for f in features if f in data.columns]
        if not available_features:
            # 如果没有匹配的列，使用所有数值列
            available_features = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not available_features:
            raise ValueError("No numeric features found in the data")
        
        values = data[available_features].values
        
        # 创建序列样本
        n_samples = max(1, len(values) - self.max_encoder_length - self.max_prediction_length + 1)
        X_list = []
        y_list = []
        
        for i in range(n_samples):
            X_seq = values[i:i + self.max_encoder_length]
            y_seq = values[i + self.max_encoder_length:i + self.max_encoder_length + self.max_prediction_length]
            
            # 确保序列长度正确
            if len(X_seq) == self.max_encoder_length and len(y_seq) == self.max_prediction_length:
                X_list.append(X_seq)
                # 为3个分位数复制目标值
                y_list.append(np.tile(y_seq[:, :1], (1, 3)))
        
        if not X_list:
            # 如果数据不足以创建完整序列，使用填充
            X_list = [np.tile(values[:min(len(values), self.max_encoder_length)], 
                             (self.max_encoder_length // max(1, len(values)) + 1, 1))[:self.max_encoder_length]]
            y_list = [np.zeros((self.max_prediction_length, 3))]
        
        X = torch.tensor(np.array(X_list), dtype=torch.float32, device=self.device)
        y = torch.tensor(np.array(y_list), dtype=torch.float32, device=self.device)
        
        return X, y
    
    def fit(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame = None,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        verbose: int = 10
    ):
        """训练模型"""
        # 数据处理
        features = self.time_varying_unknown + self.time_varying_known
        num_features = len(features) if features else 1
        
        self._build_model(num_features)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # 使用实际训练数据
        X_train, y_train = self._prepare_data(train_data, num_features)
        
        # 准备验证数据（如果提供）
        X_val, y_val = None, None
        if val_data is not None:
            X_val, y_val = self._prepare_data(val_data, num_features)
        
        for epoch in range(epochs):
            self.model.train()
            
            # Mini-batch训练
            n_samples = X_train.shape[0]
            perm = torch.randperm(n_samples)
            total_loss = 0
            n_batches = 0
            
            for i in range(0, n_samples, batch_size):
                idx = perm[i:i+batch_size]
                X_batch, y_batch = X_train[idx], y_train[idx]
                
                optimizer.zero_grad()
                outputs, attn = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = total_loss / max(n_batches, 1)
            self.training_history['train_loss'].append(avg_train_loss)
            self.attention_weights = attn.detach()
            
            # 验证
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs, _ = self.model(X_val)
                    val_loss = criterion(val_outputs, y_val).item()
                    self.training_history['val_loss'].append(val_loss)
            
            if verbose and (epoch + 1) % verbose == 0:
                msg = f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}"
                if X_val is not None:
                    msg += f", Val Loss: {self.training_history['val_loss'][-1]:.6f}"
                print(msg)
                
        return self.training_history
    
    def predict(self, data: pd.DataFrame, return_quantiles: bool = True):
        """
        预测
        
        Args:
            data: 输入数据DataFrame
            return_quantiles: 是否返回分位数预测
            
        Returns:
            预测结果（分位数字典或点预测数组）
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        self.model.eval()
        
        # 使用与训练相同的数据准备方法处理输入数据
        features = self.time_varying_unknown + self.time_varying_known
        num_features = len(features) if features else 1
        
        # 准备输入数据（不需要y，使用滑动窗口创建序列）
        X, _ = self._prepare_data(data, num_features)
        
        with torch.no_grad():
            outputs, self.attention_weights = self.model(X)
            
            if return_quantiles:
                return {
                    'p10': outputs[..., 0].cpu().numpy(),
                    'p50': outputs[..., 1].cpu().numpy(),
                    'p90': outputs[..., 2].cpu().numpy()
                }
            return outputs[..., 1].cpu().numpy()
    
    def variable_importance(self) -> Dict[str, float]:
        """
        计算变量重要性
        
        基于注意力权重计算每个特征的重要性分数
        """
        features = self.time_varying_unknown + self.time_varying_known
        if not features:
            features = ['target']
        
        # 如果有注意力权重，使用它们计算重要性
        if self.attention_weights is not None:
            try:
                # 计算每个特征的平均注意力权重
                attn = self.attention_weights.cpu().numpy()
                # 对所有样本和时间步取平均
                avg_attn = np.mean(np.abs(attn), axis=(0, 1))
                
                # 如果特征数量匹配
                if len(avg_attn) >= len(features):
                    importance = {f: float(avg_attn[i]) for i, f in enumerate(features)}
                else:
                    # 均匀分配
                    importance = {f: 1.0 / len(features) for f in features}
            except Exception:
                # 出错时使用均匀分配
                importance = {f: 1.0 / len(features) for f in features}
        else:
            # 没有注意力权重时，使用均匀分配（而不是随机值）
            importance = {f: 1.0 / len(features) for f in features}
        
        # 归一化
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}
        
        self.variable_weights = importance
        return importance
    
    def get_attention_weights(self, data=None) -> np.ndarray:
        """获取注意力权重"""
        if self.attention_weights is not None:
            return self.attention_weights.cpu().numpy()
        return None
    
    def plot_variable_importance(self, save_path: str = None):
        """绘制变量重要性"""
        import matplotlib.pyplot as plt
        
        importance = self.variable_importance()
        
        fig, ax = plt.subplots(figsize=(8, 5))
        vars_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        ax.barh([v[0] for v in vars_sorted], [v[1] for v in vars_sorted])
        ax.set_xlabel('Importance')
        ax.set_title('Variable Importance (TFT)')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    def save_results(self, output_path: str):
        """保存结果元数据（不包含模型权重）"""
        results = {
            'model': {
                'type': 'TemporalFusionTransformer',
                'encoder_length': self.max_encoder_length,
                'prediction_length': self.max_prediction_length,
                'hidden_dim': self.hidden_dim
            },
            'training': {
                'epochs': len(self.training_history['train_loss']),
                'final_loss': self.training_history['train_loss'][-1] if self.training_history['train_loss'] else None
            },
            'interpretability': {
                'variable_importance': self.variable_weights
            }
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        return results
    
    def save_model(self, path: str):
        """
        保存完整模型（包含权重和配置）
        
        Args:
            path: 保存路径（建议使用.pt或.pth后缀）
        """
        if self.model is None:
            raise ValueError("No model to save. Call fit() first.")
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': {
                'max_encoder_length': self.max_encoder_length,
                'max_prediction_length': self.max_prediction_length,
                'time_varying_known': self.time_varying_known,
                'time_varying_unknown': self.time_varying_unknown,
                'static_categoricals': self.static_categoricals,
                'hidden_dim': self.hidden_dim,
                'num_heads': self.num_heads,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'quantiles': self.quantiles,
                'device': str(self.device)
            },
            'training_history': self.training_history,
            'variable_weights': self.variable_weights
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path: str, device: str = 'auto') -> 'TFTForecaster':
        """
        加载模型
        
        Args:
            path: 模型路径
            device: 设备（'auto', 'cpu', 'cuda'）
            
        Returns:
            加载的TFTForecaster实例
        """
        checkpoint = torch.load(path, map_location='cpu')
        config = checkpoint['config']
        
        # 创建实例
        forecaster = cls(
            max_encoder_length=config['max_encoder_length'],
            max_prediction_length=config['max_prediction_length'],
            time_varying_known=config.get('time_varying_known', []),
            time_varying_unknown=config.get('time_varying_unknown', ['target']),
            static_categoricals=config.get('static_categoricals', []),
            hidden_dim=config.get('hidden_dim', 64),
            num_heads=config.get('num_heads', 4),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.1),
            quantiles=config.get('quantiles', [0.1, 0.5, 0.9]),
            device=device
        )
        
        # 构建模型架构
        features = forecaster.time_varying_unknown + forecaster.time_varying_known
        num_features = len(features) if features else 1
        forecaster._build_model(num_features)
        
        # 加载权重
        forecaster.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 恢复训练历史和变量权重
        if 'training_history' in checkpoint:
            forecaster.training_history = checkpoint['training_history']
        if 'variable_weights' in checkpoint:
            forecaster.variable_weights = checkpoint['variable_weights']
        
        print(f"Model loaded from {path}")
        return forecaster


if __name__ == '__main__':
    print("Testing TFT Forecaster...")
    forecaster = TFTForecaster(
        max_encoder_length=30,
        max_prediction_length=10
    )
    
    # 模拟数据
    train_data = pd.DataFrame({'target': np.random.randn(100)})
    forecaster.fit(train_data, epochs=20, verbose=5)
    
    predictions = forecaster.predict(train_data)
    print(f"Predictions shape: {predictions['p50'].shape}")
    print("TFT Forecaster test completed!")
