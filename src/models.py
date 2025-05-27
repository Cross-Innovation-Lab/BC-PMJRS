from torch import nn
import torch.nn.functional as F
import numpy as np
import math
import sys
import os
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.transformer import TransformerEncoder

# 修复BC-PMJRS模块导入
BCPMJRS_AVAILABLE = False
try:
    # 明确导入路径
    import importlib.util
    bc_pmjrs_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'modules', 'bc_pmjrs.py')
    
    if os.path.exists(bc_pmjrs_path):
        spec = importlib.util.spec_from_file_location("bc_pmjrs", bc_pmjrs_path)
        bc_pmjrs_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bc_pmjrs_module)
        
        MutualInformationEstimator = bc_pmjrs_module.MutualInformationEstimator
        GlobalLocalCrossModalAttention = bc_pmjrs_module.GlobalLocalCrossModalAttention
        AdaptiveFusionModule = bc_pmjrs_module.AdaptiveFusionModule
        
        BCPMJRS_AVAILABLE = True
        print("BC-PMJRS modules loaded successfully via importlib")
    else:
        raise ImportError(f"BC-PMJRS file not found at {bc_pmjrs_path}")
        
except Exception as e:
    print(f"Failed to load BC-PMJRS modules: {e}")
    print("Creating fallback implementations...")
    
    # 创建完整的fallback实现
    class MutualInformationEstimator(nn.Module):
        def __init__(self, feature_dim, temperature=0.07):
            super().__init__()
            self.temperature = temperature
            self.critic = nn.Sequential(
                nn.Linear(feature_dim * 2, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
        
        def forward(self, features1, features2):
            batch_size = features1.size(0)
            
            # 确保特征是2D的
            if features1.dim() > 2:
                features1 = features1.view(batch_size, -1)
            if features2.dim() > 2:
                features2 = features2.view(batch_size, -1)
            
            # 确保特征维度匹配
            if features1.size(1) != features2.size(1):
                min_dim = min(features1.size(1), features2.size(1))
                features1 = features1[:, :min_dim]
                features2 = features2[:, :min_dim]
            
            # Positive pairs (真实配对)
            pos_pairs = torch.cat([features1, features2], dim=1)
            pos_scores = self.critic(pos_pairs)
            
            # Negative pairs (随机配对)
            neg_indices = torch.randperm(batch_size).to(features1.device)
            neg_features2 = features2[neg_indices]
            neg_pairs = torch.cat([features1, neg_features2], dim=1)
            neg_scores = self.critic(neg_pairs)
            
            # 使用InfoNCE损失正确计算互信息
            # 正样本应该有高分数，负样本应该有低分数
            pos_exp = torch.exp(pos_scores / self.temperature)
            neg_exp = torch.exp(neg_scores / self.temperature)
            
            # InfoNCE损失：-log(exp(pos) / (exp(pos) + exp(neg)))
            mi_estimate = -torch.mean(torch.log(pos_exp / (pos_exp + neg_exp) + 1e-8))
            
            return mi_estimate

    class GlobalLocalCrossModalAttention(nn.Module):
        def __init__(self, d_model, n_heads=8, dropout=0.1):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            
            # Global context extraction
            self.global_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.global_proj = nn.Linear(d_model, d_model)
            
            # Local feature enhancement
            self.local_q = nn.Linear(d_model, d_model)
            self.local_k = nn.Linear(d_model, d_model)
            self.local_v = nn.Linear(d_model, d_model)
            
            # Cross-modal projection matrices
            self.cross_modal_proj = nn.ModuleDict({
                'lv': nn.Linear(d_model, d_model),
                'la': nn.Linear(d_model, d_model),
                'vl': nn.Linear(d_model, d_model),
                'va': nn.Linear(d_model, d_model),
                'al': nn.Linear(d_model, d_model),
                'av': nn.Linear(d_model, d_model)
            })
            
            self.dropout = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm(d_model)
        
        def extract_global_context(self, features, mask=None):
            """Extract global context using attention pooling"""
            seq_len = features.size(0)
            
            # Self-attention to get importance weights
            attn_output, attn_weights = self.global_attention(features, features, features, attn_mask=mask)
            
            # Weighted pooling
            if mask is not None:
                mask = mask.float().unsqueeze(-1)
                attn_output = attn_output * mask
                global_context = attn_output.sum(dim=0) / mask.sum(dim=0)
            else:
                global_context = attn_output.mean(dim=0)
            
            return self.global_proj(global_context)
        
        def enhance_local_features(self, local_features, global_context, proj_key):
            """Enhance local features using global context from other modality"""
            # local_features: (seq_len, batch_size, d_model)
            # global_context: (batch_size, d_model)
            
            seq_len = local_features.size(0)
            batch_size = local_features.size(1)
            
            # Project features
            Q = self.local_q(local_features)  # (seq_len, batch_size, d_model)
            K = self.local_k(global_context.unsqueeze(0))  # (1, batch_size, d_model)
            V = self.local_v(global_context.unsqueeze(0))  # (1, batch_size, d_model)
            
            # Compute attention scores
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_model)
            attn_weights = F.softmax(scores, dim=-1)
            
            # Apply attention
            enhanced = torch.matmul(attn_weights, V)
            
            # Apply cross-modal projection
            enhanced = self.cross_modal_proj[proj_key](enhanced)
            
            # Residual connection and layer norm
            output = self.layer_norm(local_features + self.dropout(enhanced))
            
            return output

    class AdaptiveFusionModule(nn.Module):
        def __init__(self, d_model, n_modalities=3, ltp_threshold=0.7, ltd_threshold=0.3):
            super().__init__()
            self.d_model = d_model
            self.n_modalities = n_modalities
            self.ltp_threshold = ltp_threshold
            self.ltd_threshold = ltd_threshold
            
            # Modality importance networks
            self.importance_nets = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.ReLU(),
                    nn.Linear(d_model // 2, 1)
                ) for _ in range(n_modalities)
            ])
            
            # Fusion projection
            self.fusion_proj = nn.Linear(d_model * n_modalities, d_model)
            
            # Adaptive parameters
            self.register_buffer('adaptation_rate', torch.tensor(0.1))
            self.register_buffer('modality_weights', torch.ones(n_modalities) / n_modalities)
        
        def forward(self, modality_features, performance_metric=None):
            """
            modality_features: list of tensors, each (batch_size, d_model)
            performance_metric: scalar indicating current performance (0-1)
            """
            batch_size = modality_features[0].size(0)
            
            # Compute importance scores
            importance_scores = []
            for i, (features, net) in enumerate(zip(modality_features, self.importance_nets)):
                score = net(features)
                importance_scores.append(score)
            
            importance_scores = torch.cat(importance_scores, dim=1)  # (batch_size, n_modalities)
            importance_weights = F.softmax(importance_scores, dim=1)
            
            # Apply LTP/LTD mechanism during training
            if self.training and performance_metric is not None:
                with torch.no_grad():
                    if performance_metric > self.ltp_threshold:
                        # Long-term potentiation - strengthen current weights
                        self.modality_weights *= (1 + self.adaptation_rate)
                    elif performance_metric < self.ltd_threshold:
                        # Long-term depression - weaken current weights
                        self.modality_weights *= (1 - self.adaptation_rate)
                    
                    # Normalize weights
                    self.modality_weights = self.modality_weights / self.modality_weights.sum()
            
            # Apply adaptive weights
            importance_weights = importance_weights * self.modality_weights.unsqueeze(0)
            importance_weights = importance_weights / importance_weights.sum(dim=1, keepdim=True)
            
            # Weighted fusion
            fused_features = torch.zeros(batch_size, self.d_model).to(modality_features[0].device)
            for i, features in enumerate(modality_features):
                fused_features += importance_weights[:, i:i+1] * features
            
            return fused_features, importance_weights

class ContrastiveLearningModule(nn.Module):
    """对比学习模块，用于减少模态间的信息冗余"""
    def __init__(self, d_l, d_a, d_v, temperature=0.07):
        super(ContrastiveLearningModule, self).__init__()
        self.temperature = temperature
        
        # 投影头，将特征映射到对比学习空间
        self.proj_l = nn.Sequential(
            nn.Linear(d_l, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        self.proj_a = nn.Sequential(
            nn.Linear(d_a, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        self.proj_v = nn.Sequential(
            nn.Linear(d_v, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
    def forward(self, h_l, h_a, h_v):
        # 投影到对比学习空间
        z_l = F.normalize(self.proj_l(h_l), dim=1)
        z_a = F.normalize(self.proj_a(h_a), dim=1)
        z_v = F.normalize(self.proj_v(h_v), dim=1)
        
        # 计算互信息损失（希望最小化）
        mi_la = self.compute_mutual_information(z_l, z_a)
        mi_lv = self.compute_mutual_information(z_l, z_v)
        mi_av = self.compute_mutual_information(z_a, z_v)
        
        mutual_info_loss = mi_la + mi_lv + mi_av
        
        return nn.Tanh(mutual_info_loss*0.1), (z_l, z_a, z_v)
    
    def compute_mutual_information(self, z1, z2):
        """计算两个特征之间的互信息（使用InfoNCE估计）"""
        batch_size = z1.size(0)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(z1, z2.T) / self.temperature
        
        # 对角线元素是正样本对
        pos_sim = torch.diag(sim_matrix)
        
        # InfoNCE损失
        loss = -torch.mean(pos_sim - torch.logsumexp(sim_matrix, dim=1))
        
        return loss


class OrthogonalProjectionModule(nn.Module):
    """正交投影模块，确保不同模态特征的正交性"""
    def __init__(self, d_l, d_a, d_v):
        super(OrthogonalProjectionModule, self).__init__()
        
        # 学习正交投影矩阵
        self.W_la = nn.Parameter(torch.eye(d_l, d_a))
        self.W_lv = nn.Parameter(torch.eye(d_l, d_v))
        self.W_av = nn.Parameter(torch.eye(d_a, d_v))
        
    def forward(self, h_l, h_a, h_v):
        # 计算正交性损失
        ortho_loss = 0
        
        # L-A正交性
        proj_la = torch.matmul(self.W_la.T, self.W_la)
        ortho_loss += torch.norm(proj_la - torch.eye(proj_la.size(0)).to(proj_la.device), p='fro')
        
        # L-V正交性
        proj_lv = torch.matmul(self.W_lv.T, self.W_lv)
        ortho_loss += torch.norm(proj_lv - torch.eye(proj_lv.size(0)).to(proj_lv.device), p='fro')
        
        # A-V正交性
        proj_av = torch.matmul(self.W_av.T, self.W_av)
        ortho_loss += torch.norm(proj_av - torch.eye(proj_av.size(0)).to(proj_av.device), p='fro')
        
        return ortho_loss


class DisentangledModalityEncoder(nn.Module):
    """解耦的模态编码器，分离共享和独特信息"""
    def __init__(self, input_dim, shared_dim=32, specific_dim=32):
        super(DisentangledModalityEncoder, self).__init__()
        
        # 共享编码器（提取可能在其他模态中也存在的信息）
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, shared_dim)
        )
        
        # 独特编码器（提取该模态独有的信息）
        self.specific_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, specific_dim)
        )
        
        # 判别器（用于对抗训练，确保共享特征不包含模态特定信息）
        self.discriminator = nn.Sequential(
            nn.Linear(shared_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 3个模态
        )
        
    def forward(self, x, modality_idx):
        shared = self.shared_encoder(x)
        specific = self.specific_encoder(x)
        
        # 判别器预测模态类型
        domain_pred = self.discriminator(shared)
        
        return shared, specific, domain_pred


class AdaptiveModalityFusion(nn.Module):
    """自适应模态融合，基于互补性动态调整融合策略"""
    def __init__(self, d_l, d_a, d_v):
        super(AdaptiveModalityFusion, self).__init__()
        
        # 互补性评估网络
        self.complementarity_net = nn.Sequential(
            nn.Linear(d_l + d_a + d_v, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6)  # 输出6个值：3个自身权重 + 3个交互权重
        )
        
        # 融合网络
        self.fusion_nets = nn.ModuleDict({
            'self': nn.Linear(d_l + d_a + d_v, 128),
            'la': nn.Linear(d_l + d_a, 64),
            'lv': nn.Linear(d_l + d_v, 64),
            'av': nn.Linear(d_a + d_v, 64),
            'final': nn.Linear(128 + 64*3, 128)
        })
        
    def forward(self, h_l, h_a, h_v):
        # 评估互补性
        concat_features = torch.cat([h_l, h_a, h_v], dim=1)
        weights = F.softmax(self.complementarity_net(concat_features), dim=1)
        
        # 自身特征融合
        self_fusion = self.fusion_nets['self'](concat_features)
        
        # 成对交互融合
        la_fusion = self.fusion_nets['la'](torch.cat([h_l, h_a], dim=1))
        lv_fusion = self.fusion_nets['lv'](torch.cat([h_l, h_v], dim=1))
        av_fusion = self.fusion_nets['av'](torch.cat([h_a, h_v], dim=1))
        
        # 加权组合
        weighted_self = self_fusion * (weights[:, 0:1] + weights[:, 1:2] + weights[:, 2:3])
        weighted_la = la_fusion * weights[:, 3:4]
        weighted_lv = lv_fusion * weights[:, 4:5]
        weighted_av = av_fusion * weights[:, 5:6]
        
        # 最终融合
        all_features = torch.cat([weighted_self, weighted_la, weighted_lv, weighted_av], dim=1)
        fused = self.fusion_nets['final'](all_features)
        
        return fused, weights








# 损失函数辅助函数
def compute_orthogonality_loss(specific_features, shared_features):
    """计算正交性损失，确保独特特征与共享特征正交"""
    loss = 0
    batch_size = shared_features.size(0)
    
    for specific in specific_features:
        # 确保维度匹配
        if specific.size(-1) != shared_features.size(-1):
            # 使用线性投影将shared_features投影到与specific相同的维度
            if not hasattr(compute_orthogonality_loss, 'projection_layers'):
                compute_orthogonality_loss.projection_layers = {}
            
            key = f"{shared_features.size(-1)}_to_{specific.size(-1)}"
            if key not in compute_orthogonality_loss.projection_layers:
                projection = nn.Linear(shared_features.size(-1), specific.size(-1)).to(specific.device)
                compute_orthogonality_loss.projection_layers[key] = projection
            
            projection_layer = compute_orthogonality_loss.projection_layers[key]
            shared_proj = projection_layer(shared_features)
        else:
            shared_proj = shared_features
            
        # 计算余弦相似度
        cos_sim = F.cosine_similarity(specific, shared_proj, dim=1)
        # 希望余弦相似度接近0（正交）
        loss += torch.mean(cos_sim ** 2)
        
    return loss / len(specific_features)


def compute_diversity_loss(features_list):
    """确保不同特征之间的多样性"""
    n_features = len(features_list)
    if n_features < 2:
        return torch.tensor(0.0).to(features_list[0].device)
        
    loss = 0
    count = 0
    
    for i in range(n_features):
        for j in range(i+1, n_features):
            # 确保特征维度匹配
            feat_i = features_list[i]
            feat_j = features_list[j]
            
            if feat_i.size(-1) != feat_j.size(-1):
                # 使用自适应池化对齐维度
                target_dim = min(feat_i.size(-1), feat_j.size(-1))
                if feat_i.size(-1) > target_dim:
                    feat_i = F.adaptive_avg_pool1d(
                        feat_i.unsqueeze(1), target_dim
                    ).squeeze(1)
                if feat_j.size(-1) > target_dim:
                    feat_j = F.adaptive_avg_pool1d(
                        feat_j.unsqueeze(1), target_dim
                    ).squeeze(1)
            
            # 计算特征间的相似度
            sim = F.cosine_similarity(feat_i, feat_j, dim=1)
            # 惩罚过高的相似度
            loss += torch.mean(F.relu(sim - 0.5))
            count += 1
            
    return loss / max(count, 1)

class EnhancedContrastiveLossCalculator:
    def __init__(self, hyp_params):
        self.dataset = hyp_params.dataset
        self.main_criterion = getattr(nn, hyp_params.criterion)()
        
        # 损失权重
        self.aux_weight = getattr(hyp_params, 'aux_weight', 0.2)
        self.mutual_info_weight = getattr(hyp_params, 'mutual_info_weight', 0.3)
        self.orthogonality_weight = getattr(hyp_params, 'orthogonality_weight', 0.1)
        self.reconstruction_weight = getattr(hyp_params, 'reconstruction_weight', 0.1)
        self.domain_adversarial_weight = getattr(hyp_params, 'domain_adversarial_weight', 0.1)
        
    def calculate_enhanced_loss(self, model_output, targets, epoch=0, max_epochs=100):
        """计算增强的损失，包括对比学习和互信息最小化"""
        
        # 1. 主任务损失
        main_output = model_output['main_output']
        main_loss = self.main_criterion(main_output, targets)
        
        # 2. 辅助任务损失
        aux_output = model_output['aux_output']
        aux_loss = self.main_criterion(aux_output, targets)
        
        # 3. 互信息损失（最小化）
        mutual_info_loss = model_output['mutual_info_loss']
        
        # 4. 正交性损失
        orthogonality_loss = model_output['orthogonality_loss']
        
        # 5. 重建损失
        reconstruction_loss = model_output['reconstruction_loss']
        
        # 6. 域对抗损失（使共享特征不包含模态特定信息）
        domain_l, domain_a, domain_v = model_output['domain_predictions']
        domain_labels = torch.cat([
            torch.zeros(domain_l.size(0)),
            torch.ones(domain_a.size(0)),
            torch.ones(domain_v.size(0)) * 2
        ]).long().to(domain_l.device)
        
        domain_preds = torch.cat([domain_l, domain_a, domain_v], dim=0)
        domain_loss = F.cross_entropy(domain_preds, domain_labels)
        
        # 动态权重调整
        progress = epoch / max_epochs
        
        # 早期更注重互信息最小化，后期更注重主任务
        dynamic_mi_weight = self.mutual_info_weight * (1 - progress * 0.5)
        dynamic_main_weight = 1.0 + progress * 0.5
        
        # 总损失
        total_loss = (
            dynamic_main_weight * main_loss +
            self.aux_weight * aux_loss +
            dynamic_mi_weight * mutual_info_loss +
            self.orthogonality_weight * orthogonality_loss +
            self.reconstruction_weight * reconstruction_loss -
            self.domain_adversarial_weight * domain_loss  # 负号因为是对抗训练
        )
        
        return {
            'total_loss': total_loss,
            'main_loss': main_loss,
            'aux_loss': aux_loss,
            'mutual_info_loss': mutual_info_loss,
            'orthogonality_loss': orthogonality_loss,
            'reconstruction_loss': reconstruction_loss,
            'domain_loss': domain_loss
        }
        
class CrossModalAttentionLayer(nn.Module):
    def __init__(self, k, x_channels, y_size, spatial=True):
        super(CrossModalAttentionLayer, self).__init__()
        self.k = k
        self.spatial = spatial

        if spatial:
            self.channel_affine = nn.Linear(x_channels, k)

        self.y_affine = nn.Linear(y_size, k, bias=False)
        self.attn_weight_affine = nn.Linear(k, 1)

    def forward(self, x, y):
        # x -> [(bs, S , dim)], len(x) = bs
        # y -> (bs, D)
        bs = y.size(0)
        y_k = self.y_affine(y) # (bs, k)
        all_spatial_attn_weights_softmax = []

        for i in range(bs):
            if self.spatial:
                x_k = self.channel_affine(x[i]) # (S, d)
                x_k += y_k[i]
                x_k = torch.tanh(x_k)
                all_spatial_attn_weights_softmax.append(F.softmax(x_k, dim=-1))

        return torch.cat(all_spatial_attn_weights_softmax, dim=0)
    
class TemporalContextModule(nn.Module):
    """时间上下文建模模块"""
    def __init__(self, feature_dim, context_window=5):
        super(TemporalContextModule, self).__init__()
        self.context_window = context_window
        self.feature_dim = feature_dim
        
        # 时间卷积
        self.temporal_conv = nn.Conv1d(
            in_channels=feature_dim,
            out_channels=feature_dim,
            kernel_size=context_window,
            padding=context_window//2
        )
        
        # 时间注意力
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=0.1
        )
        
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        # x: [seq_len, batch, feature_dim]
        batch_size = x.size(1)
        
        # 时间卷积
        x_conv = self.temporal_conv(x.permute(1, 2, 0))  # [batch, feature_dim, seq_len]
        x_conv = x_conv.permute(2, 0, 1)  # [seq_len, batch, feature_dim]
        
        # 时间注意力
        x_attn, _ = self.temporal_attention(x, x, x)
        
        # 残差连接
        x_enhanced = self.layer_norm(x + x_conv + x_attn)
        
        return x_enhanced
    
class ModalitySpecificityModule(nn.Module):
    """提取并保留每个模态的独特信息"""
    def __init__(self, d_l, d_a, d_v, hidden_dim=64):
        super(ModalitySpecificityModule, self).__init__()
        
        self.d_l = d_l
        self.d_a = d_a
        self.d_v = d_v
        self.hidden_dim = hidden_dim
        
        # 独特性提取网络
        self.l_specific = nn.Sequential(
            nn.Linear(d_l, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, d_l),
            nn.Tanh()
        )
        
        self.a_specific = nn.Sequential(
            nn.Linear(d_a, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, d_a),
            nn.Tanh()
        )
        
        self.v_specific = nn.Sequential(
            nn.Linear(d_v, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, d_v),
            nn.Tanh()
        )
        
        # 共享信息提取 - 输出维度设为各模态维度的平均值
        self.shared_dim = (d_l + d_a + d_v) // 3  # 40
        self.shared_projection = nn.Sequential(
            nn.Linear(d_l + d_a + d_v, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, self.shared_dim)  # 输出40维，与模态维度一致
        )
        
        # 门控机制，控制独特信息的保留程度
        self.specificity_gates = nn.ModuleDict({
            'l': nn.Sequential(nn.Linear(d_l * 2, d_l), nn.Sigmoid()),
            'a': nn.Sequential(nn.Linear(d_a * 2, d_a), nn.Sigmoid()),
            'v': nn.Sequential(nn.Linear(d_v * 2, d_v), nn.Sigmoid())
        })
        
    def forward(self, h_l, h_a, h_v):
        # 🔧 确保输入张量的形状正确性
        # 如果输入是1维，则添加batch维度
        if h_l.dim() == 1:
            h_l = h_l.unsqueeze(0)
        if h_a.dim() == 1:
            h_a = h_a.unsqueeze(0)
        if h_v.dim() == 1:
            h_v = h_v.unsqueeze(0)
        
        # 确保所有张量都至少是2维的 [batch_size, feature_dim]
        if h_l.dim() > 2:
            h_l = h_l.view(-1, h_l.size(-1))
        if h_a.dim() > 2:
            h_a = h_a.view(-1, h_a.size(-1))
        if h_v.dim() > 2:
            h_v = h_v.view(-1, h_v.size(-1))
        
        # 提取每个模态的独特表示
        l_specific_raw = self.l_specific(h_l)
        a_specific_raw = self.a_specific(h_a)
        v_specific_raw = self.v_specific(h_v)
        
        # 计算共享表示 - 使用dim=-1确保在最后一个维度上连接
        concat_features = torch.cat([h_l, h_a, h_v], dim=-1)
        shared_repr = self.shared_projection(concat_features)
        
        # 使用门控机制调节独特性 - 使用dim=-1确保在最后一个维度上连接
        l_gate = self.specificity_gates['l'](torch.cat([h_l, l_specific_raw], dim=-1))
        a_gate = self.specificity_gates['a'](torch.cat([h_a, a_specific_raw], dim=-1))
        v_gate = self.specificity_gates['v'](torch.cat([h_v, v_specific_raw], dim=-1))
        
        l_specific = l_specific_raw * l_gate + h_l * (1 - l_gate)
        a_specific = a_specific_raw * a_gate + h_a * (1 - a_gate)
        v_specific = v_specific_raw * v_gate + h_v * (1 - v_gate)
        
        return l_specific, a_specific, v_specific, shared_repr
    
    




class BCPMJRSModel(nn.Module):
    """Brain Computing-inspired Predefined Multimodal Joint Representation Spaces Model"""
    def __init__(self, hyp_params):
        super(BCPMJRSModel, self).__init__()
        
        # Dimensions
        self.orig_d_l = hyp_params.orig_d_l
        self.orig_d_a = hyp_params.orig_d_a
        self.orig_d_v = hyp_params.orig_d_v
        self.d_l = self.d_a = self.d_v = hyp_params.hidden_dim  # 256
        
        # Modality flags
        self.vonly = hyp_params.vonly
        self.aonly = hyp_params.aonly
        self.lonly = hyp_params.lonly
        
        # Hyperparameters
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask
        
        # Output dimension
        output_dim = hyp_params.output_dim
        
        # 1. Modality-specific encoders (projection layers)
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        
        # 2. Global-Local Cross-Modal Attention
        self.global_local_attention = GlobalLocalCrossModalAttention(
            d_model=self.d_l, 
            n_heads=self.num_heads, 
            dropout=self.attn_dropout
        )
        
        # 3. Cross-modal Transformers (similar to MulT)
        if self.lonly:
            self.trans_l_with_a = self.get_network(self_type='la')
            self.trans_l_with_v = self.get_network(self_type='lv')
        if self.aonly:
            self.trans_a_with_l = self.get_network(self_type='al')
            self.trans_a_with_v = self.get_network(self_type='av')
        if self.vonly:
            self.trans_v_with_l = self.get_network(self_type='vl')
            self.trans_v_with_a = self.get_network(self_type='va')
        
        # 4. Self-attention layers
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
        
        # 5. Mutual Information Estimators
        self.mi_estimator_la = MutualInformationEstimator(self.d_l)
        self.mi_estimator_lv = MutualInformationEstimator(self.d_l)
        self.mi_estimator_av = MutualInformationEstimator(self.d_a)
        
        # 6. Adaptive Fusion Module
        self.adaptive_fusion = AdaptiveFusionModule(
            d_model=self.d_l,
            n_modalities=3,
            ltp_threshold=0.7,
            ltd_threshold=0.3
        )
        
        # 7. Output layers
        self.proj1 = nn.Linear(self.d_l, self.d_l)
        self.proj2 = nn.Linear(self.d_l, self.d_l)
        self.out_layer = nn.Linear(self.d_l, output_dim)
        
        # For MI-max computation
        self.mi_max_estimator = MutualInformationEstimator(self.d_l + output_dim)
    
    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2*self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2*self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2*self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
    
    def forward(self, x_l, x_a, x_v, return_mi_loss=False):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        # Apply dropout and transpose
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)
        
        # Project features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        
        # Permute to (seq_len, batch_size, d_model)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        
        # Extract global contexts
        global_l = self.global_local_attention.extract_global_context(proj_x_l)
        global_a = self.global_local_attention.extract_global_context(proj_x_a)
        global_v = self.global_local_attention.extract_global_context(proj_x_v)
        
        # Enhance features with global-local attention
        enhanced_l = self.global_local_attention.enhance_local_features(proj_x_l, global_a, 'la')
        enhanced_l = self.global_local_attention.enhance_local_features(enhanced_l, global_v, 'lv')
        
        enhanced_a = self.global_local_attention.enhance_local_features(proj_x_a, global_l, 'al')
        enhanced_a = self.global_local_attention.enhance_local_features(enhanced_a, global_v, 'av')
        
        enhanced_v = self.global_local_attention.enhance_local_features(proj_x_v, global_l, 'vl')
        enhanced_v = self.global_local_attention.enhance_local_features(enhanced_v, global_a, 'va')
        
        # Cross-modal transformers
        if self.lonly:
            h_l_with_as = self.trans_l_with_a(enhanced_l, enhanced_a, enhanced_a)
            h_l_with_vs = self.trans_l_with_v(enhanced_l, enhanced_v, enhanced_v)
            h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
            h_ls = self.trans_l_mem(h_ls)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = h_ls[-1]
        
        if self.aonly:
            h_a_with_ls = self.trans_a_with_l(enhanced_a, enhanced_l, enhanced_l)
            h_a_with_vs = self.trans_a_with_v(enhanced_a, enhanced_v, enhanced_v)
            h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
            h_as = self.trans_a_mem(h_as)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = h_as[-1]
        
        if self.vonly:
            h_v_with_ls = self.trans_v_with_l(enhanced_v, enhanced_l, enhanced_l)
            h_v_with_as = self.trans_v_with_a(enhanced_v, enhanced_a, enhanced_a)
            h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
            h_vs = self.trans_v_mem(h_vs)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = h_vs[-1]
        
        # Get final representations for each modality
        if self.partial_mode == 1:
            if self.lonly:
                modality_features = [last_h_l[:, :self.d_l], last_h_l[:, self.d_l:], torch.zeros_like(last_h_l[:, :self.d_l])]
            elif self.aonly:
                modality_features = [torch.zeros_like(last_h_a[:, :self.d_a]), last_h_a[:, :self.d_a], last_h_a[:, self.d_a:]]
            else:  # vonly
                modality_features = [last_h_v[:, :self.d_v], torch.zeros_like(last_h_v[:, :self.d_v]), last_h_v[:, self.d_v:]]
        else:
            modality_features = [last_h_l[:, :self.d_l], last_h_a[:, :self.d_a], last_h_v[:, :self.d_v]]
        
        # Adaptive fusion
        fused_features, importance_weights = self.adaptive_fusion(modality_features)
        
        # Final projection
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(fused_features)), p=self.out_dropout, training=self.training))
        last_hs_proj += fused_features  # Residual connection
        
        # Output
        output = self.out_layer(last_hs_proj)
        
        # Compute MI losses if requested
        mi_loss_min = torch.tensor(0.0).to(x_l.device)
        mi_loss_max = torch.tensor(0.0).to(x_l.device)
        
        if return_mi_loss and self.training:
            try:
                # 确保modality_features都是2D张量
                mod_features = []
                for feat in modality_features:
                    if feat.dim() > 2:
                        feat = feat.view(feat.size(0), -1)
                    mod_features.append(feat)
                
                # MI-min: minimize MI between different modalities
                mi_la = self.mi_estimator_la(mod_features[0], mod_features[1])
                mi_lv = self.mi_estimator_lv(mod_features[0], mod_features[2])
                mi_av = self.mi_estimator_av(mod_features[1], mod_features[2])
                
                mi_loss_min = (mi_la + mi_lv + mi_av) / 3.0
                
                # 确保MI损失在合理范围内
                mi_loss_min = torch.clamp(mi_loss_min, 0, 5)
                
                # MI-max: 简化为零或者使用任务相关的损失
                # 这里我们暂时设为0，避免复杂的MI最大化计算
                mi_loss_max = torch.tensor(0.0).to(x_l.device)
                
            except Exception as e:
                print(f"Warning: MI loss calculation failed: {e}")
                mi_loss_min = torch.tensor(0.0).to(x_l.device)
                mi_loss_max = torch.tensor(0.0).to(x_l.device)
        
        if return_mi_loss:
            return output, fused_features, mi_loss_min, mi_loss_max
        else:
            return output, fused_features
    
    @property
    def partial_mode(self):
        return self.lonly + self.aonly + self.vonly