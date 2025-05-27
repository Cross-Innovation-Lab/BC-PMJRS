import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MutualInformationEstimator(nn.Module):
    """InfoNCE-based mutual information estimator"""
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
        
        # Positive pairs
        pos_pairs = torch.cat([features1, features2], dim=1)
        pos_scores = self.critic(pos_pairs)
        
        # Negative pairs - shuffle features2
        neg_indices = torch.randperm(batch_size).to(features1.device)
        neg_features2 = features2[neg_indices]
        neg_pairs = torch.cat([features1, neg_features2], dim=1)
        neg_scores = self.critic(neg_pairs)
        
        # InfoNCE loss
        scores = torch.cat([pos_scores, neg_scores], dim=0)
        labels = torch.cat([torch.ones(batch_size), torch.zeros(batch_size)]).to(features1.device)
        
        mi_estimate = F.binary_cross_entropy_with_logits(scores.squeeze(), labels)
        return -mi_estimate  # Negative because we want to maximize MI


class GlobalLocalCrossModalAttention(nn.Module):
    """Global-Local Cross-Modal Interaction Mechanism"""
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
    """Adaptive Fusion with LTP/LTD mechanism"""
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


class EnhanceSuppressOptimizer:
    """Adaptive optimizer with LTP/LTD mechanism based on validation loss"""
    def __init__(self, base_optimizer, initial_lr, reset_threshold=10):
        self.base_optimizer = base_optimizer
        self.initial_lr = initial_lr
        self.reset_threshold = reset_threshold
        
        # Track validation loss
        self.prev_valid_loss = None
        self.current_epoch = 0
        
        # Track enhancement/suppression counts
        self.enhance_count = 0
        self.suppress_count = 0
        
        # Store initial learning rates for reset
        self.initial_param_lrs = []
        for param_group in self.base_optimizer.param_groups:
            self.initial_param_lrs.append(param_group['lr'])
    
    def step(self, closure=None):
        """Standard optimizer step"""
        return self.base_optimizer.step(closure)
    
    def zero_grad(self):
        """Zero gradients"""
        self.base_optimizer.zero_grad()
    
    def update_lr_on_validation(self, current_valid_loss, epoch):
        """Update learning rate based on validation loss trend"""
        self.current_epoch = epoch
        
        # Skip epoch 0
        if epoch == 0:
            self.prev_valid_loss = current_valid_loss
            print(f"Epoch {epoch}: Initial valid loss = {current_valid_loss:.4f}")
            return
        
        # Calculate epoch factor (epoch // 10, but at least 1)
        epoch_factor = max(1, epoch // 10)
        
        if self.prev_valid_loss is not None:
            # Check if validation loss improved (decreased)
            if current_valid_loss < self.prev_valid_loss:
                # LTP: Performance improved - enhance learning rate
                enhance_factor = (1 + epoch_factor) / epoch_factor
                self.enhance_count += 1
                
                for param_group in self.base_optimizer.param_groups:
                    old_lr = param_group['lr']
                    param_group['lr'] *= enhance_factor
                    new_lr = param_group['lr']
                
                print(f"Epoch {epoch}: Valid loss improved ({self.prev_valid_loss:.4f} -> {current_valid_loss:.4f})")
                print(f"  LTP activated: lr {old_lr:.6f} -> {new_lr:.6f} (factor: {enhance_factor:.3f})")
                print(f"  Enhancement count: {self.enhance_count}")
                
            else:
                # LTD: Performance degraded - suppress learning rate
                suppress_factor = 1 / epoch_factor
                self.suppress_count += 1
                
                for param_group in self.base_optimizer.param_groups:
                    old_lr = param_group['lr']
                    param_group['lr'] *= suppress_factor
                    new_lr = param_group['lr']
                
                print(f"Epoch {epoch}: Valid loss worsened ({self.prev_valid_loss:.4f} -> {current_valid_loss:.4f})")
                print(f"  LTD activated: lr {old_lr:.6f} -> {new_lr:.6f} (factor: {suppress_factor:.3f})")
                print(f"  Suppression count: {self.suppress_count}")
        
        # Check if we need to reset learning rate
        if self.enhance_count >= self.reset_threshold or self.suppress_count >= self.reset_threshold:
            print(f"\nResetting learning rate after {self.enhance_count} enhancements and {self.suppress_count} suppressions")
            self.reset_learning_rate()
            self.enhance_count = 0
            self.suppress_count = 0
        
        # Update previous validation loss
        self.prev_valid_loss = current_valid_loss
    
    def reset_learning_rate(self):
        """Reset learning rate to initial values"""
        for param_group, initial_lr in zip(self.base_optimizer.param_groups, self.initial_param_lrs):
            param_group['lr'] = initial_lr
        print(f"Learning rate reset to initial value: {self.initial_param_lrs[0]:.6f}")
    
    def get_current_lr(self):
        """Get current learning rate"""
        return self.base_optimizer.param_groups[0]['lr']
    
    @property
    def param_groups(self):
        """Access to parameter groups for scheduler compatibility"""
        return self.base_optimizer.param_groups
    
    @property
    def state(self):
        """Access to optimizer state"""
        return self.base_optimizer.state
    
    def state_dict(self):
        """Get state dict for checkpointing"""
        return {
            'base_optimizer': self.base_optimizer.state_dict(),
            'initial_lr': self.initial_lr,
            'prev_valid_loss': self.prev_valid_loss,
            'current_epoch': self.current_epoch,
            'enhance_count': self.enhance_count,
            'suppress_count': self.suppress_count,
            'initial_param_lrs': self.initial_param_lrs
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint"""
        self.base_optimizer.load_state_dict(state_dict['base_optimizer'])
        self.initial_lr = state_dict['initial_lr']
        self.prev_valid_loss = state_dict['prev_valid_loss']
        self.current_epoch = state_dict['current_epoch']
        self.enhance_count = state_dict['enhance_count']
        self.suppress_count = state_dict['suppress_count']
        self.initial_param_lrs = state_dict['initial_param_lrs']