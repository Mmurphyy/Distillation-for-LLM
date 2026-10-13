import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
)


class EnhancedAttention(nn.Module):
    """
    增强型注意力机制，专为知识推理场景设计
    动态融合题目信息和多步推理过程
    """
    def __init__(self, hidden_size, num_attention_heads=8, dropout_prob=0.1):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = hidden_size
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)
        
        # 知识推理增强
        self.reasoning_gate = nn.Linear(hidden_size, 1)
        self.reasoning_transform = nn.Linear(hidden_size, hidden_size)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
        
    def forward(self, hidden_states, attention_mask=None):
        # 标准自注意力机制
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # 计算注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # 知识推理增强：动态融合
        reasoning_gate = torch.sigmoid(self.reasoning_gate(hidden_states))
        reasoning_hidden = self.reasoning_transform(hidden_states)
        
        # 将推理信息与注意力融合
        enhanced_output = context_layer * reasoning_gate + reasoning_hidden * (1 - reasoning_gate)
        
        output = self.dense(enhanced_output)
        output = self.dropout(output)
        output = self.LayerNorm(output + hidden_states)
        
        return output


class KnowledgeDistillationModel(nn.Module):
    """
    知识蒸馏模型，包含题目理解和知识推理两个层次
    支持从大型教师模型向小型学生模型迁移知识
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 初始化学生模型
        student_model_name = config["model"]["student_model"]
        self.student_config = AutoConfig.from_pretrained(student_model_name)
        self.student_model = AutoModelForCausalLM.from_pretrained(student_model_name)
        self.student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
        
        # 如果使用PEFT/LoRA进行微调
        if config["model"]["use_peft"]:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=config["model"]["lora_r"],
                lora_alpha=config["model"]["lora_alpha"],
                lora_dropout=config["model"]["lora_dropout"],
                target_modules=config["model"]["target_modules"],
            )
            self.student_model = get_peft_model(self.student_model, peft_config)
            
        # 增强注意力模块
        self.enhanced_attention = EnhancedAttention(
            self.student_config.hidden_size,
            num_attention_heads=8,
            dropout_prob=0.1,
        )
        
        # 动态知识路由
        self.knowledge_router = nn.Linear(self.student_config.hidden_size, 4)  # 知识领域数量
        
        # 多任务学习头
        self.answer_head = nn.Linear(self.student_config.hidden_size, self.student_config.vocab_size)
        self.reasoning_head = nn.Linear(self.student_config.hidden_size, self.student_config.vocab_size)
        
    def freeze_layers(self, layers_to_freeze=None):
        """冻结指定层的参数"""
        if layers_to_freeze is None:
            return
        
        # 冻结指定的层
        for name, param in self.student_model.named_parameters():
            freeze_layer = False
            for layer_name in layers_to_freeze:
                if layer_name in name:
                    freeze_layer = True
                    break
            
            if freeze_layer:
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    def unfreeze_layers(self, layers_to_unfreeze=None):
        """解冻指定层的参数"""
        if layers_to_unfreeze is None:
            return
        
        # 解冻指定的层
        for name, param in self.student_model.named_parameters():
            unfreeze_layer = False
            for layer_name in layers_to_unfreeze:
                if layer_name in name:
                    unfreeze_layer = True
                    break
            
            if unfreeze_layer:
                param.requires_grad = True
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        labels=None,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=True,
    ):
        # 学生模型前向传播
        outputs = self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        # 从学生模型获取隐藏状态
        if output_hidden_states:
            # 获取编码器最后一层的隐藏状态
            encoder_hidden_states = outputs.encoder_last_hidden_state  # [batch_size, seq_len, hidden_size]
            
            # 应用增强注意力机制
            enhanced_hidden_states = self.enhanced_attention(
                encoder_hidden_states,
                attention_mask=attention_mask.unsqueeze(1).unsqueeze(2),
            )
            
            # 知识路由：计算不同知识领域的权重
            # 取序列中第一个token的表示作为整体表示
            sequence_output = enhanced_hidden_states[:, 0, :]
            routing_logits = self.knowledge_router(sequence_output)
            routing_weights = F.softmax(routing_logits, dim=-1)
            
            # 多任务输出
            answer_logits = self.answer_head(sequence_output)
            reasoning_logits = self.reasoning_head(sequence_output)
            
            # 将额外输出添加到结果中
            outputs.enhanced_hidden_states = enhanced_hidden_states
            outputs.routing_weights = routing_weights
            outputs.answer_logits = answer_logits
            outputs.reasoning_logits = reasoning_logits
        
        return outputs
    
    def generate_answer(
        self,
        input_ids,
        attention_mask=None,
        max_new_tokens=50,
        num_beams=4,
        **kwargs
    ):
        """生成答案的方法"""
        # 直接使用学生模型的生成功能
        return self.student_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            **kwargs
        )


class DistillationTrainer:
    """
    知识蒸馏训练器，负责从教师模型向学生模型传递知识
    """
    def __init__(self, config):
        self.config = config
        
        # 初始化教师模型（无需梯度）
        teacher_model_name = config["model"]["teacher_model"]
        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            teacher_model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16 if config["training"]["bf16"] else torch.float16,
        )
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
        
        # 冻结教师模型参数
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        # 初始化学生模型
        self.student = KnowledgeDistillationModel(config)
        
        # 温度参数（用于软目标蒸馏）
        self.temperature = 2.0
        
        # 损失权重
        self.distillation_weight = 0.5
        self.ce_weight = 0.5
        self.multi_task_weights = config["training"]["multi_task_weights"]
    
    def compute_distillation_loss(self, teacher_logits, student_logits):
        """
        计算蒸馏损失（KL散度）
        """
        # 应用温度
        teacher_logits_scaled = teacher_logits / self.temperature
        student_logits_scaled = student_logits / self.temperature
        
        # 计算KL散度
        distillation_loss = F.kl_div(
            F.log_softmax(student_logits_scaled, dim=-1),
            F.softmax(teacher_logits_scaled, dim=-1),
            reduction="batchmean",
        ) * (self.temperature ** 2)
        
        return distillation_loss
    
    def compute_loss(self, teacher_outputs, student_outputs, labels):
        """
        计算总损失
        """
        # 标准交叉熵损失
        ce_loss = student_outputs.loss if hasattr(student_outputs, "loss") else 0.0
        
        # 如果教师输出有logits，计算蒸馏损失
        distillation_loss = 0.0
        if hasattr(teacher_outputs, "logits") and hasattr(student_outputs, "logits"):
            distillation_loss = self.compute_distillation_loss(
                teacher_outputs.logits,
                student_outputs.logits,
            )
        
        # 多任务损失（如果有）
        multi_task_loss = 0.0
        if hasattr(student_outputs, "answer_logits") and hasattr(student_outputs, "reasoning_logits"):
            answer_loss = F.cross_entropy(student_outputs.answer_logits, labels.view(-1))
            reasoning_loss = F.cross_entropy(student_outputs.reasoning_logits, labels.view(-1))
            
            multi_task_loss = (
                self.multi_task_weights["answer_prediction"] * answer_loss +
                self.multi_task_weights["reasoning_generation"] * reasoning_loss
            )
        
        # 总损失
        total_loss = (
            self.ce_weight * ce_loss +
            self.distillation_weight * distillation_loss +
            multi_task_loss
        )
        
        return {
            "total_loss": total_loss,
            "ce_loss": ce_loss,
            "distillation_loss": distillation_loss,
            "multi_task_loss": multi_task_loss,
        }
    
    def get_teacher_outputs(self, input_ids, attention_mask=None):
        """
        获取教师模型的输出
        """
        with torch.no_grad():
            # 直接使用输入参数传递给教师模型，不进行任何转换
            # Accelerator已经确保数据在正确的设备上
            return self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
    
    def get_student_outputs(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        output_hidden_states=True,
    ):
        """
        获取学生模型的输出
        """
        # 直接使用输入参数传递给学生模型，不进行任何转换
        # Accelerator已经确保数据在正确的设备上
        return self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=output_hidden_states,
        )
    
    def save_model(self, output_dir):
        """
        保存模型
        """
        os.makedirs(output_dir, exist_ok=True)
        self.student.student_model.save_pretrained(output_dir)
        self.student.student_tokenizer.save_pretrained(output_dir) 
