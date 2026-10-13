import os
import json
import logging
import time
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from ..models import DistillationTrainer


class KnowledgeDistillationTrainer:
    """
    知识蒸馏训练器
    实现完整的训练流程，包括渐进式解冻、混合精度训练、梯度累积等
    """
    def __init__(self, config, distillation_trainer, dataloaders):
        self.config = config
        self.trainer = distillation_trainer
        self.dataloaders = dataloaders
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 训练配置
        self.num_epochs = config["training"]["num_train_epochs"]
        self.gradient_accumulation_steps = config["training"]["gradient_accumulation_steps"]
        self.max_grad_norm = config["training"]["max_grad_norm"]
        
        # 创建优化器
        self.optimizer = self._create_optimizer()
        
        # 创建学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 确定混合精度类型
        mixed_precision = "no"
        if config["training"]["fp16"]:
            mixed_precision = "fp16"
        elif config["training"]["bf16"]:
            mixed_precision = "bf16"
        
        # Accelerator实例
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
        )
        
        # 训练状态
        self.global_step = 0
        self.best_metric = 0.0
        self.metrics_history = []
        
        # 日志和检查点目录
        self.output_dir = config["output_dir"]
        self.log_dir = config["log_dir"]
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 日志设置
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, "training.log")),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)
        
    def _create_optimizer(self):
        """创建优化器"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.trainer.student.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.config["training"]["weight_decay"],
            },
            {
                "params": [p for n, p in self.trainer.student.named_parameters()
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        
        return AdamW(
            optimizer_grouped_parameters, 
            lr=float(self.config["training"]["learning_rate"])
        )
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        train_dataloader = self.dataloaders.get("train_student")
        if train_dataloader is None:
            return None
            
        num_training_steps = len(train_dataloader) // self.gradient_accumulation_steps * self.num_epochs
        num_warmup_steps = int(num_training_steps * self.config["training"]["warmup_ratio"])
        
        return get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    
    def train(self):
        """执行完整的训练循环"""
        self.logger.info(f"***** 开始训练 *****")
        self.logger.info(f"设备: {self.device}")
        self.logger.info(f"训练轮数: {self.num_epochs}")
        self.logger.info(f"梯度累积步数: {self.gradient_accumulation_steps}")
        self.logger.info(f"混合精度: {self.accelerator.mixed_precision}")
        
        # 准备训练
        student_model = self.trainer.student
        optimizer = self.optimizer
        scheduler = self.scheduler
        
        # 如果使用渐进式解冻策略
        use_progressive_freezing = self.config["training"]["progressive_freezing"]
        freezing_schedule = self.config["training"]["freezing_layers_schedule"] if use_progressive_freezing else []
        
        # 将模型、优化器等移到加速器中
        student_model, optimizer, scheduler, train_dataloader, valid_dataloader = self.accelerator.prepare(
            student_model,
            optimizer, 
            scheduler,
            self.dataloaders["train_student"],
            self.dataloaders["valid_student"]
        )
        
        # 冻结所有参数作为初始状态（如果使用渐进式解冻）
        if use_progressive_freezing:
            self.logger.info("初始冻结所有参数（使用渐进式解冻策略）")
            for param in student_model.parameters():
                param.requires_grad = False
            
        # 训练循环
        for epoch in range(self.num_epochs):
            self.logger.info(f"开始第 {epoch + 1}/{self.num_epochs} 轮训练")
            
            # 如果使用渐进式解冻，根据计划解冻层
            if use_progressive_freezing and epoch < len(freezing_schedule):
                layers_to_unfreeze = freezing_schedule[epoch]
                self.logger.info(f"解冻层: {', '.join(layers_to_unfreeze)}")
                student_model.unfreeze_layers(layers_to_unfreeze)
            
            # 训练一个epoch
            self._train_epoch(epoch, train_dataloader)
            
            # 评估
            metrics = self._evaluate(valid_dataloader)
            self.metrics_history.append(metrics)
            
            # 保存进度
            accuracy = metrics.get("accuracy", 0.0)
            if accuracy > self.best_metric:
                self.best_metric = accuracy
                self._save_checkpoint(f"best_model")
                self.logger.info(f"保存最佳模型，准确率: {accuracy:.4f}")
            
            # 每个epoch保存一次
            self._save_checkpoint(f"epoch_{epoch + 1}")
            
        self.logger.info("***** 训练完成 *****")
        return self.metrics_history
    
    def _train_epoch(self, epoch, train_dataloader):
        """训练一个epoch"""
        student_model = self.trainer.student
        student_model.train()
        
        # 进度条
        epoch_iterator = tqdm(
            train_dataloader, 
            desc=f"训练 (Epoch {epoch + 1})",
            disable=not self.accelerator.is_local_main_process,
        )
        
        # 训练数据
        total_loss = 0
        num_batches = 0
        start_time = time.time()
        
        for step, batch in enumerate(epoch_iterator):
            # 准备数据 - 当使用accelerate时，batch已经是在正确设备上的张量
            input_ids = batch["input_ids"] if "input_ids" in batch else None
            attention_mask = batch["attention_mask"] if "attention_mask" in batch else None
            labels = batch["labels"] if "labels" in batch else None
            
            # 如果有教师模型，获取教师输出
            teacher_outputs = None
            if "train_teacher" in self.dataloaders and hasattr(self.trainer, "teacher_model"):
                teacher_outputs = self.trainer.get_teacher_outputs(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            
            # 获取学生模型输出
            student_outputs = self.trainer.get_student_outputs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            # 计算损失
            loss_dict = self.trainer.compute_loss(teacher_outputs, student_outputs, labels)
            loss = loss_dict["total_loss"]
            
            # 使用加速器进行反向传播和梯度缩放
            self.accelerator.backward(loss / self.gradient_accumulation_steps)
            
            # 梯度累积
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # 梯度裁剪
                if self.max_grad_norm > 0:
                    self.accelerator.clip_grad_norm_(student_model.parameters(), self.max_grad_norm)
                
                # 参数更新
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # 更新进度条
                epoch_iterator.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "step": self.global_step,
                })
                
                # 记录日志
                if self.global_step % self.config["training"]["logging_steps"] == 0:
                    self.logger.info(
                        f"Epoch: {epoch + 1}, Step: {self.global_step}, "
                        f"Loss: {loss.item():.4f}, "
                        f"CE Loss: {loss_dict['ce_loss']:.4f}, "
                        f"Distillation Loss: {loss_dict['distillation_loss']:.4f}, "
                        f"Multi-task Loss: {loss_dict['multi_task_loss']:.4f}"
                    )
                
                # 评估和保存
                if (
                    self.config["training"]["eval_steps"] > 0 and
                    self.global_step % self.config["training"]["eval_steps"] == 0
                ):
                    metrics = self._evaluate(valid_dataloader)  # 使用加速器prepare后的valid_dataloader
                    student_model.train()  # 恢复训练模式
                    
                    # 如果达到新的最佳性能，保存模型
                    accuracy = metrics.get("accuracy", 0.0)
                    if accuracy > self.best_metric:
                        self.best_metric = accuracy
                        self._save_checkpoint(f"best_model")
                        self.logger.info(f"保存最佳模型，准确率: {accuracy:.4f}")
                
                # 定期保存
                if (
                    self.config["training"]["save_steps"] > 0 and
                    self.global_step % self.config["training"]["save_steps"] == 0
                ):
                    self._save_checkpoint(f"step_{self.global_step}")
            
            total_loss += loss.item()
            num_batches += 1
        
        # 一个epoch的统计
        avg_loss = total_loss / num_batches
        elapsed = time.time() - start_time
        self.logger.info(
            f"Epoch {epoch + 1} 完成 - 平均损失: {avg_loss:.4f}, 耗时: {elapsed:.2f} 秒"
        )
    
    def _evaluate(self, eval_dataloader):
        """在验证集上评估模型"""
        student_model = self.trainer.student
        student_model.eval()
        
        all_preds = []
        all_labels = []
        total_eval_loss = 0
        
        # 评估进度条
        eval_iterator = tqdm(
            eval_dataloader,
            desc="评估",
            disable=not self.accelerator.is_local_main_process,
        )
        
        with torch.no_grad():
            for batch in eval_iterator:
                # 准备数据 - 当使用accelerate时，batch已经是在正确设备上的张量
                input_ids = batch["input_ids"] if "input_ids" in batch else None
                attention_mask = batch["attention_mask"] if "attention_mask" in batch else None
                labels = batch["labels"] if "labels" in batch else None
                
                # 模型前向传播
                outputs = student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                
                loss = outputs.loss if hasattr(outputs, "loss") else None
                if loss is not None:
                    total_eval_loss += loss.item()
                
                # 收集预测结果和标签
                if hasattr(outputs, "logits"):
                    preds = torch.argmax(outputs.logits, dim=-1)
                    all_preds.append(preds)
                    all_labels.append(labels)
        
        # 如果有预测结果，计算评估指标
        metrics = {}
        if all_preds and all_labels:
            # 合并所有批次的预测和标签
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            
            # 计算准确率
            correct = (all_preds == all_labels).sum().item()
            total = all_labels.numel()
            accuracy = correct / total if total > 0 else 0
            
            # 记录指标
            metrics["accuracy"] = accuracy
            metrics["eval_loss"] = total_eval_loss / len(eval_dataloader)
            
            self.logger.info(f"评估结果 - 准确率: {accuracy:.4f}, 损失: {metrics['eval_loss']:.4f}")
        
        return metrics
    
    def _save_checkpoint(self, checkpoint_name):
        """保存检查点"""
        if not self.accelerator.is_local_main_process:
            return
            
        output_dir = os.path.join(self.output_dir, checkpoint_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存模型
        unwrapped_model = self.accelerator.unwrap_model(self.trainer.student)
        unwrapped_model.student_model.save_pretrained(output_dir)
        
        # 保存tokenizer
        unwrapped_model.student_tokenizer.save_pretrained(output_dir)
        
        # 保存训练状态和配置
        with open(os.path.join(output_dir, "training_state.json"), "w") as f:
            json.dump(
                {
                    "global_step": self.global_step,
                    "best_metric": self.best_metric,
                    "epoch": self.num_epochs,
                },
                f,
            )
        
        self.logger.info(f"保存检查点到 {output_dir}") 
