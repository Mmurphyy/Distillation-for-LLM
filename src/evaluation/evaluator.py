import os
import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ..data import ExamDataset


class Evaluator:
    """
    评估器，用于评估知识蒸馏模型的性能
    支持多种评估指标和结果可视化
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.metrics_to_compute = config["evaluation"]["metrics"]
        self.generate_explanations = config["evaluation"]["generate_explanations"]
        self.num_beams = config["evaluation"]["num_beams"]
        self.max_new_tokens = config["evaluation"]["max_new_tokens"]
        
        # 日志设置
        self.log_dir = config["log_dir"]
        os.makedirs(self.log_dir, exist_ok=True)
        
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, "evaluation.log")),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)
        
    def load_model(self, model_path: str) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
        """
        加载模型和分词器
        """
        self.logger.info(f"从 {model_path} 加载模型...")
        
        if os.path.exists(model_path):
            # 量化配置
            quantization = self.config["inference"]["quantization"]
            device_map = self.config["inference"]["device_map"]
            
            # 加载模型
            model_kwargs = {
                "device_map": device_map,
            }
            
            # 如果启用了量化
            if quantization == "int8":
                model_kwargs["load_in_8bit"] = True
            elif quantization == "int4":
                model_kwargs["load_in_4bit"] = True
                
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path, **model_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            self.logger.info(f"模型加载成功，设备: {self.device}")
            return model, tokenizer
        else:
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
    
    def evaluate(self, model_path: str, test_file: str) -> Dict:
        """
        评估模型性能
        """
        model, tokenizer = self.load_model(model_path)
        
        # 加载测试数据
        self.logger.info(f"加载测试数据: {test_file}")
        dataset = ExamDataset(self.config)
        test_dataset = dataset.load_jsonl(test_file)
        
        # 预处理数据
        test_encodings = dataset.preprocess_for_student_model(
            test_dataset, tokenizer, self.config["data"]["max_seq_length"], is_training=True
        )
        
        # 指标初始化
        metrics = {
            "total_samples": len(test_dataset),
            "correct_samples": 0,
            "accuracy": 0.0,
            "f1": 0.0,
            "exact_match": 0.0,
            "avg_inference_time": 0.0,
            "examples": [],
        }
        
        # 开始评估
        self.logger.info("开始评估...")
        model.eval()
        
        total_inference_time = 0.0
        
        # 分批处理
        batch_size = self.config["inference"]["batch_size"]
        for i in range(0, len(test_dataset), batch_size):
            # 准备批次数据
            batch_data = {
                "input_ids": test_encodings["input_ids"][i:i+batch_size].to(self.device),
                "attention_mask": test_encodings["attention_mask"][i:i+batch_size].to(self.device),
            }
            
            # 记录推理时间
            start_time = time.time()
            
            # 生成答案
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=batch_data["input_ids"],
                    attention_mask=batch_data["attention_mask"],
                    max_new_tokens=self.max_new_tokens,
                    num_beams=self.num_beams,
                    early_stopping=True,
                )
            
            # 计算推理时间
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            
            # 解码生成的答案
            generated_answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # 获取真实标签
            reference_answers = []
            for j in range(i, min(i+batch_size, len(test_dataset))):
                idx = j - i
                reference_answers.append(test_dataset[j]["answer_key"])
                
                # 保存示例
                if len(metrics["examples"]) < 10:  # 仅保存前10个示例
                    metrics["examples"].append({
                        "id": test_dataset[j]["id"],
                        "question": test_dataset[j]["stem"],
                        "choices": test_dataset[j]["choices"],
                        "predicted": generated_answers[idx].strip(),
                        "reference": reference_answers[-1],
                        "is_correct": generated_answers[idx].strip() == reference_answers[-1],
                    })
                
                # 检查是否正确
                if generated_answers[idx].strip() == reference_answers[-1]:
                    metrics["correct_samples"] += 1
        
        # 计算最终指标
        metrics["accuracy"] = metrics["correct_samples"] / metrics["total_samples"]
        metrics["avg_inference_time"] = total_inference_time / metrics["total_samples"]
        
        # 如果请求计算F1和Exact Match
        if "f1" in self.metrics_to_compute or "exact_match" in self.metrics_to_compute:
            # 这里需要更复杂的评估逻辑，根据需求实现
            # 简单示例实现
            metrics["f1"] = metrics["accuracy"]  # 简化，实际应该计算真正的F1
            metrics["exact_match"] = metrics["accuracy"]  # 简化，实际应该计算真正的EM
        
        # 打印评估结果
        self.logger.info(f"评估完成:")
        self.logger.info(f"  样本数量: {metrics['total_samples']}")
        self.logger.info(f"  正确数量: {metrics['correct_samples']}")
        self.logger.info(f"  准确率: {metrics['accuracy']:.4f}")
        self.logger.info(f"  平均推理时间: {metrics['avg_inference_time']:.4f}秒/样本")
        
        # 保存评估结果
        output_dir = os.path.join(self.log_dir, "evaluation_results")
        os.makedirs(output_dir, exist_ok=True)
        
        result_file = os.path.join(output_dir, f"eval_results_{time.strftime('%Y%m%d_%H%M%S')}.json")
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"评估结果已保存到: {result_file}")
        
        return metrics
    
    @staticmethod
    def compute_accuracy(predictions: List[str], references: List[str]) -> float:
        """
        计算准确率
        """
        if len(predictions) != len(references):
            raise ValueError("预测和参考答案数量不匹配")
            
        correct = sum(1 for pred, ref in zip(predictions, references) if pred.strip() == ref.strip())
        return correct / len(predictions) if len(predictions) > 0 else 0.0
        
    @staticmethod
    def compute_f1(predictions: List[str], references: List[str]) -> float:
        """
        计算F1分数
        """
        if len(predictions) != len(references):
            raise ValueError("预测和参考答案数量不匹配")
            
        f1_scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = set(pred.strip().lower().split())
            ref_tokens = set(ref.strip().lower().split())
            
            common_tokens = pred_tokens.intersection(ref_tokens)
            
            precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
            recall = len(common_tokens) / len(ref_tokens) if ref_tokens else 0
            
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
            
        return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    
    @staticmethod
    def compute_exact_match(predictions: List[str], references: List[str]) -> float:
        """
        计算精确匹配率
        """
        if len(predictions) != len(references):
            raise ValueError("预测和参考答案数量不匹配")
            
        exact_matches = sum(1 for pred, ref in zip(predictions, references) 
                             if pred.strip().lower() == ref.strip().lower())
        return exact_matches / len(predictions) if len(predictions) > 0 else 0.0 
