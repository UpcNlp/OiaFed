"""
DDDR联邦训练器

基于FedCL框架实现DDDR的完整联邦训练流程，负责全局联邦逻辑。
"""

import os
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, List

from ...fl.abstract_trainer import AbstractFederationTrainer
from ...methods.learners.dddr import DDDRLearner
from ...methods.aggregators import FedAvgAggregator
from ...registry import registry
from ...api.decorators import trainer


@trainer("dddr", description="DDDR联邦训练器")
class DDDRFederationTrainer(AbstractFederationTrainer):
    """DDDR联邦训练器 - 负责全局联邦训练流程"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # DDDR特定配置
        self.num_tasks = config.get("num_tasks", 5)
        self.classes_per_task = config.get("classes_per_task", 20)
        self.total_classes = config.get("total_classes", 100)
        
        # 扩散模型配置
        self.ldm_config_path = config.get("ldm_config", "ldm/ldm_dddr.yaml")
        self.ldm_ckpt_path = config.get("ldm_ckpt", "models/ldm/text2img-large/model.ckpt")
        
        # 数据生成配置
        self.pre_size = config.get("pre_size", 200)
        self.cur_size = config.get("cur_size", 50)
        self.n_iter = config.get("n_iter", 5)
        
        # 训练配置
        self.com_rounds = config.get("com_rounds", 100)
        self.local_epochs = config.get("local_epochs", 5)
        self.com_rounds_gen = config.get("com_rounds_gen", 10)
        self.g_local_train_steps = config.get("g_local_train_steps", 50)
        
        # 损失权重
        self.w_kd = config.get("w_kd", 10.0)
        self.w_ce_pre = config.get("w_ce_pre", 0.5)
        self.w_scl = config.get("w_scl", 1.0)
        
        # 初始化组件
        self._initialize_components()
        
        # 任务状态
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        
        # 扩散模型生成器
        self._generator = None
        self.generator_init_embedding = None
        self._init_diffusion_generator()
        
        # 客户端学习器列表
        self.learners = []
    
    def _initialize_components(self):
        """初始化组件"""
        # 初始化聚合器
        self.aggregator = FedAvgAggregator()
        
        # 初始化数据管理器
        self.data_manager = None  # 需要外部设置
        
        self.logger.info("DDDR联邦训练器组件初始化完成")
    
    def _init_diffusion_generator(self):
        """初始化扩散模型生成器"""
        try:
            from omegaconf import OmegaConf
            from ...models.ldm import LatentDiffusion
            
            if not os.path.exists(self.ldm_config_path):
                self.logger.warning(f"扩散模型配置文件不存在: {self.ldm_config_path}")
                return
            
            if not os.path.exists(self.ldm_ckpt_path):
                self.logger.warning(f"扩散模型检查点不存在: {self.ldm_ckpt_path}")
                return
            
            # 加载配置
            config = OmegaConf.load(self.ldm_config_path)
            config.model.params.ckpt_path = self.ldm_ckpt_path
            config['model']["params"]['personalization_config']["params"]['num_classes'] = \
                self.classes_per_task
            
            # 创建模型
            self._generator = LatentDiffusion(**config['model']["params"])
            self._generator.load_state_dict(
                torch.load(self.ldm_ckpt_path, map_location="cpu")["state_dict"], 
                strict=False
            )
            
            # 保存初始嵌入状态
            self.generator_init_embedding = copy.deepcopy(
                self._generator.embedding_manager.state_dict()
            )
            
            # 设置学习率
            self._generator.learning_rate = (
                config.data.params.batch_size * 
                config.model.base_learning_rate
            )
            
            self._generator.to(self.device)
            self.logger.info(f"扩散模型生成器初始化完成，学习率: {self._generator.learning_rate:.2e}")
        except Exception as e:
            self.logger.warning(f"扩散模型初始化失败: {e}")
            self._generator = None
    
    def _create_learner_config(self, client_id: str) -> Dict[str, Any]:
        """创建学习器配置"""
        return {
            "client_id": client_id,
            "base_model": "resnet18",
            "num_classes": self.total_classes,
            "feature_dim": 512,
            "proj_dim": 256,
            "w_kd": self.w_kd,
            "w_ce_pre": self.w_ce_pre,
            "w_scl": self.w_scl,
            "ldm_config": self.ldm_config_path,
            "ldm_ckpt": self.ldm_ckpt_path,
            "pre_size": self.pre_size,
            "cur_size": self.cur_size,
            "n_iter": self.n_iter,
            "syn_imgs_dir": "syn_imgs",
            "optimizer": {
                "type": "sgd",
                "learning_rate": 0.01
            },
            "local_epochs": self.local_epochs,
            "device": self.device,
            "dataset": "cifar100",
            "batch_size": self.batch_size,
            "g_local_train_steps": self.g_local_train_steps,
            "g_local_bs": 12
        }
    
    def _prepare_task_data(self, task_id: int, class_ids: List[int]):
        """准备任务数据"""
        self.logger.info(f"准备任务 {task_id} 数据，类别: {class_ids}")
        
        # 检查是否需要生成合成数据
        if self._generator is not None:
            syn_imgs_dir = os.path.join("syn_imgs", f"task_{task_id}")
            if not self._check_synthetic_data_exists(syn_imgs_dir, class_ids):
                self.logger.info(f"为任务 {task_id} 生成合成数据")
                self._generate_synthetic_data(task_id, class_ids)
    
    def _check_synthetic_data_exists(self, syn_imgs_dir: str, class_ids: List[int]) -> bool:
        """检查合成数据是否存在"""
        if not os.path.exists(syn_imgs_dir):
            return False
        
        for class_id in class_ids:
            class_dir = os.path.join(syn_imgs_dir, str(class_id))
            if not os.path.exists(class_dir):
                return False
            
            # 检查是否有足够的图像
            image_files = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]
            if len(image_files) < self.cur_size:
                return False
        
        return True
    
    def _generate_synthetic_data(self, task_id: int, class_ids: List[int]):
        """生成合成数据"""
        if self._generator is None:
            self.logger.warning("扩散模型未初始化，跳过数据生成")
            return
        
        # 执行联邦类别反演训练
        self.logger.info("开始联邦类别反演训练")
        inv_text_embeds = self._federated_class_inversion(task_id, class_ids)
        
        # 生成图像
        self._synthesis_images(inv_text_embeds, task_id, class_ids)
        
        self.logger.info(f"任务 {task_id} 合成数据生成完成")
    
    def _federated_class_inversion(self, task_id: int, class_ids: List[int]):
        """联邦类别反演训练"""
        if self._generator is None:
            return None
        
        # 重置生成器嵌入
        self._generator.embedding_manager.load_state_dict(self.generator_init_embedding)
        
        # 联邦训练轮次
        prog_bar = tqdm(range(self.com_rounds_gen), desc='联邦类别反演训练')
        for _ in prog_bar:
            local_weights = []
            m = max(int(self.config.get("frac", 1.0) * self.num_clients), 1)
            idxs_users = np.random.choice(range(self.num_clients), m, replace=False)
            
            for idx in idxs_users:
                # 直接使用现有的客户端学习器
                if idx < len(self.learners):
                    learner = self.learners[idx]
                    
                    # 设置训练数据用于生成器训练
                    if self.data_manager is not None:
                        train_dataset = self.data_manager.get_client_data(idx, class_ids)
                        learner.set_train_data(train_dataset, class_ids)
                    
                    # 训练生成器嵌入
                    w = learner.train_generator_embeddings()
                    if w is not None:
                        local_weights.append(copy.deepcopy(w))
            
            # 聚合权重
            if local_weights:
                global_weights = self._average_weights(local_weights, self.config.get('g_sigma', 0))
                self._generator.embedding_manager.load_state_dict(global_weights)
        
        # 返回最终的嵌入状态
        return copy.deepcopy(self._generator.embedding_manager.string_to_param_dict)
    
    def _synthesis_images(self, inv_text_embeds, task_id: int, class_ids: List[int]):
        """合成图像"""
        if self._generator is None or inv_text_embeds is None:
            return
        
        # 导入DDIM采样器
        from ...models.ldm import DDIMSampler
        from einops import rearrange
        from PIL import Image
        
        sampler = DDIMSampler(self._generator)
        
        outdir = os.path.join("syn_imgs", f"task_{task_id}")
        os.makedirs(outdir, exist_ok=True)
        
        prompt = "a photo of *"
        n_samples = 40
        scale = 10.0
        ddim_steps = 50
        ddim_eta = 0.0
        H = 256
        W = 256
        
        # 设置嵌入
        self._generator.embedding_manager.string_to_param_dict = inv_text_embeds
        
        with torch.no_grad():
            for tmp_cls in class_ids:
                base_count = 0
                class_dir = os.path.join(outdir, str(tmp_cls))
                os.makedirs(class_dir, exist_ok=True)
                
                with self._generator.ema_scope():
                    uc = None
                    tmp_cls_tensor = torch.LongTensor([tmp_cls - min(class_ids)] * n_samples).to(self.device)
                    
                    if scale != 1.0:
                        uc = self._generator.get_learned_conditioning(n_samples * [""], tmp_cls_tensor)
                    
                    for _ in trange(self.n_iter, desc=f"生成类别 {tmp_cls}"):
                        c = self._generator.get_learned_conditioning(n_samples * [prompt], tmp_cls_tensor)
                        shape = [4, H//8, W//8]
                        samples_ddim, _ = sampler.sample(
                            S=ddim_steps,
                            conditioning=c,
                            batch_size=n_samples,
                            shape=shape,
                            verbose=False,
                            unconditional_guidance_scale=scale,
                            unconditional_conditioning=uc,
                            eta=ddim_eta
                        )
                        
                        x_samples_ddim = self._generator.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                        
                        for x_sample in x_samples_ddim:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            Image.fromarray(x_sample.astype(np.uint8)).save(
                                os.path.join(class_dir, f"{tmp_cls}-{base_count}.jpg")
                            )
                            base_count += 1
    
    def _average_weights(self, weights_list, sigma=0.0):
        """平均权重"""
        if not weights_list:
            return {}
        
        # 计算平均权重
        avg_weights = {}
        for key in weights_list[0].keys():
            avg_weights[key] = torch.stack([w[key] for w in weights_list]).mean(0)
            
            # 添加差分隐私噪声
            if sigma > 0:
                noise = torch.randn_like(avg_weights[key]) * sigma
                avg_weights[key] = avg_weights[key] + noise
        
        return avg_weights
    
    def _create_task_learners(self) -> List[DDDRLearner]:
        """为当前任务创建学习器"""
        learners = []
        
        for client_id in range(self.num_clients):
            config = self._create_learner_config(f"client_{client_id}")
            learner = DDDRLearner(f"client_{client_id}", config)
            
            # 更新任务信息
            learner.update_task(
                task_id=self._cur_task,
                known_classes=self._known_classes,
                total_classes=self._total_classes
            )
            
            # 设置训练数据
            if self.data_manager is not None:
                class_ids = list(range(self._known_classes, self._total_classes))
                train_dataset = self.data_manager.get_client_data(client_id, class_ids)
                learner.set_train_data(train_dataset, class_ids)
                
                # 设置合成数据 - 数据变换由数据管理器提供
                syn_imgs_dir = "syn_imgs"
                transform = self.data_manager.get_train_transform() if hasattr(self.data_manager, 'get_train_transform') else None
                learner.set_synthetic_data(syn_imgs_dir, self._cur_task, transform)
            
            learners.append(learner)
        
        # 保存学习器列表供后续使用
        self.learners = learners
        
        return learners
    
    async def train(self, **kwargs) -> Dict[str, Any]:
        """执行DDDR联邦训练"""
        self.logger.info("开始DDDR联邦训练")
        
        # 初始化全局模型
        global_weights = self._initialize_global_model()
        
        # 记录训练历史
        training_history = []
        evaluation_history = []
        
        # 执行任务序列
        for task_id in range(self.num_tasks):
            self._cur_task = task_id
            self._known_classes = task_id * self.classes_per_task
            self._total_classes = (task_id + 1) * self.classes_per_task
            
            self.logger.info(f"开始任务 {task_id}: 类别 {self._known_classes}-{self._total_classes}")
            
            # 准备任务数据
            class_ids = list(range(self._known_classes, self._total_classes))
            self._prepare_task_data(task_id, class_ids)
            
            # 创建学习器
            learners = self._create_task_learners()
            
            # 联邦训练轮次
            for round_num in range(self.com_rounds):
                self.logger.info(f"任务 {task_id}, 轮次 {round_num + 1}/{self.com_rounds}")
                
                # 客户端训练
                client_results = []
                for client_id, learner in enumerate(learners):
                    # 设置全局权重
                    learner.set_model_weights(global_weights)
                    
                    # 执行本地训练
                    result = await learner.train_epoch(
                        round_num=round_num,
                        local_epochs=self.local_epochs
                    )
                    
                    client_results.append(result)
                
                # 聚合模型权重
                global_weights = self.aggregator.aggregate(client_results)
                
                # 评估全局模型
                if round_num % 10 == 0 or round_num == self.com_rounds - 1:
                    eval_result = await self._evaluate_global_model(global_weights, learners)
                    evaluation_history.append({
                        "task": task_id,
                        "round": round_num,
                        **eval_result
                    })
                    
                    self.logger.info(
                        f"任务 {task_id}, 轮次 {round_num}: "
                        f"准确率 {eval_result['accuracy']:.4f}, "
                        f"损失 {eval_result['loss']:.4f}"
                    )
                
                # 记录训练历史
                training_history.append({
                    "task": task_id,
                    "round": round_num,
                    "num_clients": len(client_results),
                    "avg_loss": sum(r["loss"] for r in client_results) / len(client_results)
                })
            
            # 任务完成后的处理
            for learner in learners:
                learner.after_task()
            
            self.logger.info(f"任务 {task_id} 完成")
        
        # 返回训练结果
        return {
            "training_history": training_history,
            "evaluation_history": evaluation_history,
            "final_accuracy": evaluation_history[-1]["accuracy"] if evaluation_history else 0.0,
            "total_tasks": self.num_tasks,
            "total_rounds": self.com_rounds
        }
    
    async def evaluate(self, **kwargs) -> Dict[str, Any]:
        """评估全局模型"""
        # 创建评估学习器
        config = self._create_learner_config("eval_client")
        eval_learner = DDDRLearner("eval_client", config)
        
        # 获取测试数据
        test_loader = self._get_test_data()
        
        # 执行评估
        result = await eval_learner.evaluate(test_loader=test_loader)
        
        return result
    
    def _initialize_global_model(self) -> Dict[str, Any]:
        """初始化全局模型"""
        # 创建临时学习器获取初始权重
        config = self._create_learner_config("temp_client")
        temp_learner = DDDRLearner("temp_client", config)
        
        return temp_learner.get_model_weights()
    
    def _get_test_data(self) -> DataLoader:
        """获取测试数据"""
        # 这里应该根据实际的数据管理器来获取数据
        # 暂时返回None，实际使用时需要实现
        self.logger.warning("需要实现测试数据获取")
        return None
    
    async def _evaluate_global_model(self, global_weights: Dict[str, Any], learners: List[DDDRLearner]) -> Dict[str, Any]:
        """评估全局模型"""
        # 使用第一个学习器进行评估
        if learners:
            learner = learners[0]
            learner.set_model_weights(global_weights)
            
            test_loader = self._get_test_data()
            if test_loader is not None:
                return await learner.evaluate(test_loader=test_loader)
        
        # 返回默认结果
        return {
            "accuracy": 0.0,
            "loss": 0.0,
            "num_samples": 0
        }
    
    def get_task_info(self) -> Dict[str, Any]:
        """获取任务信息"""
        return {
            "cur_task": self._cur_task,
            "known_classes": self._known_classes,
            "total_classes": self._total_classes,
            "num_tasks": self.num_tasks,
            "classes_per_task": self.classes_per_task
        }
    
    def set_data_manager(self, data_manager):
        """设置数据管理器"""
        self.data_manager = data_manager
