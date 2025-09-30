"""
DDDR联邦训练器

基于FedCL框架实现DDDR的完整联邦训练流程，负责全局联邦逻辑。
"""

import os
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from einops import rearrange
from copy import deepcopy
from typing import Dict, Any, List, Optional
from omegaconf import OmegaConf

from ...fl.server import FLTrainerBase
from ...fl.results import EvaluationResult
from ...methods.learners.dddr import DDDRLearner
from ...methods.aggregators import FedAvgAggregator
from ...api.decorators import trainer
from ...models.ldm import LatentDiffusion


@trainer("dddr")
class DDDRFederationTrainer(FLTrainerBase):
    """DDDR联邦训练器 - 基于DDDR-master OURs.py设计，适配FedCL架构"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 基本配置
        self.num_clients = config.get("num_clients", 10)
        self.num_tasks = config.get("num_tasks", 5)
        self.classes_per_task = config.get("classes_per_task", 10)
        self.total_classes = config.get("total_classes", 50)
        
        # 训练参数（完全对齐ours.py的args超参数）
        self.com_round = config.get("com_round", 10)  # 对应ours.py的com_round
        self.com_round_gen = config.get("com_round_gen", 5)  # 对应ours.py的com_round_gen
        self.local_ep = config.get("local_ep", 1)  # 对应ours.py的local_ep
        self.frac = config.get("frac", 1.0)  # 对应ours.py的frac
        self.num_users = config.get("num_users", self.num_clients)  # 对应ours.py的num_users
        self.batch_size = config.get("batch_size", 32)
        self.learning_rate = config.get("learning_rate", 0.001)
        
        # DDDR特定参数（完全对齐ours.py）
        self.pre_size = config.get("pre_size", 200)
        self.cur_size = config.get("cur_size", 50)
        self.n_iter = config.get("n_iter", 2)
        self.g_local_train_steps = config.get("g_local_train_steps", 5)  # 对应ours.py的g_local_train_steps
        self.w_kd = config.get("w_kd", 10.0)  # 对应ours.py的w_kd
        self.w_ce_pre = config.get("w_ce_pre", 0.5)  # 对应ours.py的w_ce_pre
        self.w_scl = config.get("w_scl", 1.0)  # 对应ours.py的w_scl
        self.g_sigma = config.get("g_sigma", 0.0)  # 对应ours.py的g_sigma
        self.classifer_dp = config.get("classifer_dp", 0.0)  # 对应ours.py的classifer_dp（注意拼写）
        self.save_cls_embeds = config.get("save_cls_embeds", False)  # 对应ours.py的save_cls_embeds
        
        # LDM配置
        self.ldm_config_path = config.get("ldm_config")
        self.ldm_ckpt_path = config.get("ldm_ckpt")
        
        # 状态变量
        self.current_task = 0
        self.known_classes = 0
        self.total_classes_seen = 0
        
        # 模型组件
        self._generator = None
        self._classifier = None
        
        # Learner代理系统
        self.learner_proxies = {}
        
        # 初始化组件（代理创建延后至注册事件触发）
        self._init_aggregator()
        self._init_diffusion_generator()
        self._init_classifier()
        
        # 🆕 确保通信系统已初始化（由抽象基类管理）
        self._ensure_communication_initialized()
        
        self.logger.info("✅ DDDRFederationTrainer 初始化完成")
    
    def _init_aggregator(self):
        """初始化聚合器 - 使用统一的聚合器"""
        # 使用父类的聚合器，不需要重复初始化
        if not hasattr(self, 'aggregator') or self.aggregator is None:
            self.aggregator = FedAvgAggregator()
        self.logger.info("✅ 聚合器初始化完成")
    
    def _init_diffusion_generator(self):
        """初始化扩散生成器"""
        if not self.ldm_config_path or not self.ldm_ckpt_path:
            raise ValueError("LDM配置路径和检查点路径必须提供")
        
        try:
            # 加载LDM配置
            ldm_config = OmegaConf.load(self.ldm_config_path)
            
            # 提取必需的配置
            first_stage_config = ldm_config.model.params.first_stage_config
            cond_stage_config = ldm_config.model.params.cond_stage_config
            personalization_config = ldm_config.model.params.personalization_config
            
            # 创建参数字典，避免重复
            model_params = dict(ldm_config.model.params)
            # 移除已经单独传递的参数
            model_params.pop('first_stage_config', None)
            model_params.pop('cond_stage_config', None)
            model_params.pop('personalization_config', None)
            
            # 初始化扩散模型
            self._generator = LatentDiffusion(
                first_stage_config=first_stage_config,
                cond_stage_config=cond_stage_config,
                personalization_config=personalization_config,
                **model_params
            )
            
            # 加载预训练权重
            if os.path.exists(self.ldm_ckpt_path):
                checkpoint = torch.load(self.ldm_ckpt_path, map_location="cpu")
                self._generator.load_state_dict(checkpoint, strict=False)
                self.logger.info(f"✅ 扩散模型权重加载成功: {self.ldm_ckpt_path}")
            else:
                self.logger.warning(f"⚠️ 检查点文件不存在: {self.ldm_ckpt_path}")
            
        except Exception as e:
            self.logger.error(f" 扩散模型初始化失败: {e}")
            raise
    
    def _init_classifier(self):
        """初始化分类器 - 在FedCL中，分类器由learner管理，trainer不直接初始化"""
        # 在FedCL架构中，分类器（网络）由learner端管理
        # trainer只负责协调和聚合，不直接持有分类器实例
        self._classifier = None
        self.logger.info(" 分类器初始化完成（由learner管理）")
    
    def _init_learner_proxies(self):
        """
        初始化Learner代理系统
        
        在DDDR中，learner代理是在客户端注册时由服务端自动创建的，
        这里只需要确保相关变量被正确初始化。
        实际的代理创建会在start_server()中的_on_register回调中进行。
        """
        # 确保learner_proxies字典存在
        if not hasattr(self, '_learner_proxies'):
            self._learner_proxies = {}
        
        # 对于DDDR，我们还需要访问父类的_learner_proxies
        # 这样get_learner_proxy等方法可以正常工作
        self.learner_proxies = self._learner_proxies
        
        self.logger.info("🔧 Learner代理系统已初始化，等待客户端注册")
    
    def _get_communication_backend(self):
        """获取通信后端 - 使用AbstractFederationTrainer启动的server通信"""
        if hasattr(self, '_server_comm') and self._server_comm is not None:
            return self._server_comm
        raise RuntimeError("Server communication not initialized; call start_server() first")
    
    def setup_training(self, data_manager=None):
        """设置训练环境"""
        self.logger.info("🔧 设置DDDR训练环境")
        
        # 准备任务数据
        self._prepare_task_data(data_manager)
        
        # 设置设备
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self._generator.to(self.device)
        self._classifier.to(self.device)
        
        self.logger.info(f"✅ 训练环境设置完成，设备: {self.device}")
    
    def _prepare_task_data(self, data_manager):
        """准备任务数据"""
        if data_manager is None:
            raise ValueError("数据管理器不能为空，必须提供真实的数据管理器")
        
        self.logger.info("📊 使用真实数据管理器")
        self._load_real_task_data(data_manager)
    
    def _load_real_task_data(self, data_manager):
        """加载真实任务数据"""
        # 获取当前任务的数据
        task_data = data_manager.get_task_data(self.current_task)
        
        # 更新类别信息
        self.known_classes = self.total_classes_seen
        self.total_classes_seen += self.classes_per_task
        
        # 存储任务数据
        self.current_task_data = task_data
        
        self.logger.info(f"✅ 任务 {self.current_task + 1} 数据加载完成，类别范围: {self.known_classes}-{self.total_classes_seen}")
    
    def execute_client_round(self, round_idx: int, client_ids: List[int]) -> List[Dict[str, Any]]:
        """执行客户端训练轮次"""
        self.logger.info(f" 执行客户端轮次 {round_idx + 1}")
        
        client_results = []
        
        # 通过learner代理执行客户端训练
        for client_id in client_ids:
            client_id_str = f"client_{client_id}"
            
            if client_id_str in self.learner_proxies:
                learner_proxy = self.learner_proxies[client_id_str]
                
                try:
                    # 通过代理执行客户端训练
                    result = learner_proxy.train_epoch(
                        epochs=self.local_epochs,
                        batch_size=self.batch_size,
                        learning_rate=self.learning_rate
                    )
                    
                    client_results.append({
                        'client_id': client_id,
                        'round': round_idx,
                        'status': 'completed',
                        'metrics': result.get('metrics', {}),
                        'weights': result.get('weights', {})
                    })
                    
                except Exception as e:
                    self.logger.error(f"客户端 {client_id_str} 训练失败: {e}")
                    client_results.append({
                        'client_id': client_id,
                        'round': round_idx,
                        'status': 'failed',
                        'error': str(e)
                    })
            else:
                self.logger.warning(f"客户端 {client_id_str} 的learner代理不存在")
        
        return client_results
    
    def execute_server_aggregation(self, client_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """执行服务端聚合"""
        self.logger.info("🔗 执行服务端聚合")
        
        # 使用统一的聚合器进行聚合
        if client_results:
            # 提取成功的客户端权重
            successful_results = [r for r in client_results if r.get('status') == 'completed']
            
            if successful_results:
                # 提取权重进行聚合
                weights_list = [r.get('weights', {}) for r in successful_results]
                
                # 使用聚合器聚合权重
                aggregated_weights = self.aggregator.aggregate(weights_list)
                
                # 更新全局模型
                self._update_global_model(aggregated_weights)
                
                aggregation_result = {
                    'status': 'completed',
                    'num_clients': len(successful_results),
                    'aggregation_method': 'fedavg',
                    'aggregated_weights': aggregated_weights
                }
            else:
                aggregation_result = {
                    'status': 'no_successful_clients',
                    'num_clients': 0
                }
        else:
            aggregation_result = {
                'status': 'no_clients',
                'num_clients': 0
            }
        
        return aggregation_result
    
    def incremental_train(self, data_manager=None, task_id=None):
        """
        增量训练 - 完全基于DDDR-master ours.py的incremental_train设计
        
        每个任务的流程：
        1. 任务初始化：更新任务状态和类别信息
        2. 数据初始化：由用户在外部完成（init_dataloader逻辑在example中）
        3. 类反演：如果需要合成图像，进行联邦类反演
        4. 图像生成：如果需要合成图像，生成合成图像
        5. 合成数据设置：设置合成数据加载器
        6. 联邦分类器训练：进行com_round轮联邦训练
        """
        if task_id is None:
            self.current_task += 1
        else:
            self.current_task = task_id
            
        self.logger.info(f"🚀 开始DDDR增量训练 - 任务 {self.current_task}")
        
        # 1. 任务状态更新（对应ours.py的setup_seed和任务计数器）
        self._update_task_state(data_manager)
        
        # 2. 数据初始化由用户在外部完成（不在框架内处理）
        # 对应ours.py的init_dataloader，但用户自己负责数据准备
        
        # 3. 类反演阶段（对应ours.py的_class_inversion）
        inv_text_embeds = None
        if self.need_syn_imgs:
            self.logger.info("开始联邦类反演")
            inv_text_embeds = self._class_inversion()
            
            # 4. 图像合成阶段（对应ours.py的_synthesis_imgs）
            self.logger.info("开始生成合成图像")
            self._synthesis_imgs(inv_text_embeds)
        
        # 5. 合成数据初始化（对应ours.py的_init_syn_dataloader）
        self.logger.info("初始化合成数据加载器")
        self._init_syn_dataloader()
        
        # 6. 联邦分类器训练阶段（对应ours.py的_fl_train）
        # 这里进行com_round轮联邦训练
        self.logger.info(f"开始联邦分类器训练 - {self.com_round} 轮")
        self._fl_train()
        
        # 7. 任务完成后处理（对应ours.py的after_task）
        self._after_task()
        
        self.logger.info(f"DDDR增量训练完成 - 任务 {self.current_task}")
    
    def _update_task_state(self, data_manager):
        """更新任务状态 - 对应ours.py中incremental_train的前几行"""
        # 对应 setup_seed(self.seed)
        if hasattr(self, 'seed'):
            import random
            import numpy as np
            import torch
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
        
        # 对应 self._cur_task += 1 和类别计算
        if data_manager:
            task_size = data_manager.get_task_size(self.current_task)
            self.total_classes_seen = self.known_classes + task_size
            
            self.logger.info(f"Learning on {self.known_classes}-{self.total_classes_seen}")
            
            # 更新生成器的类别数（如果需要）
            if self._generator and hasattr(self._generator, 'embedding_manager'):
                # 这里可能需要根据实际LDM实现来调整
                pass
        else:
            self.logger.warning("未提供data_manager，使用配置中的类别信息")
    
    def _after_task(self):
        """任务完成后处理 - 对应ours.py的after_task"""
        self.known_classes = self.total_classes_seen
        # 这里可以添加保存模型等逻辑
        self.logger.info(f"任务 {self.current_task} 完成，已知类别数: {self.known_classes}")
    
    @property
    def need_syn_imgs(self):
        """判断是否需要生成合成图像"""
        return self.config.get('syn_image_path') is None
    
    def _class_inversion(self):
        """联邦类反演 - 完全对齐ours.py的_class_inversion方法"""
        self.logger.info("🔄 开始类别反演")
        
        # 将生成器移到GPU并重置嵌入到初始状态（与ours.py line 206-207一致）
        device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self._generator = self._generator.to(device)
        
        if hasattr(self, 'generator_init_embedding'):
            self._generator.embedding_manager.load_state_dict(self.generator_init_embedding)
        
        # 获取超参数（与ours.py一致）
        com_round_gen = self.com_round_gen
        frac = self.frac
        num_users = self.num_users
        g_sigma = self.g_sigma
        
        # 进度条（与ours.py line 208一致）
        prog_bar = tqdm(range(com_round_gen), desc='Class Inversion')
        
        for _ in prog_bar:
            local_weights = []
            
            # 客户端选择（与ours.py line 211-212一致）
            m = max(int(frac * num_users), 1)
            idxs_users = np.random.choice(range(min(num_users, len(self._learner_proxies))), m, replace=False)
            client_ids = list(self._learner_proxies.keys())
            
            # 客户端本地生成器训练（与ours.py line 213-218一致）
            for idx in idxs_users:
                if idx >= len(client_ids):
                    continue
                    
                client_id = client_ids[idx]
                try:
                    proxy = self._learner_proxies[client_id]
                    # 对应ours.py的_local_update_g调用，返回embedding_manager.state_dict()
                    w = proxy.call_method("train_generator_embeddings")
                    if w is not None:
                        local_weights.append(deepcopy(w))
                except Exception as e:
                    self.logger.warning(f"客户端 {client_id} 生成器训练失败: {e}")
            
            # 权重聚合（与ours.py line 219-220一致）
            if local_weights:
                global_weights = self._average_embedding_weights(local_weights, g_sigma=g_sigma)
                self._generator.embedding_manager.load_state_dict(global_weights)
        
        # 导出最终的类别嵌入（与ours.py line 221一致）
        inv_text_embeds = deepcopy(self._generator.embedding_manager.string_to_param_dict)
        
        # 保存类别嵌入（与ours.py line 222一致）
        if self.save_cls_embeds:
            self._save_class_embeddings(inv_text_embeds)
        
        self.logger.info("✅ 类别反演完成")
        return inv_text_embeds
    
    def _fl_train(self):
        """
        联邦训练 - 完全对齐ours.py的_fl_train方法
        
        区分首任务和增量任务：
        - 首任务：使用local_update
        - 增量任务：使用local_finetune（包含回放和知识蒸馏）
        """
        self.logger.info(f"🎓 开始联邦训练 - Task {self.current_task}")
        
        # 将网络移到GPU（与ours.py line 123一致）
        device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # 进度条（与ours.py line 124一致）
        prog_bar = tqdm(range(self.com_round))
        
        for com in prog_bar:
            local_weights = []
            
            # 客户端选择（与ours.py line 127-128一致）
            m = max(int(self.frac * self.num_users), 1)
            idxs_users = np.random.choice(range(min(self.num_users, len(self._learner_proxies))), m, replace=False)
            client_ids = list(self._learner_proxies.keys())
            
            # 客户端本地训练（与ours.py line 129-135一致）
            for idx in idxs_users:
                if idx >= len(client_ids):
                    continue
                    
                client_id = client_ids[idx]
                proxy = self._learner_proxies[client_id]
                
                try:
                    if self.current_task == 0:
                        # 首任务：使用local_update（对应ours.py line 131）
                        w = proxy.call_method("local_update")
                    else:
                        # 增量任务：使用local_finetune（对应ours.py line 133-134）
                        w = proxy.call_method("local_finetune")
                    
                    if w is not None:
                        local_weights.append(deepcopy(w))
                        
                except Exception as e:
                    self.logger.error(f"客户端 {client_id} 训练失败: {e}")
            
            # 聚合权重（与ours.py line 137一致）
            if local_weights:
                global_weights = self._average_weights(local_weights, dp_si=self.classifer_dp)
                
                # 更新全局网络（与ours.py line 138一致）
                self._update_global_classifier(global_weights)
                
                # 测试（与ours.py line 140-143一致）
                test_acc = self._compute_test_accuracy()
                info = f"Task {self.current_task}, Epoch {com + 1}/{self.com_round} => Test_accy {test_acc:.2f}"
                prog_bar.set_description(info)
        
        self.logger.info("✅ 联邦训练完成")
    
    def _init_syn_dataloader(self):
        """
        初始化合成数据加载器 - 对应ours.py的_init_syn_dataloader
        
        为各客户端设置当前任务的合成数据和历史任务的合成数据
        """
        if not self.need_syn_imgs:
            # 如果用户提供了syn_image_path，跳过合成数据设置
            self.logger.info("使用用户提供的合成图像路径，跳过合成数据加载器设置")
            return
        
        syn_imgs_dir = self.config.get("syn_imgs_dir", os.path.join(self.config.get("save_dir", "outputs"), "syn_imgs"))
        
        # 为每个客户端设置合成数据
        for client_id, proxy in self._learner_proxies.items():
            try:
                # 设置当前任务的合成数据
                proxy.call_method("set_current_syn_data", {
                    "syn_imgs_dir": syn_imgs_dir,
                    "task_id": self.current_task,
                    "cur_size": self.cur_size
                })
                
                # 设置历史任务的合成数据（如果不是首任务）
                if self.current_task > 0:
                    proxy.call_method("set_replay_syn_data", {
                        "syn_imgs_dir": syn_imgs_dir,
                        "current_task": self.current_task,
                        "pre_size": self.pre_size
                    })
                
                self.logger.debug(f"✅ 客户端 {client_id} 合成数据设置完成")
                
            except Exception as e:
                self.logger.error(f"客户端 {client_id} 合成数据设置失败: {e}")
    
    def _init_text_embeddings(self):
        """初始化文本嵌入"""
        # 为当前任务的每个类别创建文本嵌入
        task_classes = list(range(self.known_classes, self.total_classes_seen))
        
        inv_text_embeds = {}
        for class_id in task_classes:
            # 创建类别文本提示
            text_prompt = f"a photo of class {class_id}"
            
            # 获取BERT嵌入
            with torch.no_grad():
                text_embeds = self._generator.get_learned_conditioning([text_prompt])
                inv_text_embeds[f"class_{class_id}"] = text_embeds.clone()
        
        self.logger.info(f"✅ 初始化了 {len(inv_text_embeds)} 个类别的文本嵌入")
        return inv_text_embeds
    
    def _update_generator_embeddings(self, state_dict: Dict[str, Any]):
        """更新生成器嵌入（直接load_state_dict）"""
        try:
            self._generator.embedding_manager.load_state_dict(state_dict, strict=False)
        except Exception as e:
            self.logger.error(f"加载生成器嵌入失败: {e}")

    def _average_embedding_weights(self, weights_list: List[Dict[str, Any]], g_sigma: float = 0.0) -> Dict[str, Any]:
        """对embedding_manager.state_dict做元素均值（完全对齐ours.py的average_weights）"""
        if not weights_list:
            return {}
        # 深拷贝第一份结构作为累加器
        import copy, torch
        agg = copy.deepcopy(weights_list[0])
        # 对其余权重逐元素相加
        for w in weights_list[1:]:
            for k in agg:
                if isinstance(agg[k], dict) and isinstance(w.get(k), dict):
                    # 递归到下一层
                    for kk in agg[k]:
                        if isinstance(agg[k][kk], torch.Tensor) and isinstance(w[k].get(kk), torch.Tensor):
                            agg[k][kk] = agg[k][kk] + w[k][kk]
                elif isinstance(agg[k], torch.Tensor) and isinstance(w.get(k), torch.Tensor):
                    agg[k] = agg[k] + w[k]
        # 求平均
        num = float(len(weights_list))
        for k in agg:
            if isinstance(agg[k], dict):
                for kk in agg[k]:
                    if isinstance(agg[k][kk], torch.Tensor):
                        agg[k][kk] = agg[k][kk] / num
            elif isinstance(agg[k], torch.Tensor):
                agg[k] = agg[k] / num
        # 可选添加高斯噪声（对应ours.py的g_sigma参数）
        if g_sigma and g_sigma > 0:
            for k in agg:
                if isinstance(agg[k], dict):
                    for kk in agg[k]:
                        if isinstance(agg[k][kk], torch.Tensor):
                            agg[k][kk] = agg[k][kk] + torch.randn_like(agg[k][kk]) * g_sigma
                elif isinstance(agg[k], torch.Tensor):
                    agg[k] = agg[k] + torch.randn_like(agg[k]) * g_sigma
        return agg
    
    def _average_weights(self, weights_list: List[Dict[str, Any]], dp_si: float = 0.0) -> Dict[str, Any]:
        """对分类器权重做FedAvg聚合（对齐ours.py的average_weights，支持差分隐私）"""
        if not weights_list:
            return {}
        
        # 使用FedAvg聚合器
        aggregation_result = self.aggregator.aggregate([
            {"model_weights": w, "num_samples": 1} for w in weights_list
        ])
        
        aggregated_weights = aggregation_result["aggregated_weights"]
        
        # 添加差分隐私噪声（对应ours.py的dp_si参数）
        if dp_si > 0:
            aggregated_weights = self._add_classifier_noise(aggregated_weights, dp_si)
        
        return aggregated_weights
    
    def _synthesis_imgs(self, inv_text_embeds: Dict[str, torch.Tensor]):
        """生成合成图像（对齐DDDR ours.py的_synthesis_imgs）"""
        self.logger.info("🎨 开始生成合成图像")

        # 将类嵌入字典设置到生成器
        try:
            self._generator.embedding_manager.string_to_param_dict = inv_text_embeds
        except Exception as e:
            self.logger.warning(f"设置类嵌入失败: {e}")

        # 采样器
        try:
            from ...models.ldm import DDIMSampler
        except Exception as e:
            self.logger.error(f"无法导入DDIMSampler: {e}")
            return

        sampler = DDIMSampler(self._generator)

        # 输出目录：syn_imgs_dir/task_{t}/class_id/*.jpg
        syn_root = self.config.get("syn_imgs_dir", os.path.join(self.config.get("save_dir", "outputs"), "syn_imgs"))
        outdir = os.path.join(syn_root, f"task_{self.current_task}")
        os.makedirs(outdir, exist_ok=True)

        # 生成参数（与ours.py保持一致）
        prompt = "a photo of *"
        n_samples = int(self.config.get("n_samples", 40))
        scale = float(self.config.get("scale", 10.0))
        ddim_steps = int(self.config.get("ddim_steps", 50))
        ddim_eta = float(self.config.get("ddim_eta", 0.0))
        H = int(self.config.get("img_h", 256))
        W = int(self.config.get("img_w", 256))
        num_iter = int(self.config.get("n_iter", self.n_iter if hasattr(self, 'n_iter') else 2))

        # 计算每类生成数量
        if len(inv_text_embeds) == 0:
            self.logger.warning("没有可用的类嵌入，跳过图像生成")
            return
        num_images_per_class = max(1, int(self.pre_size // len(inv_text_embeds))) if hasattr(self, 'pre_size') else 40

        # 获取类别ID集合与最小类别ID（相对ID计算）
        try:
            class_ids = [int(name.split('_')[-1]) for name in inv_text_embeds.keys()]
        except Exception:
            class_ids = list(range(len(inv_text_embeds)))
        min_class_id = min(class_ids) if class_ids else 0

        device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self._generator = self._generator.to(device)

        with torch.no_grad():
            for tmp_cls in class_ids:
                base_count = 0
                class_dir = os.path.join(outdir, str(tmp_cls))
                os.makedirs(class_dir, exist_ok=True)

                with self._generator.ema_scope():
                    uc = None
                    tmp_cls_tensor = torch.LongTensor([tmp_cls - min_class_id] * n_samples).to(device)
                    if scale != 1.0:
                        uc = self._generator.get_learned_conditioning(n_samples * [""], tmp_cls_tensor)

                    for _ in trange(num_iter, desc=f"Sampling {tmp_cls}"):
                        c = self._generator.get_learned_conditioning(n_samples * [prompt], tmp_cls_tensor)
                        shape = [4, H // 8, W // 8]
                        samples_ddim, _ = sampler.sample(
                            S=ddim_steps,
                            conditioning=c,
                            batch_size=n_samples,
                            shape=shape,
                            verbose=False,
                            unconditional_guidance_scale=scale,
                            unconditional_conditioning=uc,
                            eta=ddim_eta,
                        )
                        x_samples_ddim = self._generator.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                        for x_sample in x_samples_ddim:
                            x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            img_path = os.path.join(class_dir, f"{tmp_cls}-{base_count}.jpg")
                            try:
                                from PIL import Image
                                Image.fromarray(x_sample.astype(np.uint8)).save(img_path)
                            except Exception as e:
                                self.logger.warning(f"保存图像失败 {img_path}: {e}")
                            base_count += 1

        self.logger.info("✅ 合成图像生成完成")
    
    def _federated_learning_train(self):
        """联邦学习训练"""
        self.logger.info(f"🎓 开始联邦学习训练 - {self.com_rounds} 轮次")
        
        # 执行标准的联邦训练流程
        for round_idx in range(self.com_rounds):
            self.logger.info(f"  轮次 {round_idx + 1}/{self.com_rounds}")
            
            # 选择参与的客户端
            num_participants = max(int(0.5 * self.num_clients), 1)
            participant_ids = np.random.choice(range(self.num_clients), num_participants, replace=False)
            
            # 执行客户端训练
            client_results = self.execute_client_round(round_idx, participant_ids)
            
            # 执行服务端聚合
            aggregation_result = self.execute_server_aggregation(client_results)
            
            self.logger.debug(f"轮次 {round_idx + 1}/{self.com_rounds} 完成")
        
        self.logger.info("✅ 联邦学习训练完成")
    
    def train(self, num_rounds: int, **kwargs) -> dict:
        """
        执行DDDR联邦训练 - 持续学习任务
        
        实现完整的DDDR持续学习流程：
        1. 启动服务端通信
        2. 等待客户端注册
        3. 执行多个任务的增量训练
        4. 每个任务包含：类反演 → 图像生成 → 联邦分类器训练
        """
        import time
        start_time = time.time()
        
        self.logger.info(f" 开始DDDR联邦持续学习训练 - {self.num_tasks} 个任务")
        
        try:
            # 🆕 服务端已在初始化阶段启动，客户端已注册
            # 执行多个任务的增量训练
            for task_idx in range(self.num_tasks):
                self.logger.info(f"📋 开始任务 {task_idx + 1}/{self.num_tasks}")
                
                # 执行单个任务的增量训练
                self.incremental_train(task_id=task_idx)
                
                # 任务间评估
                if task_idx < self.num_tasks - 1:  # 不是最后一个任务
                    test_acc = self._compute_test_accuracy()
                    self.logger.info(f"✅ 任务 {task_idx + 1} 完成，准确率: {test_acc:.2f}%")
            
            # 计算训练时间
            training_time = time.time() - start_time
            
            # 构建并返回训练结果
            result = self.build_training_result(
                num_rounds=self.num_tasks,  # 使用任务数作为轮数
                training_time=training_time,
                execution_mode="pseudo_federation"
            )
            
            self.logger.info(f"✅ DDDR联邦持续学习训练完成，耗时: {training_time:.2f}秒")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ DDDR联邦训练失败: {e}")
            raise
        finally:
            # 停止服务端通信
            self.stop_server()
    

    
    def evaluate(self, test_data: Optional[Any] = None, **kwargs) -> EvaluationResult:
        """执行联邦模型评估"""
        self.logger.info("🔍 开始DDDR联邦评估")
        
        if not self._learner_proxies:
            self.logger.warning("⚠️ 没有可用的客户端代理，无法进行评估")
            return EvaluationResult(
                metrics={"accuracy": 0.0, "loss": 1.0},
                task_metrics={},
                evaluation_time=0.0,
                metadata={"error": "no_clients"}
            )
        
        import time
        start_time = time.time()
        
        # 收集所有客户端的评估结果
        client_evaluations = []
        for client_id, proxy in self._learner_proxies.items():
            try:
                result = proxy.call_method("evaluate", {
                    "test_loader": test_data,
                    "metrics": ["accuracy", "loss"]
                })
                client_evaluations.append(result)
                
            except Exception as e:
                self.logger.error(f"客户端 {client_id} 评估失败: {e}")
        
        # 聚合评估结果
        if client_evaluations:
            total_samples = sum(r.get("num_samples", 0) for r in client_evaluations)
            if total_samples > 0:
                weighted_accuracy = sum(
                    r.get("accuracy", 0) * r.get("num_samples", 0) 
                    for r in client_evaluations
                ) / total_samples
                
                weighted_loss = sum(
                    r.get("loss", 0) * r.get("num_samples", 0) 
                    for r in client_evaluations
                ) / total_samples
            else:
                weighted_accuracy = 0.0
                weighted_loss = 1.0
        else:
            weighted_accuracy = 0.0
            weighted_loss = 1.0
        
        evaluation_time = time.time() - start_time
        
        metrics = {
            "accuracy": weighted_accuracy,
            "loss": weighted_loss,
            "num_clients": len(client_evaluations),
            "total_samples": total_samples if client_evaluations else 0
        }
        
        task_metrics = {
            f"task_{self.current_task}": {
                "accuracy": weighted_accuracy,
                "loss": weighted_loss
            }
        }
        
        self.logger.info(
            f"✅ 联邦评估完成 - 准确率: {weighted_accuracy:.4f}, "
            f"损失: {weighted_loss:.4f}, 客户端数: {len(client_evaluations)}"
        )
        
        return EvaluationResult(
            metrics=metrics,
            task_metrics=task_metrics,
            evaluation_time=evaluation_time,
            metadata={"current_task": self.current_task}
        )
    
    # ============ 辅助方法 ============
    
    def _add_classifier_noise(self, weights: Dict[str, Any], noise_sigma: float) -> Dict[str, Any]:
        """为分类器权重添加差分隐私噪声"""
        import torch
        noisy_weights = {}
        for k, v in weights.items():
            if isinstance(v, torch.Tensor):
                noise = torch.randn_like(v) * noise_sigma
                noisy_weights[k] = v + noise
            else:
                noisy_weights[k] = v
        return noisy_weights
    
    def _update_global_classifier(self, weights: Dict[str, Any]):
        """更新全局分类器并广播给所有客户端 - 对应ours.py的self._network.load_state_dict"""
        # 在FedCL中，trainer不直接持有分类器，只负责广播权重给learner
        # 对应ours.py的: self._network.load_state_dict(global_weights)
        
        # 广播给所有客户端
        for client_id, proxy in self._learner_proxies.items():
            try:
                proxy.call_method("set_model_weights", {"weights": weights})
            except Exception as e:
                self.logger.error(f"更新客户端 {client_id} 模型失败: {e}")
        
        self.logger.debug(f"✅ 全局分类器权重已广播给 {len(self._learner_proxies)} 个客户端")
    
    def _compute_test_accuracy(self) -> float:
        """计算测试准确率 - 对应ours.py的_compute_accuracy方法"""
        try:
            # 在FedCL中，trainer通过learner代理来获取评估结果
            # 对应ours.py的: test_acc = self._compute_accuracy(self._network, self.test_loader)
            
            client_accuracies = []
            for client_id, proxy in self._learner_proxies.items():
                try:
                    # 调用learner的evaluate方法
                    result = proxy.call_method("evaluate")
                    if result and "accuracy" in result:
                        client_accuracies.append(result["accuracy"] * 100)
                except Exception as e:
                    self.logger.debug(f"客户端 {client_id} 评估失败: {e}")
            
            if client_accuracies:
                # 返回所有客户端的平均准确率
                avg_accuracy = sum(client_accuracies) / len(client_accuracies)
                return avg_accuracy
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"测试准确率计算失败: {e}")
            return 0.0
    
    def _save_class_embeddings(self, inv_text_embeds: Dict[str, Any]):
        """保存类嵌入 - 对应ours.py的save_cls_embeds"""
        try:
            save_dir = self.config.get("save_dir", "outputs")
            cls_embeds_dir = os.path.join(save_dir, "cls_embeds_ckpt")
            os.makedirs(cls_embeds_dir, exist_ok=True)
            
            # 获取类别范围
            min_class_id = self.known_classes
            max_class_id = self.total_classes_seen - 1
            
            # 保存嵌入权重
            embed_path = os.path.join(
                cls_embeds_dir, 
                f"{min_class_id}-{max_class_id}_embedding_manager.pt"
            )
            
            if self._generator and hasattr(self._generator, 'embedding_manager'):
                torch.save(self._generator.embedding_manager.state_dict(), embed_path)
                self.logger.info(f"✅ 类嵌入已保存: {embed_path}")
            
        except Exception as e:
            self.logger.error(f"保存类嵌入失败: {e}")
