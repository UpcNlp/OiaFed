    def evaluate_learner(self, learner_id: str, learner: Any) -> Dict[str, Any]:
        """评估单个学习器
        
        Args:
            learner_id: 学习器ID
            learner: 学习器实例
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        try:
            if not self.evaluation_engine:
                self.logger.warning("评估引擎未初始化，跳过评估")
                return {}
            
            evaluation_config = self.config_manager.get_evaluation_config()
            self.logger.debug(f"评估配置结构: {evaluation_config}")
            
            # 支持新格式: evaluation.tasks 列表
            evaluation_tasks = []
            
            if "tasks" in evaluation_config:
                # 过滤出当前learner的任务
                all_tasks = evaluation_config["tasks"]
                learner_tasks = [task for task in all_tasks if task.get("learner") == learner_id]
                
                if learner_tasks:
                    self.logger.debug(f"找到 {len(learner_tasks)} 个针对learner {learner_id} 的评估任务")
                    for task in learner_tasks:
                        evaluation_tasks.append({
                            "evaluator": task.get("evaluator"),
                            "test_dataset": task.get("test_data"),  # 注意字段名映射
                            "name": f"{task.get('evaluator')}_{task.get('test_data')}"
                        })
                else:
                    self.logger.debug(f"在tasks列表中未找到针对learner {learner_id} 的评估任务")
            
            if not evaluation_tasks:
                self.logger.debug(f"没有为learner {learner_id} 找到有效的评估任务")
                return {}
                
            results = {}
            
            # 执行评估任务
            for task in evaluation_tasks:
                evaluator_id = task["evaluator"]
                test_dataset_id = task["test_dataset"]
                task_name = task.get("name", f"{evaluator_id}_{test_dataset_id}")
                
                self.logger.debug(f"执行评估任务: {task_name}")
                
                if evaluator_id not in self.evaluators:
                    self.logger.warning(f"评估器 {evaluator_id} 不存在，跳过任务 {task_name}")
                    continue
                    
                if test_dataset_id not in self.test_dataloaders:
                    self.logger.warning(f"测试数据集 {test_dataset_id} 不存在，跳过任务 {task_name}")
                    continue
                
                evaluator = self.evaluators[evaluator_id]
                test_dataloader = self.test_dataloaders[test_dataset_id]
                
                try:
                    # 执行评估
                    eval_result = evaluator.evaluate(
                        model=learner.get_model() if hasattr(learner, 'get_model') else learner,
                        dataloader=test_dataloader,
                        learner_id=learner_id
                    )
                    
                    # 保存结果
                    results[task_name] = eval_result
                    
                    self.logger.info(f"✅ 评估任务完成: {task_name} - {eval_result}")
                    
                except Exception as e:
                    self.logger.error(f"评估任务 {task_name} 执行失败: {e}")
                    results[task_name] = {"error": str(e)}
            
            if results:
                self.logger.info(f"🎯 learner {learner_id} 评估完成，共执行 {len(results)} 个任务")
            return results
            
        except Exception as e:
            self.logger.error(f"评估learner {learner_id} 失败: {e}")
            return {}
