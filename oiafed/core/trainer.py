"""
Trainer（服务端训练器）抽象基类
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
import asyncio

from .types import TrainResult, EvalResult, ClientUpdate, RoundResult, RoundMetrics
from ..infra.logging import get_logger

if TYPE_CHECKING:
    from .model import Model
    from .aggregator import Aggregator
    from ..proxy.collection import ProxyCollection
    from ..infra.tracker import Tracker


class Trainer(ABC):
    """
    服务端训练器抽象基类

    职责：
    - 编排训练流程
    - 管理客户端（通过代理）
    - 协调聚合过程
    - 控制训练生命周期

    用户需要实现 train_round() 方法来定义单轮训练逻辑

    Callback 触发点：
    - on_train_begin: run() 开始时
    - on_train_end: run() 结束时
    - on_round_begin: train_round() 开始时
    - on_round_end: train_round() 结束时
    """

    def __init__(
        self,
        # ========== 核心依赖（必需）==========
        learners: "ProxyCollection",     # 学习器代理集合
        aggregator: "Aggregator",        # 聚合器

        # ========== 可选依赖 ==========
        dataset: Optional[Any] = None,           # 服务端数据集（用于评估，向后兼容）
        datasets: Optional[Dict[str, List[Any]]] = None,  # 数据集字典（按 split 分组）
        model: Optional["Model"] = None,         # 全局模型
        tracker: Optional["Tracker"] = None,     # 指标追踪器（所有记录通过它）
        callbacks: Optional[Any] = None,         # 回调管理器

        # ========== 配置 ==========
        config: Optional[Dict[str, Any]] = None, # 训练器配置
        node_id: Optional[str] = None,           # 节点ID（用于日志）
    ):
        """
        初始化训练器

        Args:
            learners: 学习器代理集合（ProxyCollection）
            aggregator: 聚合器
            dataset: 服务端数据集（可选，用于评估，向后兼容）
            datasets: 数据集字典，按 split 分组（可选），如 {"train": [...], "test": [...]}
            model: 全局模型（可选）
            tracker: 指标追踪器（可选，所有记录通过它）
            callbacks: 回调管理器（可选）
            config: 训练器配置
            node_id: 节点ID（用于日志）
        """
        self._learners = learners
        self._aggregator = aggregator
        self._dataset = dataset
        self._datasets = datasets or {}
        self._model = model
        self._tracker = tracker
        self._callbacks = callbacks
        self._config = config or {}
        self._node_id = node_id or "trainer"

        self.logger = get_logger(self._node_id, "training")

        # 内部状态
        self._current_round = 0
        self._is_running = False
    
    # ==================== 属性访问 ====================

    @property
    def learners(self) -> "ProxyCollection":
        """获取学习器代理集合"""
        return self._learners

    @property
    def aggregator(self) -> "Aggregator":
        """获取聚合器"""
        return self._aggregator

    @property
    def dataset(self) -> Optional[Any]:
        """获取服务端数据集"""
        return self._dataset

    @property
    def model(self) -> Optional["Model"]:
        """获取全局模型"""
        return self._model

    @property
    def tracker(self) -> Optional["Tracker"]:
        """获取追踪器"""
        return self._tracker

    @property
    def callbacks(self) -> Optional[Any]:
        """获取回调管理器"""
        return self._callbacks

    @property
    def config(self) -> Dict[str, Any]:
        """获取配置"""
        return self._config

    @property
    def current_round(self) -> int:
        """获取当前轮次"""
        return self._current_round
    
    # ==================== 核心方法（分层训练） ====================

    async def run(self) -> Dict[str, Any]:
        """
        训练会话层 - 管理整个联邦学习训练流程

        职责：
        1. 触发训练开始回调
        2. 初始化训练环境（广播初始权重）
        3. 循环执行多轮训练（调用 train_round）
        4. 触发训练结束回调

        Returns:
            训练结果字典，包含轮次结果和汇总指标

        用户可重写此方法实现自定义训练流程，或重写 train_round 实现自定义轮次逻辑
        """
        import time

        config = self._config
        max_rounds = config.get("max_rounds", 100)

        # 触发训练开始回调
        if self._callbacks:
            await self._callbacks.on_train_begin(self, {"max_rounds": max_rounds})

        # 同步训练信息（Tracker run_id 等）到所有 Learner
        await self._sync_training_info()

        # 获取在线学习器
        learners = self.get_connected_learners()
        self.logger.info(f"开始训练，客户端数量: {len(learners)}")

        # *** 新增：输出客户端状态 ***
        self._print_client_status()

        # 广播初始权重（如果有模型）
        if self._model:
            initial_weights = self._model.get_weights()
            await self.broadcast_to_learners("set_weights", initial_weights)

        # 训练循环
        round_results = []
        start_time = time.time()

        for round_num in range(1, max_rounds + 1):
            self._current_round = round_num

            # 计算进度信息
            progress_pct = round_num / max_rounds * 100
            elapsed = time.time() - start_time

            # 计算 ETA（从第2轮开始才有意义）
            if round_num > 1:
                avg_time_per_round = elapsed / (round_num - 1)
                eta_seconds = avg_time_per_round * (max_rounds - round_num + 1)
                eta_str = self._format_time(eta_seconds)
            else:
                eta_str = "估算中..."

            # 显示进度信息
            self.logger.info(
                f"{'='*70}\n"
                f"轮次 {round_num}/{max_rounds} ({progress_pct:.1f}%) | "
                f"已用时: {self._format_time(elapsed)} | 预计剩余: {eta_str}\n"
                f"{'='*70}"
            )

            # 执行单轮训练（内部会触发 on_round_begin/end）
            round_result = await self.train_round(round_num)
            round_results.append(round_result)

            # 记录轮次指标到 tracker
            # 注意：只记录评估指标（eval_*），不记录训练指标（avg_accuracy等）
            if self.tracker:
                # 过滤掉训练指标，只保留评估指标
                metrics_to_log = {
                    k: v for k, v in round_result.to_dict().items()
                    if k.startswith('eval_') or k in ['round_num', 'num_clients', 'total_samples']
                }
                if metrics_to_log:  # 只有当有评估指标时才记录
                    self.tracker.log_metrics(metrics_to_log, step=round_num)

        self.logger.info("训练完成")

        # 触发训练结束回调
        if self._callbacks:
            await self._callbacks.on_train_end(self, {"rounds_completed": max_rounds})

        # 检查配置是否需要关闭学习器
        shutdown_learners = config.get("shutdown_learners", True)  # 默认关闭学习器
        if shutdown_learners:
            self.logger.debug("发送关闭信号到所有学习器")
            try:
                # 1. 先停止心跳任务（避免在 learners 关闭后继续发送心跳）
                self._stop_heartbeat_if_needed()

                # 2. 广播关闭信号，并收集响应
                results = await self.broadcast_to_learners("_fed_shutdown", {"reason": "training_completed"})
                self.logger.debug("关闭信号已发送")

                # 3. 主动标记已响应的 learners 为断开（避免等待 gRPC channel 检测）
                self._mark_learners_disconnecting(results)

                # 4. 等待 learners 断开连接（检测连接状态，而不是固定等待时间）
                await self._wait_for_learners_disconnect(timeout=config.get("shutdown_wait_time", 10.0))
                self.logger.debug("Learners 已断开连接")
            except Exception as e:
                self.logger.warning(f"发送关闭信号失败: {e}")

        # 返回训练结果汇总
        return {
            "total_rounds": max_rounds,
            "completed_rounds": len(round_results),
            "final_round_metrics": round_results[-1].metrics.metrics if round_results else {},
        }

    @abstractmethod
    async def train_round(self, round_num: int) -> RoundResult:
        """
        轮次层 - 处理单个联邦学习轮次

        职责：
        1. 选择客户端
        2. 收集训练更新
        3. 聚合权重
        4. 广播新权重
        5. 评估（可选）
        6. 返回轮次结果

        Args:
            round_num: 轮次编号（从1开始）

        Returns:
            RoundResult: 包含更新、权重、指标

        典型实现（FedAvg）：
        ```python
        async def train_round(self, round_num: int) -> RoundResult:
            # 1. 选择客户端
            selected = self.select_learners(num_clients)

            # 2. 收集更新
            results = await self.collect_results(selected, "fit", fit_config)
            updates = [ClientUpdate.from_result(l._target_id, r)
                       for l, r in zip(selected, results) if isinstance(r, TrainResult)]

            # 3. 聚合
            new_weights = self.aggregator.aggregate(updates, self.model)
            self.model.set_weights(new_weights)

            # 4. 广播
            await self.broadcast_to_learners("set_weights", new_weights)

            # 5. 计算指标
            metrics = RoundMetrics(...)

            return RoundResult(
                round_num=round_num,
                updates=updates,
                aggregated_weights=new_weights,
                metrics=metrics
            )
        ```

        扩展点：
        - FedAvg: 标准同步聚合
        - FedProx: 添加 proximal term 到 fit_config
        - 个性化联邦学习: 只聚合部分层
        - 异步联邦学习: 不等待所有客户端
        """
        pass

    # ==================== 框架提供的方法（无需覆盖） ====================

    def get_connected_learners(self) -> List[Any]:
        """
        获取所有在线学习器（动态更新）

        使用 ProxyCollection 的动态状态，只返回当前可用的代理

        Returns:
            可用的学习器代理列表
        """
        return self._learners.get_available_proxies()

    def select_learners(
        self,
        n: int,
        strategy: str = "random",
        require_healthy: bool = False,
        **kwargs
    ) -> List[Any]:
        """
        选择学习器子集

        Args:
            n: 选择数量
            strategy: 选择策略（random等）
            require_healthy: 是否只选择健康的学习器（通过 health 状态过滤）
            **kwargs: 策略特定参数

        Returns:
            选中的学习器代理列表

        改进：
        - 支持健康检查：require_healthy=True 时只选择健康节点
        - 使用动态可用状态：只从当前在线的 learner 中选择
        """
        import random

        # 获取候选学习器（只选择可用的）
        if require_healthy:
            all_learners = self.get_healthy_learners()
            if len(all_learners) < n:
                self.logger.warning(
                    f"警告: 只有 {len(all_learners)} 个健康learner，少于请求的 {n} 个",
                    level="warning"
                )
        else:
            all_learners = self._learners.get_available_proxies()

        if n >= len(all_learners):
            return all_learners
        return random.sample(all_learners, n)

    async def broadcast_to_learners(
        self,
        method: str,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        广播调用所有学习器

        Args:
            method: 方法名
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            结果字典 {learner_id: result}
        """
        return await self._learners.broadcast(method, *args, **kwargs)

    async def broadcast_to_selected(
        self,
        learners: List[Any],
        method: str,
        *args,
        **kwargs
    ) -> List[Any]:
        """
        向指定的学习器列表广播调用

        Args:
            learners: 目标学习器列表（Proxy 对象）
            method: 方法名
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            结果列表（与 learners 顺序对应）
        """
        if not learners:
            return []

        tasks = []
        for learner in learners:
            task = getattr(learner, method)(*args, **kwargs)
            tasks.append(asyncio.create_task(task))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return list(results)

    async def collect_results(
        self,
        learners: List[Any],
        method: str,
        *args,
        timeout: Optional[float] = None,
        min_responses: Optional[int] = None,
        **kwargs
    ) -> List[Union[TrainResult, EvalResult, Exception]]:
        """
        收集学习器结果（支持超时和部分响应）

        Args:
            learners: 学习器列表
            method: 方法名（fit / evaluate）
            *args: 位置参数
            timeout: 整体超时时间（秒），None 表示无限等待
            min_responses: 最少需要的成功响应数，达到后可以提前返回
            **kwargs: 关键字参数

        Returns:
            结果列表（成功返回 Result，失败返回 Exception或TimeoutError）

        改进：
        - 支持整体超时，避免无限等待
        - 支持最小响应数，不需要等所有learner
        - 详细的失败日志
        """
        if not learners:
            return []

        # DEBUG: 记录调用详情
        learner_ids = [getattr(l, '_target_id', 'unknown') for l in learners]
        self.logger.debug(f"collect_results: method={method}, targets={learner_ids}, args={args}, kwargs={kwargs}")

        # 创建任务列表
        tasks = []
        learner_ids = []
        for learner in learners:
            # 动态调用方法
            task = getattr(learner, method)(*args, **kwargs)
            tasks.append(asyncio.create_task(task))
            # 获取learner的ID（假设有_target_id属性）
            learner_id = getattr(learner, '_target_id', 'unknown')
            learner_ids.append(learner_id)
            self.logger.debug(f"collect_results: 创建任务, {learner_id}.{method}()")

        # 如果设置了最小响应数和超时，使用 asyncio.wait()
        if min_responses is not None or timeout is not None:
            return await self._collect_with_flexibility(
                tasks, learner_ids, min_responses, timeout
            )
        else:
            # 使用原始的 gather 方式（向后兼容）
            results = await asyncio.gather(*tasks, return_exceptions=True)
            self._log_collection_summary(learner_ids, results, method)
            return list(results)

    async def _collect_with_flexibility(
        self,
        tasks: List[asyncio.Task],
        learner_ids: List[str],
        min_responses: Optional[int],
        timeout: Optional[float]
    ) -> List[Union[TrainResult, EvalResult, Exception]]:
        """
        灵活的结果收集（支持超时和最小响应数）

        策略：
        1. 如果设置了 min_responses，达到数量后可以提前返回
        2. 如果设置了 timeout，超时后取消剩余任务
        3. 取消的任务返回 TimeoutError
        """
        num_learners = len(tasks)
        min_responses = min_responses or num_learners  # 默认等待所有响应
        results = [None] * num_learners

        try:
            # 等待任务完成
            done, pending = await asyncio.wait(
                tasks,
                timeout=timeout,
                return_when=asyncio.FIRST_EXCEPTION  # 遇到第一个异常时继续等待
            )

            # 收集已完成任务的结果
            success_count = 0
            for task in done:
                idx = tasks.index(task)
                try:
                    result = task.result()
                    results[idx] = result
                    if not isinstance(result, Exception):
                        success_count += 1
                except Exception as e:
                    results[idx] = e
                    self.logger.exception(f"Learner {learner_ids[idx]} 失败: {e}")

            # 检查是否达到最小响应数
            if success_count >= min_responses:
                self.logger.info(f"已收集到足够响应: {success_count}/{min_responses}")
            else:
                self.logger.warning(
                    f"警告: 只收集到 {success_count}/{min_responses} 个响应",
                )

            # 取消未完成的任务
            if pending:
                cancelled_ids = [learner_ids[tasks.index(t)] for t in pending]
                self.logger.warning(f"取消 {len(pending)} 个超时任务: {cancelled_ids}")
                for task in pending:
                    task.cancel()
                    idx = tasks.index(task)
                    results[idx] = TimeoutError(f"Learner {learner_ids[idx]} 超时")

        except Exception as e:
            self.logger.exception(f"收集结果时发生错误: {e}", level="error")
            # 取消所有未完成任务
            for task in tasks:
                if not task.done():
                    task.cancel()

        return results

    def _log_collection_summary(
        self,
        learner_ids: List[str],
        results: List[Any],
        method: str
    ) -> None:
        """记录收集结果摘要"""
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        failed_count = len(results) - success_count

        if failed_count > 0:
            failed_ids = [
                learner_ids[i] for i, r in enumerate(results)
                if isinstance(r, Exception)
            ]
            self.logger.warning(
                f"{method} 完成: 成功 {success_count}/{len(results)}, "
                f"失败: {failed_ids}"
            )
        else:
            self.logger.info(f"{method} 完成: 全部成功 ({success_count}/{len(results)})")

    # ==================== 健康管理辅助方法 ====================

    def get_healthy_learners(self) -> List[Any]:
        """
        获取所有健康的学习器

        通过 ProxyCollection.get_healthy_proxies() 实现，
        内部查询每个 Proxy 的 health 属性（来自 Node）

        Returns:
            健康的学习器代理列表
        """
        return self._learners.get_healthy_proxies()

    async def get_learner_states(self) -> Dict[str, str]:
        """
        获取所有学习器的状态（并发查询）

        通过 ProxyCollection.get_all_states() 实现，
        每个 Proxy 内部调用远程的 state 属性

        Returns:
            {learner_id: state, ...}
        """
        return await self._learners.get_all_states()

    def get_learners_by_state(self, state: str) -> List[Any]:
        """
        按状态过滤学习器

        Args:
            state: 目标状态（idle / training / evaluating）

        Returns:
            处于该状态的学习器列表
        """
        return self._learners.get_proxies_by_state(state)

    # ==================== 辅助方法 ====================

    def _print_client_status(self):
        """
        打印客户端状态概览

        在训练开始时调用，输出所有客户端的连接和健康状态
        """
        # 获取统计信息
        stats = self._learners.get_stats()
        total_count = stats["total"]
        available_count = stats["available"]
        unavailable_count = stats["unavailable"]

        if total_count == 0:
            self.logger.warning("没有客户端！")
            return

        # 获取所有代理（包括不可用的）
        all_learners = self._learners.get_all_proxies()
        available_ids = set(stats["available_ids"])

        # 尝试获取健康状态
        try:
            healthy_learners = self.get_healthy_learners()
            healthy_ids = set(getattr(l, '_target_id', 'unknown') for l in healthy_learners)
            num_healthy = len(healthy_learners)
        except Exception:
            # 如果不支持健康检查，假设所有可用的都是健康的
            healthy_ids = available_ids
            num_healthy = available_count

        # 计算百分比
        available_pct = (available_count / total_count * 100) if total_count > 0 else 0
        health_pct = (num_healthy / total_count * 100) if total_count > 0 else 0

        # 构建输出
        separator = "=" * 70
        status_lines = [
            "",
            separator,
            "客户端状态概览",
            separator,
            f"总客户端数: {total_count}",
            f"在线客户端: {available_count} ({available_pct:.0f}%)",
            f"离线客户端: {unavailable_count}",
            f"健康客户端: {num_healthy} ({health_pct:.0f}%)",
            "",
        ]

        # 如果客户端数量不多（<=20），列出详细信息
        if total_count <= 20:
            status_lines.append("客户端列表:")

            for learner in all_learners:
                learner_id = getattr(learner, '_target_id', 'unknown')
                is_available = learner_id in available_ids
                is_healthy = learner_id in healthy_ids

                # 状态图标
                if is_available and is_healthy:
                    status_icon = "✓"
                    status_str = "ONLINE/HEALTHY"
                elif is_available:
                    status_icon = "⚠"
                    status_str = "ONLINE/UNHEALTHY"
                else:
                    status_icon = "✗"
                    status_str = "OFFLINE"

                status_lines.append(f"  {status_icon} {learner_id} [{status_str}]")
        else:
            # 客户端太多，只显示统计信息
            status_lines.append(f"（共 {total_count} 个客户端，详细列表已省略）")

        status_lines.append(separator)
        status_lines.append("")

        # 输出状态信息
        status_msg = "\n".join(status_lines)
        self.logger.info(status_msg)

        # 警告信息
        if unavailable_count > 0:
            self.logger.warning(
                f"检测到 {unavailable_count} 个离线客户端，它们不会参与训练"
            )

        if num_healthy < available_count:
            unhealthy_count = available_count - num_healthy
            self.logger.warning(
                f"检测到 {unhealthy_count} 个不健康的在线客户端，可能影响训练质量"
            )

    def _format_time(self, seconds: float) -> str:
        """
        格式化时间显示

        Args:
            seconds: 秒数

        Returns:
            格式化的时间字符串（如 "1h 23m 45s" 或 "5m 30s"）
        """
        if seconds < 60:
            return f"{int(seconds)}秒"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            secs = int(seconds % 60)
            return f"{minutes}分{secs}秒"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}小时{minutes}分"

    def _mark_learners_disconnecting(self, results: Dict[str, Any]):
        """
        主动标记已响应 shutdown 的 learners 为断开状态

        Args:
            results: broadcast_to_learners 返回的结果字典 {learner_id: result}

        此方法直接修改 node_comm 的连接表，避免等待 gRPC channel 检测断开
        """
        # 从 ProxyCollection 获取 node
        node = self._learners._node if hasattr(self._learners, '_node') else None
        if not node:
            self.logger.debug("无法获取 node 引用，跳过标记断开")
            return

        # 获取 node_comm 实例
        comm = node._comm if hasattr(node, '_comm') else None
        if not comm:
            self.logger.debug("无法获取 comm 引用，跳过标记断开")
            return

        # 获取连接表
        connection_table = comm._connection_table if hasattr(comm, '_connection_table') else None
        if connection_table is None:
            self.logger.debug("无法获取连接表，跳过标记断开")
            return

        # 导入 ConnectionStatus
        from ..comm.message import ConnectionStatus

        # 统计成功响应的 learners
        success_count = 0
        for learner_id, result in results.items():
            # 检查是否成功响应（不是异常）
            if not isinstance(result, Exception):
                # 检查连接表中是否有此 learner
                if learner_id in connection_table:
                    # 标记为断开
                    connection_table[learner_id].status = ConnectionStatus.DISCONNECTED
                    success_count += 1
                    self.logger.debug(f"已标记 {learner_id} 为断开状态")

        if success_count > 0:
            self.logger.debug(f"已主动标记 {success_count} 个 learners 为断开状态")

    def _stop_heartbeat_if_needed(self):
        """
        停止心跳任务（如果使用 gRPC 传输）

        在关闭 learners 之前调用，避免在 learners 关闭后继续发送心跳导致 gRPC 错误
        """
        # 从 ProxyCollection 获取 node
        node = self._learners._node if hasattr(self._learners, '_node') else None
        if not node:
            self.logger.debug("无法获取 node 引用，跳过停止心跳")
            return

        # 获取 transport 层
        comm = node._comm if hasattr(node, '_comm') else None
        if not comm:
            self.logger.debug("无法获取 comm 引用，跳过停止心跳")
            return

        transport = comm._transport if hasattr(comm, '_transport') else None
        if not transport:
            self.logger.debug("无法获取 transport 引用，跳过停止心跳")
            return

        # 检查是否是 gRPC 传输且有控制线程
        if hasattr(transport, '_control_running') and transport._control_running:
            self.logger.debug("停止 gRPC 心跳任务")
            transport._stop_control_thread()
            self.logger.debug("gRPC 心跳任务已停止")

    async def _wait_for_learners_disconnect(self, timeout: float = 10.0):
        """
        等待所有 learners 断开连接

        Args:
            timeout: 超时时间（秒）

        此方法会轮询检查连接状态，直到所有 learners 断开或超时
        """
        import time
        start_time = time.time()
        check_interval = 0.1  # 100ms 检查一次

        # 从 ProxyCollection 获取 node
        node = self._learners._node if hasattr(self._learners, '_node') else None
        if not node:
            self.logger.warning("无法获取 node 引用，跳过等待 learners 断开")
            return

        learner_ids = [proxy._target_id for proxy in self._learners]

        while time.time() - start_time < timeout:
            # 获取当前连接的节点
            connected_nodes = node.get_connected_nodes()

            # 检查是否还有 learners 连接
            still_connected = [lid for lid in learner_ids if lid in connected_nodes]

            if not still_connected:
                # 所有 learners 都断开了
                return

            # 等待一小段时间再检查
            await asyncio.sleep(check_interval)

        # 超时，记录调试信息（这在 gRPC 中是正常的，因为 channel 断开检测有延迟）
        connected_nodes = node.get_connected_nodes()
        still_connected = [lid for lid in learner_ids if lid in connected_nodes]
        if still_connected:
            self.logger.debug(
                f"等待 learners 断开超时（{timeout}秒），"
                f"仍有 {len(still_connected)} 个 learners 连接: {still_connected}"
            )

    async def _sync_training_info(self):
        """
        同步训练信息到所有 Learner（Trainer端）

        遍历所有 SyncCallback，收集同步信息并广播
        """
        if not self._callbacks:
            return

        from ..callback.sync_callback import SyncCallback

        sync_info = {}

        # 收集所有 SyncCallback 的同步信息
        for callback in self._callbacks._callbacks:
            if isinstance(callback, SyncCallback):
                try:
                    info = await callback.collect_sync_info()
                    sync_info.update(info)
                except Exception as e:
                    self.logger.error(f"Failed to collect sync info from {callback}: {e}")

        # 如果有同步信息，广播给所有 Learner
        if sync_info:
            self.logger.info(f"Broadcasting training sync info: {list(sync_info.keys())}")
            try:
                await self.broadcast_to_learners("sync_training_info", sync_info)
                self.logger.info("Training sync info broadcasted successfully")
            except Exception as e:
                self.logger.error(f"Failed to broadcast sync info: {e}")

    def increment_round(self) -> int:
        """增加轮次计数"""
        self._current_round += 1
        return self._current_round