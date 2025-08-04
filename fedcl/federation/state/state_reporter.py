# fedcl/federation/state/state_reporter.py
"""
状态汇报器模块

提供状态汇报功能，将状态变化信息发送到监控系统、日志系统等。
"""

import time
import json
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from loguru import logger

from ...exceptions import FedCLError


class StateReportError(FedCLError):
    """状态汇报错误"""
    pass


@dataclass
class StateReport:
    """状态汇报数据结构"""
    component_id: str
    component_type: str  # 'server', 'client', 'auxiliary'
    old_state: str
    new_state: str
    timestamp: float
    metadata: Dict[str, Any]
    duration: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)
    
    def to_json(self) -> str:
        """转换为JSON格式"""
        return json.dumps(self.to_dict(), default=str, ensure_ascii=False)


class StateReporter:
    """
    状态汇报器
    
    负责收集和发送状态变化信息到各种目标（文件、网络、监控系统等）。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化状态汇报器
        
        Args:
            config: 汇报配置
        """
        self.config = config or {}
        
        # 汇报缓存
        self.report_buffer: deque = deque(maxlen=self.config.get('buffer_size', 1000))
        self.buffer_lock = threading.RLock()
        
        # 汇报目标
        self.report_targets: List[Callable] = []
        
        # 统计信息
        self.stats = {
            'total_reports': 0,
            'successful_reports': 0,
            'failed_reports': 0,
            'last_report_time': None
        }
        
        # 汇报过滤器
        self.component_filters: Dict[str, List[str]] = {}  # component_id -> allowed_states
        self.state_filters: Dict[str, List[str]] = {}      # state -> allowed_components
        
        # 自动汇报配置
        self.auto_flush_interval = self.config.get('auto_flush_interval', 60)  # 秒
        self.auto_flush_count = self.config.get('auto_flush_count', 100)
        self._flush_timer = None
        
        # 初始化默认汇报目标
        self._init_default_targets()
        
        # 启动自动刷新
        if self.config.get('enable_auto_flush', True):
            self._start_auto_flush()
        
        logger.debug("状态汇报器初始化完成")
    
    def report_state_change(self, component_id: str, component_type: str,
                          old_state: Any, new_state: Any, 
                          metadata: Optional[Dict[str, Any]] = None,
                          duration: Optional[float] = None):
        """
        汇报状态变化
        
        Args:
            component_id: 组件ID
            component_type: 组件类型
            old_state: 旧状态
            new_state: 新状态
            metadata: 元数据
            duration: 状态持续时间
        """
        try:
            # 检查过滤器
            if not self._should_report(component_id, str(new_state)):
                return
            
            # 创建状态汇报
            report = StateReport(
                component_id=component_id,
                component_type=component_type,
                old_state=str(old_state),
                new_state=str(new_state),
                timestamp=time.time(),
                metadata=metadata or {},
                duration=duration
            )
            
            # 添加到缓存
            with self.buffer_lock:
                self.report_buffer.append(report)
                self.stats['total_reports'] += 1
            
            # 检查是否需要立即刷新
            if len(self.report_buffer) >= self.auto_flush_count:
                self._flush_reports()
            
            logger.debug(f"状态汇报添加: {component_id} {old_state} -> {new_state}")
            
        except Exception as e:
            logger.error(f"状态汇报失败: {e}")
            self.stats['failed_reports'] += 1
    
    def add_report_target(self, target: Callable[[StateReport], bool]):
        """
        添加汇报目标
        
        Args:
            target: 汇报目标函数，接收StateReport对象，返回是否成功
        """
        self.report_targets.append(target)
        logger.debug(f"添加汇报目标: {target.__name__}")
    
    def remove_report_target(self, target: Callable):
        """移除汇报目标"""
        if target in self.report_targets:
            self.report_targets.remove(target)
            logger.debug(f"移除汇报目标: {target.__name__}")
    
    def add_component_filter(self, component_id: str, allowed_states: List[str]):
        """
        添加组件过滤器
        
        Args:
            component_id: 组件ID
            allowed_states: 允许汇报的状态列表
        """
        self.component_filters[component_id] = allowed_states
        logger.debug(f"添加组件过滤器: {component_id} -> {allowed_states}")
    
    def add_state_filter(self, state: str, allowed_components: List[str]):
        """
        添加状态过滤器
        
        Args:
            state: 状态名称
            allowed_components: 允许汇报该状态的组件列表
        """
        self.state_filters[state] = allowed_components
        logger.debug(f"添加状态过滤器: {state} -> {allowed_components}")
    
    def flush_reports(self):
        """手动刷新汇报缓存"""
        self._flush_reports()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取汇报统计信息"""
        with self.buffer_lock:
            return {
                **self.stats,
                'buffer_size': len(self.report_buffer),
                'target_count': len(self.report_targets),
                'filter_count': len(self.component_filters) + len(self.state_filters)
            }
    
    def get_recent_reports(self, limit: int = 100) -> List[StateReport]:
        """获取最近的汇报记录"""
        with self.buffer_lock:
            return list(self.report_buffer)[-limit:]
    
    def close(self):
        """关闭汇报器，清理资源"""
        # 停止自动刷新
        if self._flush_timer:
            self._flush_timer.cancel()
        
        # 刷新剩余汇报
        self._flush_reports()
        
        logger.debug("状态汇报器已关闭")
    
    def _should_report(self, component_id: str, state: str) -> bool:
        """检查是否应该汇报此状态变化"""
        # 检查组件过滤器
        if component_id in self.component_filters:
            if state not in self.component_filters[component_id]:
                return False
        
        # 检查状态过滤器
        if state in self.state_filters:
            if component_id not in self.state_filters[state]:
                return False
        
        return True
    
    def _flush_reports(self):
        """刷新汇报缓存到目标"""
        if not self.report_buffer or not self.report_targets:
            return
        
        with self.buffer_lock:
            reports_to_send = list(self.report_buffer)
            self.report_buffer.clear()
        
        successful_count = 0
        failed_count = 0
        
        for report in reports_to_send:
            for target in self.report_targets:
                try:
                    if target(report):
                        successful_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    logger.error(f"汇报目标 {target.__name__} 处理失败: {e}")
                    failed_count += 1
        
        # 更新统计
        self.stats['successful_reports'] += successful_count
        self.stats['failed_reports'] += failed_count
        self.stats['last_report_time'] = time.time()
        
        if reports_to_send:
            logger.debug(f"刷新状态汇报: {len(reports_to_send)} 条记录, "
                        f"成功: {successful_count}, 失败: {failed_count}")
    
    def _start_auto_flush(self):
        """启动自动刷新定时器"""
        self._flush_reports()
        
        # 设置下一次自动刷新
        self._flush_timer = threading.Timer(self.auto_flush_interval, self._start_auto_flush)
        self._flush_timer.daemon = True
        self._flush_timer.start()
    
    def _init_default_targets(self):
        """初始化默认汇报目标"""
        # 文件汇报目标
        if self.config.get('enable_file_report', True):
            log_file = self.config.get('log_file', 'state_reports.log')
            self.add_report_target(self._create_file_target(log_file))
        
        # 控制台汇报目标
        if self.config.get('enable_console_report', False):
            self.add_report_target(self._create_console_target())
    
    def _create_file_target(self, log_file: str) -> Callable:
        """创建文件汇报目标"""
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        def file_target(report: StateReport) -> bool:
            try:
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(report.to_json() + '\n')
                return True
            except Exception as e:
                logger.error(f"文件汇报失败: {e}")
                return False
        
        return file_target
    
    def _create_console_target(self) -> Callable:
        """创建控制台汇报目标"""
        def console_target(report: StateReport) -> bool:
            try:
                print(f"[STATE REPORT] {report.component_id}: {report.old_state} -> {report.new_state}")
                return True
            except Exception as e:
                logger.error(f"控制台汇报失败: {e}")
                return False
        
        return console_target


class StateMonitor:
    """
    状态监控器
    
    提供状态监控和告警功能。
    """
    
    def __init__(self, reporter: StateReporter):
        """
        初始化状态监控器
        
        Args:
            reporter: 状态汇报器
        """
        self.reporter = reporter
        self.alert_rules: List[Dict[str, Any]] = []
        self.state_counters = defaultdict(lambda: defaultdict(int))  # component_type -> state -> count
        
    def add_alert_rule(self, rule: Dict[str, Any]):
        """
        添加告警规则
        
        Args:
            rule: 告警规则配置
                {
                    'name': '规则名称',
                    'condition': '告警条件',
                    'component_type': '组件类型',
                    'state': '状态',
                    'threshold': '阈值',
                    'action': '告警动作'
                }
        """
        self.alert_rules.append(rule)
        logger.debug(f"添加告警规则: {rule['name']}")
    
    def check_alerts(self, report: StateReport):
        """检查告警规则"""
        # 更新状态计数器
        self.state_counters[report.component_type][report.new_state] += 1
        
        # 检查告警规则
        for rule in self.alert_rules:
            if self._evaluate_rule(rule, report):
                self._trigger_alert(rule, report)
    
    def _evaluate_rule(self, rule: Dict[str, Any], report: StateReport) -> bool:
        """评估告警规则"""
        # 简单的规则评估逻辑
        if rule.get('component_type') and rule['component_type'] != report.component_type:
            return False
        
        if rule.get('state') and rule['state'] != report.new_state:
            return False
        
        if rule.get('condition') == 'error_state' and 'ERROR' not in report.new_state.upper():
            return False
        
        return True
    
    def _trigger_alert(self, rule: Dict[str, Any], report: StateReport):
        """触发告警"""
        alert_message = f"告警: {rule['name']} - {report.component_id} 进入状态 {report.new_state}"
        logger.warning(alert_message)
        
        # 可以在这里实现具体的告警动作
        # 例如发送邮件、短信、webhook等
