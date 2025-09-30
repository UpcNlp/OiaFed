
import asyncio
import inspect
from typing import Any, Dict

from ...fl.client import BaseFLNode
from ...api.decorators import learner

@learner('SimpleLearner')
class SimpleLearnerStub(BaseFLNode):
    """
    简单学习器存根实现
    演示如何使用BaseFLNode构建自己的学习器
    """
    
    def __init__(self, learner_id: str = None, transport=None, node_id: str = None, **kwargs):
        # 处理参数兼容性，node_id 和 learner_id 都可以使用
        if node_id and not learner_id:
            learner_id = node_id
        elif not learner_id:
            learner_id = "simple_learner"
            
        super().__init__(learner_id, transport)
        self.model = {'param1': 1.0, 'param2': 2.0}
        self.training_data = None
        
        # 添加client_id属性以兼容客户端代码
        self.client_id = learner_id
        
        # 注册RPC方法处理器
        self._rpc_handlers = {
            'train': self.train,
            'get_model': self.get_model,
            'set_model': self.set_model,
            'ping': self.ping,
            'get_status': self.get_status,
            '__get_methods__': self.get_methods
        }
        
    async def on_push_received(self, data: Any):
        """接收全局模型推送"""
        if isinstance(data, dict) and 'global_model' in data:
            self.model = data['global_model']
            self.logger.info("接收到全局模型更新")
            
    async def train(self, data: Any = None) -> Dict[str, Any]:
        """训练方法 - 用户业务逻辑"""
        self.logger.info("开始本地训练")
        
        # 模拟训练过程
        await asyncio.sleep(0.1)
        
        # 模拟参数更新
        for key in self.model:
            if isinstance(self.model[key], (int, float)):
                self.model[key] += 0.01
                
        return {
            'model': self.model.copy(),
            'metrics': {'loss': 0.1, 'accuracy': 0.95},
            'samples': 100
        }
        
    def get_model(self) -> Dict:
        """获取模型"""
        return self.model.copy()
        
    def set_model(self, model: Dict):
        """设置模型"""
        self.model = model
        
    async def ping(self) -> str:
        """健康检查"""
        return "pong"
        
    def get_status(self) -> Dict:
        """获取状态"""
        return {
            'learner_id': self.node_id,
            'connected': self.connected,
            'registered': self.registered,
            'model_params': len(self.model) if self.model else 0
        }
        
    def get_methods(self) -> Dict[str, Dict]:
        """返回可调用方法信息"""
        methods = {}
        for name, method in self._rpc_handlers.items():
            if not name.startswith('_'):
                sig = inspect.signature(method)
                methods[name] = {
                    'params': [p.name for p in sig.parameters.values()],
                    'is_async': asyncio.iscoroutinefunction(method)
                }
        return methods
        
    async def handle_rpc(self, request: Dict) -> Dict:
        """处理RPC请求"""
        method_name = request.get('method')
        args = request.get('args', ())
        kwargs = request.get('kwargs', {})
        
        try:
            handler = self._rpc_handlers.get(method_name)
            if not handler:
                raise ValueError(f"未知方法: {method_name}")
                
            if asyncio.iscoroutinefunction(handler):
                result = await handler(*args, **kwargs)
            else:
                result = handler(*args, **kwargs)
                
            return {'result': result}
            
        except Exception as e:
            return {'error': str(e)}
    
    async def train_task(self,*args,**kwargs):
        pass

    async def evaluate_task(self,*args,**kwargs):
        pass
    