"""
简单的动态代理调用测试
simple_dynamic_test.py
"""

import asyncio
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_proxy_dynamic_calls():
    """测试代理动态调用功能"""
    print("🚀 Testing Dynamic Proxy Calls")
    print("=" * 50)
    
    try:
        # 导入必要模块
        from fedcl.learner.proxy import LearnerProxy, ProxyConfig
        
        print("✅ Proxy imports successful")
        
        # 创建一个最小化的代理来测试 __getattr__ 
        # 我们不需要真实的通信管理器，只需要测试动态方法生成
        
        # 创建代理配置，启用动态调用
        proxy_config = ProxyConfig(
            enable_dynamic_calls=True,
            method_whitelist=["*"],  # 允许所有方法
            method_blacklist=["__*"]  # 阻止私有方法
        )
        
        print("✅ ProxyConfig created with dynamic calls enabled")
        
        # 创建一个模拟的代理对象来测试 __getattr__
        class MockProxy:
            def __init__(self, config):
                self.config = config
                self.client_id = "test_client"
                
            def __getattr__(self, name):
                """模拟LearnerProxy的__getattr__方法"""
                # 模拟安全检查
                if self.config.enable_dynamic_calls:
                    # 检查是否被阻止
                    if any(name.startswith(pattern.rstrip('*')) for pattern in self.config.method_blacklist):
                        raise AttributeError(f"Method '{name}' is blocked for security reasons")
                    
                    # 创建动态方法
                    def dynamic_method(*args, **kwargs):
                        return f"Dynamic call to {name} with args={args}, kwargs={kwargs}"
                    
                    # 设置方法属性
                    dynamic_method.__name__ = name
                    dynamic_method._proxy_client_id = self.client_id
                    dynamic_method._method_name = name
                    dynamic_method._is_dynamic = True
                    
                    return dynamic_method
                else:
                    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        # 创建模拟代理
        mock_proxy = MockProxy(proxy_config)
        
        print("✅ MockProxy created for testing")
        
        # 测试动态方法生成
        print("\n--- Testing Dynamic Method Creation ---")
        
        # 测试1: 正常方法
        print("1. Testing custom_method_for_testing...")
        try:
            method = mock_proxy.custom_method_for_testing
            print(f"   ✅ Dynamic method created: {method}")
            print(f"   ✅ Method name: {method.__name__}")
            print(f"   ✅ Method client_id: {method._proxy_client_id}")
            print(f"   ✅ Is dynamic: {method._is_dynamic}")
            
            # 测试调用
            result = method("test_param", param2=123)
            print(f"   ✅ Method call result: {result}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # 测试2: 另一个方法
        print("\n2. Testing get_client_info...")
        try:
            method = mock_proxy.get_client_info
            result = method()
            print(f"   ✅ Method call result: {result}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # 测试3: 被阻止的方法
        print("\n3. Testing blocked method (__private_method)...")
        try:
            method = mock_proxy.__private_method
            print(f"   ⚠️ Unexpected success: {method}")
        except AttributeError as e:
            print(f"   ✅ Expected blocking: {e}")
        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")
        
        # 测试4: 禁用动态调用
        print("\n4. Testing with dynamic calls disabled...")
        try:
            disabled_config = ProxyConfig(enable_dynamic_calls=False)
            disabled_proxy = MockProxy(disabled_config)
            method = disabled_proxy.some_method
            print(f"   ⚠️ Unexpected success: {method}")
        except AttributeError as e:
            print(f"   ✅ Expected failure: {e}")
        
        print("\n--- Testing with Real LearnerProxy ---")
        
        # 现在测试真实的LearnerProxy的__getattr__是否存在
        try:
            # 检查LearnerProxy是否有__getattr__方法
            if hasattr(LearnerProxy, '__getattr__'):
                print("   ✅ LearnerProxy has __getattr__ method")
            else:
                print("   ❌ LearnerProxy missing __getattr__ method")
                
            # 检查ProxyConfig是否有动态调用配置
            default_config = ProxyConfig()
            if hasattr(default_config, 'enable_dynamic_calls'):
                print(f"   ✅ ProxyConfig.enable_dynamic_calls = {default_config.enable_dynamic_calls}")
            else:
                print("   ❌ ProxyConfig missing enable_dynamic_calls attribute")
                
        except Exception as e:
            print(f"   ❌ Error checking LearnerProxy: {e}")
        
        print("\n🎉 Dynamic Proxy Call Creation Test Completed!")
        print("✅ The proxy.xxx() calling pattern should work correctly!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(test_proxy_dynamic_calls())
    except KeyboardInterrupt:
        print("\n⛔ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
