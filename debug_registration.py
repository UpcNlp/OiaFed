#!/usr/bin/env python3

import asyncio
from fedcl.communication.memory_manager import MemoryCommunicationManager
from fedcl.types import CommunicationConfig, RegistrationRequest, TransportConfig
from fedcl.transport.memory import MemoryTransport
from fedcl.utils.auto_logger import setup_auto_logging

async def test_registration():
    """测试注册功能"""
    setup_auto_logging()
    
    try:
        # 创建配置
        transport_config = TransportConfig(type="memory")
        comm_config = CommunicationConfig()
        
        # 创建transport和通信管理器
        transport = MemoryTransport(transport_config)
        comm = MemoryCommunicationManager('test_server', transport, comm_config)
        
        print("Starting communication manager...")
        await comm.start()
        
        # 创建注册请求
        reg = RegistrationRequest(
            client_id='test_client',
            client_type='learner', 
            capabilities=['train', 'evaluate']
        )
        
        print("Registering client...")
        response = await comm.register_client(reg)
        print(f"Registration response: {response}")
        
        print("Getting client info...")
        client_info = await comm.get_client_info('test_client')
        print(f"Client info: {client_info}")
        
        print("Getting client list...")
        client_list = await comm.list_clients()
        print(f"Client list: {client_list}")
        
        if response.success and client_info is not None and len(client_list) > 0:
            print("✅ Registration test PASSED")
        else:
            print("❌ Registration test FAILED")
            
    except Exception as e:
        print(f"❌ Registration test ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            await comm.stop()
        except:
            pass

if __name__ == "__main__":
    asyncio.run(test_registration())
