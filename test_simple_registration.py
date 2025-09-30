#!/usr/bin/env python3

import pytest
import asyncio
from fedcl.communication.memory_manager import MemoryCommunicationManager
from fedcl.types import CommunicationConfig, RegistrationRequest, TransportConfig, RegistrationStatus
from fedcl.transport.memory import MemoryTransport
from fedcl.utils.auto_logger import setup_auto_logging

@pytest.mark.asyncio
async def test_simple_registration():
    """最简单的注册测试"""
    setup_auto_logging()
    
    # 创建配置
    transport_config = TransportConfig(type="memory")
    comm_config = CommunicationConfig()
    
    # 创建transport和通信管理器
    transport = MemoryTransport(transport_config)
    await transport.start()
    
    comm = MemoryCommunicationManager('test_server', transport, comm_config)
    await comm.start()
    
    try:
        # 注册客户端
        reg = RegistrationRequest(
            client_id='test_client_1',
            client_type='learner', 
            capabilities=['train', 'evaluate']
        )
        
        response = await comm.register_client(reg)
        assert response.success == True
        assert response.client_id == 'test_client_1'
        
        # 获取客户端信息
        client_info = await comm.get_client_info('test_client_1')
        assert client_info is not None
        assert client_info.status == RegistrationStatus.REGISTERED
        
        # 获取客户端列表
        client_list = await comm.list_clients()
        client_ids = [client.client_id for client in client_list]
        assert 'test_client_1' in client_ids
        
        print("✅ 简单注册测试通过")
        
    finally:
        await comm.stop()
        await transport.stop()

if __name__ == "__main__":
    asyncio.run(test_simple_registration())
