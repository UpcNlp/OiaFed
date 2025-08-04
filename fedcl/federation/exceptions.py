# fedcl/federation/exceptions.py
"""
联邦学习相关异常定义
"""


class FederationError(Exception):
    """联邦学习相关异常基类"""
    pass


class ClientError(FederationError):
    """客户端相关异常"""
    pass


class ServerError(FederationError):
    """服务端相关异常"""
    pass


class CommunicationError(FederationError):
    """通信相关异常"""
    pass


class AggregationError(FederationError):
    """聚合相关异常"""
    pass


class ModelSyncError(FederationError):
    """模型同步相关异常"""
    pass
