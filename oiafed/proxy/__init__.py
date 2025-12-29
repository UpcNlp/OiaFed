"""
Proxy 模块

提供对等节点的代理抽象
"""

from .peer import PeerProxy
from .collection import ProxyCollection

__all__ = [
    "PeerProxy",
    "ProxyCollection",
]
