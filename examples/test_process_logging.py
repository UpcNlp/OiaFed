"""
测试 Process 模式下的日志合并
examples/test_process_logging.py
"""
import asyncio
import subprocess
import sys
import time
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def test_process_mode():
    """测试 Process 模式下的日志合并"""
    print("\n" + "=" * 80)
    print("测试场景: Process 模式日志合并")
    print("=" * 80)
    print("配置: configs/distributed/fmnist/")
    print("预期结果:")
    print("  - Server 日志: server_fmnist 的所有组件日志合并到 server_fmnist.log")
    print("  - Client 日志: client_0 和 client_1 各自独立的日志文件")
    print()

    # 启动服务端进程
    print("启动服务端...")
    server_process = subprocess.Popen(
        [sys.executable, "-m", "fedcl.main", "configs/distributed/fmnist/server.yaml"],
        env={"PYTHONPATH": str(project_root)},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # 等待服务端启动
    await asyncio.sleep(3)

    if server_process.poll() is not None:
        print("✗ 服务端启动失败")
        stdout, _ = server_process.communicate()
        print(stdout)
        return False

    print("✓ 服务端已启动")

    # 启动客户端进程
    print("启动客户端 0...")
    client0_process = subprocess.Popen(
        [sys.executable, "-m", "fedcl.main", "configs/distributed/fmnist/client_0.yaml"],
        env={"PYTHONPATH": str(project_root)},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    await asyncio.sleep(2)

    print("启动客户端 1...")
    client1_process = subprocess.Popen(
        [sys.executable, "-m", "fedcl.main", "configs/distributed/fmnist/client_1.yaml"],
        env={"PYTHONPATH": str(project_root)},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # 等待训练完成（3轮应该很快）
    print("\n等待训练完成（3轮）...")
    await asyncio.sleep(60)  # 给予60秒完成3轮训练

    # 检查日志文件结构
    print("\n检查日志文件结构...")

    # 查找最新的日志目录
    logs_dir = project_root / "logs"
    if not logs_dir.exists():
        print("✗ 日志目录不存在")
        success = False
    else:
        # 查找包含 fmnist 的最新运行目录
        run_dirs = sorted([d for d in logs_dir.glob("*/run_*") if d.is_dir()],
                         key=lambda x: x.stat().st_mtime, reverse=True)

        if not run_dirs:
            print("✗ 未找到日志运行目录")
            success = False
        else:
            latest_run = run_dirs[0]
            runtime_dir = latest_run / "runtime"

            print(f"日志目录: {runtime_dir}")

            if runtime_dir.exists():
                log_files = list(runtime_dir.glob("*.log"))
                print(f"\n找到 {len(log_files)} 个日志文件:")
                for log_file in sorted(log_files):
                    size = log_file.stat().st_size
                    print(f"  - {log_file.name:30s} ({size:>8,} bytes)")

                # 检查关键日志文件
                server_log = runtime_dir / "server_fmnist.log"
                client0_log = runtime_dir / "client_0.log"
                client1_log = runtime_dir / "client_1.log"

                checks = {
                    "server_fmnist.log 存在": server_log.exists(),
                    "server_fmnist.log 非空": server_log.exists() and server_log.stat().st_size > 0,
                    "client_0.log 存在": client0_log.exists(),
                    "client_0.log 非空": client0_log.exists() and client0_log.stat().st_size > 0,
                    "client_1.log 存在": client1_log.exists(),
                    "client_1.log 非空": client1_log.exists() and client1_log.stat().st_size > 0,
                }

                print("\n日志文件检查:")
                all_passed = True
                for check, passed in checks.items():
                    status = "✓" if passed else "✗"
                    print(f"  {status} {check}")
                    if not passed:
                        all_passed = False

                # 检查 server_fmnist.log 是否包含各个组件的日志
                if server_log.exists() and server_log.stat().st_size > 0:
                    print("\n分析 server_fmnist.log 内容...")
                    with open(server_log, 'r') as f:
                        content = f.read()

                    components = {
                        "ProxyManager": "proxy_manager" in content.lower() or "代理管理器" in content,
                        "HeartbeatService": "heartbeat" in content.lower() or "心跳" in content,
                        "SecurityService": "security" in content.lower() or "安全" in content,
                        "RegistryService": "registry" in content.lower() or "注册" in content,
                        "ConnectionManager": "connection_manager" in content.lower() or "连接管理" in content,
                    }

                    print("组件日志合并检查:")
                    for component, found in components.items():
                        status = "✓" if found else "✗"
                        print(f"  {status} {component}")

                success = all_passed
            else:
                print("✗ runtime/ 目录不存在")
                success = False

    # 清理进程
    print("\n停止所有进程...")
    for process in [server_process, client0_process, client1_process]:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    print("\n" + "=" * 80)
    if success:
        print("✓ 测试通过")
    else:
        print("✗ 测试失败")
    print("=" * 80)

    return success


if __name__ == "__main__":
    try:
        result = asyncio.run(test_process_mode())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\n测试被中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
