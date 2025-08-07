#!/usr/bin/env python3
"""
FedCL CLI 核心模块

提供专业的命令行工具，类似于 wandb, docker 等
"""

import os
import sys
import signal
import argparse
import time
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
from datetime import datetime

from .launcher import FedCLLauncher


class FedCLCLI:
    """FedCL 专业命令行工具"""
    
    VERSION = "1.0.0"
    
    def __init__(self):
        self.config_dir = Path.home() / ".fedcl"
        self.config_dir.mkdir(exist_ok=True)
        
        self.pid_file = self.config_dir / "fedcl.pid"
        self.status_file = self.config_dir / "status.json"
        
    def run_command(self, args) -> int:
        """执行命令"""
        try:
            if args.command == "run" or args.command == "start":
                return self._run_experiment(args)
            elif args.command == "daemon":
                return self._run_daemon(args)
            elif args.command == "stop":
                return self._stop_daemon()
            elif args.command == "status":
                return self._show_status()
            elif args.command == "logs":
                return self._show_logs(args)
            elif args.command == "clean":
                return self._clean()
            elif args.command == "init":
                return self._init_project(args)
            else:
                print(f"❌ Unknown command: {args.command}")
                return 1
                
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
            return 0
        except Exception as e:
            print(f"❌ Error: {e}")
            return 1
    
    def _run_experiment(self, args) -> int:
        """运行实验"""
        config_path = args.config
        
        if not Path(config_path).exists():
            print(f"❌ Config not found: {config_path}")
            return 1
        
        print(f"🚀 Starting FedCL experiment...")
        print(f"📋 Config: {config_path}")
        
        try:
            # 创建启动器
            launcher = FedCLLauncher(
                config_path=config_path,
                daemon=False
            )
            
            # 设置日志
            launcher.setup_logging()
            
            # 保存状态
            self._save_status({
                "status": "running",
                "config": str(Path(config_path).absolute()),
                "start_time": datetime.now().isoformat(),
                "mode": "foreground"
            })
            
            # 启动实验
            launcher.run()
            
            # 更新状态
            self._save_status({
                "status": "completed",
                "config": str(Path(config_path).absolute()),
                "end_time": datetime.now().isoformat(),
                "mode": "foreground"
            })
            
            print("✅ Experiment completed successfully!")
            return 0
            
        except Exception as e:
            self._save_status({
                "status": "failed",
                "error": str(e),
                "end_time": datetime.now().isoformat()
            })
            print(f"❌ Experiment failed: {e}")
            return 1
    
    def _run_daemon(self, args) -> int:
        """后台运行"""
        config_path = args.config
        
        if not Path(config_path).exists():
            print(f"❌ Config not found: {config_path}")
            return 1
        
        # 检查是否已有后台进程
        if self._is_daemon_running():
            print("⚠️  Daemon is already running")
            print("Use 'fedcl stop' to stop it first")
            return 1
        
        print(f"🚀 Starting FedCL daemon...")
        print(f"📋 Config: {config_path}")
        
        try:
            # 从fedcl包内部获取launcher路径
            from fedcl import cli
            cli_module_path = Path(cli.__file__).parent / "launcher.py"
            
            # 启动后台进程
            cmd = [
                sys.executable, 
                "-c",
                f"from fedcl.cli.launcher import FedCLLauncher; "
                f"launcher = FedCLLauncher('{config_path}', daemon=True); "
                f"launcher.setup_logging(); launcher.run()"
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True
            )
            
            # 等待一下确保进程启动
            time.sleep(1)
            
            if process.poll() is None:
                # 保存PID
                with open(self.pid_file, 'w') as f:
                    f.write(str(process.pid))
                
                # 保存状态
                self._save_status({
                    "status": "running",
                    "config": str(Path(config_path).absolute()),
                    "start_time": datetime.now().isoformat(),
                    "mode": "daemon",
                    "pid": process.pid
                })
                
                print(f"✅ Daemon started successfully!")
                print(f"📊 PID: {process.pid}")
                print(f"📝 Logs: logs/daemon/")
                print(f"🛑 Stop: fedcl stop")
                return 0
            else:
                print("❌ Failed to start daemon")
                return 1
                
        except Exception as e:
            print(f"❌ Failed to start daemon: {e}")
            return 1
    
    def _stop_daemon(self) -> int:
        """停止后台进程"""
        if not self._is_daemon_running():
            print("ℹ️  No daemon is running")
            return 0
        
        try:
            # 读取PID
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            print(f"🛑 Stopping daemon (PID: {pid})...")
            
            # 发送终止信号
            os.kill(pid, signal.SIGTERM)
            
            # 等待进程结束
            max_wait = 10
            for _ in range(max_wait):
                try:
                    os.kill(pid, 0)  # 检查进程是否还存在
                    time.sleep(0.5)
                except OSError:
                    break
            else:
                # 强制杀死
                print("⚠️  Process didn't stop gracefully, forcing...")
                os.kill(pid, signal.SIGKILL)
            
            # 清理文件
            self.pid_file.unlink(missing_ok=True)
            
            # 更新状态
            self._save_status({
                "status": "stopped",
                "stop_time": datetime.now().isoformat()
            })
            
            print("✅ Daemon stopped successfully!")
            return 0
            
        except Exception as e:
            print(f"❌ Failed to stop daemon: {e}")
            return 1
    
    def _show_status(self) -> int:
        """显示状态"""
        print("📊 FedCL Status")
        print("=" * 50)
        
        # 检查后台进程
        daemon_running = self._is_daemon_running()
        daemon_status = "🟢 Running" if daemon_running else "🔴 Stopped"
        print(f"Daemon: {daemon_status}")
        
        # 显示详细状态
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    status = json.load(f)
                
                print(f"Status: {status.get('status', 'unknown')}")
                if 'config' in status:
                    print(f"Config: {status['config']}")
                if 'start_time' in status:
                    print(f"Started: {status['start_time']}")
                if 'mode' in status:
                    print(f"Mode: {status['mode']}")
                if 'pid' in status and daemon_running:
                    print(f"PID: {status['pid']}")
                
            except Exception as e:
                print(f"⚠️  Failed to read status: {e}")
        
        # 显示日志目录
        log_dirs = []
        if Path("logs").exists():
            log_dirs.append("logs/")
        if Path("experiments").exists():
            exp_count = len(list(Path("experiments").iterdir()))
            log_dirs.append(f"experiments/ ({exp_count} experiments)")
        
        if log_dirs:
            print(f"Logs: {', '.join(log_dirs)}")
        
        return 0
    
    def _show_logs(self, args) -> int:
        """显示日志"""
        log_dir = Path("logs")
        
        if not log_dir.exists():
            print("📝 No logs found")
            return 0
        
        # 寻找最新的日志文件
        log_files = []
        
        # 守护进程日志
        daemon_logs = log_dir / "daemon"
        if daemon_logs.exists():
            stdout_log = daemon_logs / "stdout.log"
            stderr_log = daemon_logs / "stderr.log"
            if stdout_log.exists():
                log_files.append(("Daemon Output", stdout_log))
            if stderr_log.exists():
                log_files.append(("Daemon Error", stderr_log))
        
        # 实验日志
        for log_file in log_dir.glob("fedcl_*.log"):
            log_files.append(("Experiment", log_file))
        
        if not log_files:
            print("📝 No log files found")
            return 0
        
        # 显示可用日志
        print("📝 Available logs:")
        for i, (name, path) in enumerate(log_files, 1):
            size = path.stat().st_size if path.exists() else 0
            print(f"  {i}. {name}: {path.name} ({size} bytes)")
        
        # 如果指定了follow，监视最新的日志
        if getattr(args, 'follow', False):
            if log_files:
                latest_log = max(log_files, key=lambda x: x[1].stat().st_mtime)[1]
                print(f"\n👁️  Following: {latest_log}")
                try:
                    subprocess.run(["tail", "-f", str(latest_log)])
                except KeyboardInterrupt:
                    print("\n🛑 Stopped following logs")
        
        return 0
    
    def _clean(self) -> int:
        """清理临时文件"""
        print("🧹 Cleaning FedCL temporary files...")
        
        cleaned = []
        
        # 清理PID文件
        if self.pid_file.exists():
            self.pid_file.unlink()
            cleaned.append("PID file")
        
        # 清理状态文件
        if self.status_file.exists():
            self.status_file.unlink()
            cleaned.append("Status file")
        
        # 清理日志文件（询问用户）
        log_dir = Path("logs")
        if log_dir.exists():
            response = input("🗑️  Delete log files? (y/N): ")
            if response.lower() == 'y':
                shutil.rmtree(log_dir)
                cleaned.append("Log directory")
        
        # 清理实验文件（询问用户）
        exp_dir = Path("experiments")
        if exp_dir.exists():
            response = input("🗑️  Delete experiment results? (y/N): ")
            if response.lower() == 'y':
                shutil.rmtree(exp_dir)
                cleaned.append("Experiment directory")
        
        if cleaned:
            print(f"✅ Cleaned: {', '.join(cleaned)}")
        else:
            print("ℹ️  Nothing to clean")
        
        return 0
    
    def _init_project(self, args) -> int:
        """初始化新项目"""
        project_name = args.name
        project_dir = Path(project_name)
        
        if project_dir.exists():
            print(f"❌ Directory already exists: {project_name}")
            return 1
        
        print(f"🎯 Initializing FedCL project: {project_name}")
        
        try:
            # 创建项目目录
            project_dir.mkdir(parents=True)
            
            # 创建基本结构
            (project_dir / "configs").mkdir()
            (project_dir / "data").mkdir()
            (project_dir / "logs").mkdir()
            (project_dir / "experiments").mkdir()
            
            # 复制配置模板
            template_dir = Path("examples/config_templates/server_client_configs")
            if template_dir.exists():
                shutil.copytree(template_dir, project_dir / "configs" / "federated")
                print("✅ Config templates copied")
            
            # 创建README
            readme_content = f"""# {project_name}

FedCL Federated Learning Project

## Quick Start

```bash
# Run experiment
fedcl run configs/federated

# Run in background
fedcl daemon configs/federated

# Check status
fedcl status

# View logs
fedcl logs --follow
```

## Directory Structure

- `configs/`: Configuration files
- `data/`: Dataset files
- `logs/`: Log files
- `experiments/`: Experiment results
"""
            (project_dir / "README.md").write_text(readme_content)
            
            print(f"✅ Project initialized: {project_name}")
            print(f"📁 Directory: {project_dir.absolute()}")
            print(f"🚀 Get started: cd {project_name} && fedcl run configs/federated")
            
            return 0
            
        except Exception as e:
            print(f"❌ Failed to initialize project: {e}")
            return 1
    
    def _is_daemon_running(self) -> bool:
        """检查后台进程是否运行"""
        if not self.pid_file.exists():
            return False
        
        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            # 检查进程是否存在
            os.kill(pid, 0)
            return True
            
        except (OSError, ValueError):
            # 进程不存在或PID文件损坏
            self.pid_file.unlink(missing_ok=True)
            return False
    
    def _save_status(self, status: dict):
        """保存状态"""
        try:
            with open(self.status_file, 'w') as f:
                json.dump(status, f, indent=2)
        except Exception:
            pass  # 忽略状态保存错误


def create_parser():
    """创建命令行解析器"""
    parser = argparse.ArgumentParser(
        prog="fedcl",
        description="FedCL - Federated Continual Learning Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  fedcl run config.yaml                # Run experiment
  fedcl daemon configs/                # Run in background
  fedcl status                         # Check status
  fedcl logs --follow                  # Follow logs
  fedcl stop                           # Stop daemon
  fedcl clean                          # Clean temporary files
  fedcl init my_project                # Initialize new project
        """
    )
    
    # 版本信息
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"FedCL {FedCLCLI.VERSION}"
    )
    
    # 子命令
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # run 命令
    run_parser = subparsers.add_parser("run", help="Run experiment")
    run_parser.add_argument("config", help="Config file or directory")
    run_parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    # start 命令（run的别名）
    start_parser = subparsers.add_parser("start", help="Start experiment (alias for run)")
    start_parser.add_argument("config", help="Config file or directory")
    start_parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    # daemon 命令
    daemon_parser = subparsers.add_parser("daemon", help="Run in background")
    daemon_parser.add_argument("config", help="Config file or directory")
    
    # stop 命令
    subparsers.add_parser("stop", help="Stop background process")
    
    # status 命令
    subparsers.add_parser("status", help="Show status")
    
    # logs 命令
    logs_parser = subparsers.add_parser("logs", help="Show logs")
    logs_parser.add_argument("--follow", "-f", action="store_true", help="Follow log output")
    
    # clean 命令
    subparsers.add_parser("clean", help="Clean temporary files")
    
    # init 命令
    init_parser = subparsers.add_parser("init", help="Initialize new project")
    init_parser.add_argument("name", help="Project name")
    
    return parser


def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    cli = FedCLCLI()
    return cli.run_command(args)


if __name__ == "__main__":
    sys.exit(main())
