#!/usr/bin/env python3
"""
Ubuntu 服务器自动化测试脚本

"""

import os
import pwd
import sys
import time
import subprocess
import json
import argparse
import shutil
import signal
import re
import logging
import tempfile
import threading
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict

# ============================
# 依赖检查 & 安装提示
# ============================
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ============================
# 日志配置
# ============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stress-test")

# ============================
# 全局配置
# ============================

@dataclass
class Config:
    """集中管理所有路径和配置，避免散乱的全局变量。"""
    home_dir: str = ""
    desktop_path: str = ""
    sudo_user: str = ""
    cuda_bin: str = ""
    cuda_lib: str = ""

    def __post_init__(self):
        self.sudo_user = os.environ.get("SUDO_USER", "")
        self.home_dir = self._get_real_home()
        self.desktop_path = os.path.join(self.home_dir, "Desktop")
        cuda_home = self._detect_cuda()
        if cuda_home:
            self.cuda_bin = os.path.join(cuda_home, "bin")
            self.cuda_lib = os.path.join(cuda_home, "lib64")
            # 注入环境变量（只做一次）
            os.environ["PATH"] = self.cuda_bin + ":" + os.environ.get("PATH", "")
            os.environ["LD_LIBRARY_PATH"] = self.cuda_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
            log.info("检测到 CUDA: %s", cuda_home)
        else:
            log.warning("未检测到 CUDA，GPU 相关功能可能不可用")

    def _get_real_home(self) -> str:
        if self.sudo_user:
            try:
                return pwd.getpwnam(self.sudo_user).pw_dir
            except KeyError:
                pass
        return str(Path.home())

    @staticmethod
    def _detect_cuda() -> Optional[str]:
        """自动检测 CUDA 安装路径，按优先级查找。"""
        # 1. 环境变量优先
        if os.environ.get("CUDA_HOME"):
            return os.environ["CUDA_HOME"]
        # 2. 常见安装路径
        cuda_dir = Path("/usr/local")
        if cuda_dir.exists():
            # 找 cuda-xx.x 或 cuda 软链接
            candidates = sorted(
                cuda_dir.glob("cuda*"),
                key=lambda p: p.is_symlink(),  # 软链接优先
                reverse=True,
            )
            for c in candidates:
                if (c / "bin" / "nvcc").exists() or c.is_symlink():
                    return str(c)
        # 3. 从 nvcc 推断
        nvcc = shutil.which("nvcc")
        if nvcc:
            return str(Path(nvcc).parent.parent)
        return None


# 全局配置实例（延迟到 main 中初始化）
cfg: Optional[Config] = None


# ============================
# 工具函数
# ============================

def run_command(cmd, check=True, capture_output=False, fatal=False):
    """运行 Shell 命令。fatal=True 时失败会退出脚本。"""
    log.debug("执行: %s", cmd)
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, check=check,
                                    capture_output=True, text=True)
            return result.stdout, result.stderr
        else:
            subprocess.run(cmd, shell=True, check=check)
            return "", ""
    except subprocess.CalledProcessError as e:
        log.error("命令失败 (rc=%s): %s", e.returncode, cmd)
        if e.stderr:
            log.error("stderr: %s", e.stderr.strip())
        if fatal:
            sys.exit(1)
        return "", str(e)


def print_banner(text):
    """打印横幅"""
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)


def yes_no_prompt(question, default="y"):
    """是/否提示"""
    options = "Y/n" if default.lower() in ("y", "yes") else "y/N"
    full_question = f"{question} [{options}]: "

    while True:
        answer = input(full_question).strip().lower()
        if not answer:
            answer = default.lower()
        if answer in ("y", "yes"):
            return True
        elif answer in ("n", "no"):
            return False
        print("请输入 y 或 n")


def check_nvidia_driver_installed():
    """检查 NVIDIA 驱动是否已安装"""
    try:
        result = subprocess.run("nvidia-smi", shell=True,
                                capture_output=True, text=True)
        return result.returncode == 0 and "NVIDIA-SMI" in result.stdout
    except Exception:
        return False


def find_gpu_burn():
    """搜索 gpu_burn 可执行文件，返回绝对路径或 None。"""
    home = cfg.home_dir if cfg else str(Path.home())
    candidates = [
        "/usr/local/bin/gpu_burn",
        os.path.join(home, "gpu-burn", "gpu_burn"),
        shutil.which("gpu_burn"),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return os.path.abspath(p)
    return None


# ============================
# GUI 环境 & 终端启动
# ============================

def fix_gui_env():
    """修复 GUI 环境变量（DISPLAY / XAUTHORITY）。"""
    try:
        if cfg.sudo_user:
            for var in ("DISPLAY", "XAUTHORITY"):
                cmd = f"sudo -u {cfg.sudo_user} bash -c 'echo ${var}'"
                result = subprocess.run(cmd, shell=True,
                                        capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip():
                    os.environ[var] = result.stdout.strip()
                    log.debug("设置 %s=%s", var, os.environ[var])
    except Exception as e:
        log.warning("获取 GUI 环境变量失败: %s", e)

    os.environ.setdefault("DISPLAY", ":0")
    user = cfg.sudo_user or os.getlogin()
    os.environ.setdefault("XAUTHORITY", f"/home/{user}/.Xauthority")


def _get_terminals():
    """返回可用终端列表 [(name, base_cmd), ...]。"""
    terminals = []
    for name in ("xterm", "gnome-terminal", "xfce4-terminal"):
        if shutil.which(name):
            terminals.append((name, [name]))
    return terminals


def launch_in_terminal(term_name, term_cmd, title, shell_cmd):
    """
    统一的终端启动方法。
    自动处理 xterm vs gnome-terminal 的参数差异。
    返回 Popen 或 None。
    """
    if term_name == "xterm":
        full_cmd = term_cmd + ["-title", title, "-e", "bash", "-c", shell_cmd]
    else:
        full_cmd = term_cmd + ["--title", title, "--", "bash", "-c", shell_cmd]

    try:
        proc = subprocess.Popen(
            full_cmd, env=os.environ,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        log.info("已启动终端窗口: %s (%s)", title, term_name)
        return proc
    except Exception as e:
        log.error("启动终端失败 [%s/%s]: %s", term_name, title, e)
        return None


def open_monitors(ping_duration=None):
    """打开 htop / nvidia-smi / ping 监控窗口。
    返回 (ping_log_file, monitor_procs) 二元组。
    monitor_procs 是需要在测试结束时清理的 Popen 对象列表。
    """
    monitor_procs = []

    if not os.environ.get("DISPLAY"):
        log.warning("未检测到 DISPLAY，跳过监控窗口")
        return None, monitor_procs

    log.info("尝试打开监控窗口...")

    # 确保至少有 xterm
    if not shutil.which("xterm"):
        run_command("sudo apt install -y xterm", check=False)

    terminals = _get_terminals()
    if not terminals:
        log.warning("未找到终端模拟器，跳过监控窗口")
        return None, monitor_procs

    term_name, term_cmd = terminals[0]
    ping_log_file = None

    # htop
    p = launch_in_terminal(term_name, term_cmd, "CPU/内存监控 (htop)", "htop")
    if p:
        monitor_procs.append(p)

    # nvidia-smi
    if shutil.which("nvidia-smi"):
        p = launch_in_terminal(
            term_name, term_cmd, "GPU 监控 (nvidia-smi)",
            "while true; do clear; nvidia-smi; sleep 1; done",
        )
        if p:
            monitor_procs.append(p)

    # ping（带日志）
    if ping_duration is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ping_log_file = f"/tmp/ping_{timestamp}.log"
        shell_cmd = f"""
echo "=== Ping test started at $(date) ===" | tee {ping_log_file}
timeout -s INT {ping_duration} ping bilibili.com | tee -a {ping_log_file}
echo "=== Ping test finished at $(date) ===" | tee -a {ping_log_file}
echo ""
echo "------ Final Statistics ------"
tail -n 10 {ping_log_file} | grep -E "packet loss|rtt|丢包|平均"
echo ""
read -p "Press Enter to close this window..."
"""
        p = launch_in_terminal(term_name, term_cmd, "网络测试 (ping)", shell_cmd)
        if p:
            monitor_procs.append(p)
            log.info("ping 日志将保存至: %s", ping_log_file)
        else:
            ping_log_file = None

    return ping_log_file, monitor_procs


# ============================
# 命令行监控
# ============================

def command_line_monitor():
    """命令行监控模式（每 5 秒刷新）"""
    print("\n📈 命令行监控模式（每 5 秒刷新，Ctrl+C 停止）")
    print("=" * 60)
    try:
        while True:
            subprocess.run("clear", shell=True)
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"监控时间: {now}  |  Ctrl+C 停止")
            print("=" * 60)

            # CPU
            print("\n📊 CPU:")
            r = subprocess.run("top -bn1 | grep 'Cpu(s)'", shell=True,
                               capture_output=True, text=True)
            if r.returncode == 0:
                print(f"  {r.stdout.strip()}")

            # 内存
            print("\n💾 内存:")
            r = subprocess.run("free -h | grep -E '^Mem:|^Swap:'", shell=True,
                               capture_output=True, text=True)
            if r.returncode == 0:
                for line in r.stdout.strip().split("\n"):
                    print(f"  {line}")

            # GPU
            if shutil.which("nvidia-smi"):
                print("\n🎮 GPU:")
                r = subprocess.run(
                    "nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,"
                    "memory.used,memory.total --format=csv,noheader",
                    shell=True, capture_output=True, text=True,
                )
                if r.returncode == 0:
                    parts = [p.strip() for p in r.stdout.strip().split(",")]
                    if len(parts) >= 5:
                        print(f"  显卡: {parts[0]}  温度: {parts[1]}°C  "
                              f"使用率: {parts[2]}  显存: {parts[3]}/{parts[4]}")

            # 负载
            print("\n⚡ 系统负载:")
            r = subprocess.run("uptime", shell=True, capture_output=True, text=True)
            if r.returncode == 0 and "load average:" in r.stdout:
                print(f"  {r.stdout.split('load average:')[1].strip()}")

            print("\n" + "=" * 60)
            for i in range(5, 0, -1):
                print(f"\r{i}秒后刷新...", end="", flush=True)
                time.sleep(1)
            print()
    except KeyboardInterrupt:
        print("\n监控已停止")


# ============================
# 数据采集（psutil 优先）
# ============================

def collect_stats():
    """收集 CPU / 内存 / GPU 状态。psutil 优先，回退到命令行。"""
    stats: Dict = {"timestamp": time.time()}

    if HAS_PSUTIL:
        try:
            stats["cpu_util"] = psutil.cpu_percent(interval=1)
            stats["mem_util"] = psutil.virtual_memory().percent
        except Exception:
            pass
    else:
        # 回退：mpstat
        if shutil.which("mpstat"):
            try:
                r = subprocess.run(
                    "mpstat 1 1 | awk '/Average/ && /all/ {print 100 - $12}'",
                    shell=True, capture_output=True, text=True,
                )
                if r.returncode == 0 and r.stdout.strip():
                    stats["cpu_util"] = float(r.stdout.strip())
            except Exception:
                pass
        # 回退：free
        try:
            r = subprocess.run(
                "free | grep Mem | awk '{print $3/$2 * 100.0}'",
                shell=True, capture_output=True, text=True,
            )
            if r.returncode == 0 and r.stdout.strip():
                stats["mem_util"] = float(r.stdout.strip())
        except Exception:
            pass

    # GPU（始终通过 nvidia-smi，psutil 不支持）
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(
                "nvidia-smi --query-gpu=temperature.gpu,power.draw,"
                "utilization.gpu,memory.used,memory.total "
                "--format=csv,noheader,nounits",
                shell=True, capture_output=True, text=True,
            )
            if r.returncode == 0 and r.stdout:
                parts = [p.strip() for p in r.stdout.strip().split(",")]
                if len(parts) >= 5:
                    stats["gpu_temp"] = float(parts[0])
                    stats["gpu_power"] = float(parts[1])
                    stats["gpu_util"] = float(parts[2])
                    stats["gpu_mem_used"] = float(parts[3])
                    stats["gpu_mem_total"] = float(parts[4])
        except Exception:
            pass

    return stats


# ============================
# 专业网络测试模块
# ============================

@dataclass
class NetworkTestConfig:
    """网络测试配置。"""
    targets: List[str] = field(default_factory=lambda: [
        "223.5.5.5",       # 阿里 DNS
        "119.29.29.29",    # 腾讯 DNS
        "114.114.114.114", # 114 DNS
        "baidu.com",       # 国内网站
    ])
    ping_count: int = 20           # 每个目标 ping 次数
    ping_interval: float = 0.2     # ping 间隔（秒）
    download_urls: List[str] = field(default_factory=lambda: [
        "https://dldir1.qq.com/qqfile/qq/PCQQ9.7.17/QQ9.7.17.29225.exe",
        "https://speed.cloudflare.com/__down?bytes=10485760",
    ])
    download_time_limit: int = 15  # 下载超时（秒）
    upload_urls: List[str] = field(default_factory=lambda: [
        "https://speed.cloudflare.com/__up",
    ])
    upload_size_mb: int = 10       # 上传测试数据大小（MB）
    upload_time_limit: int = 15    # 上传超时（秒）
    iperf3_server: str = ""        # iperf3 服务器地址，空则跳过
    iperf3_duration: int = 10      # iperf3 测试时长（秒）
    iperf3_port: int = 5201        # iperf3 端口
    http_urls: List[str] = field(default_factory=lambda: [
        "https://www.baidu.com",
        "https://www.bilibili.com",
        "https://www.qq.com",
    ])
    dns_servers: List[str] = field(default_factory=lambda: [
        "223.5.5.5",
        "119.29.29.29",
        "114.114.114.114",
        "8.8.8.8",
    ])
    dns_domain: str = "www.baidu.com"
    # 持续 ping 监控
    continuous_ping_interval: float = 5.0   # 持续 ping 间隔（秒）
    # 周期性全量测试
    periodic_test_interval: int = 1800      # 全量测试间隔（秒），默认 30 分钟
    periodic_test_count: int = 0            # 已完成的周期测试次数（运行时填充）


class ContinuousPingMonitor:
    """持续 ping 监控，在后台对多个目标保持长时 ping。"""

    def __init__(self, targets: List[str], interval: float = 5.0, log_dir: str = "/tmp"):
        self.targets = targets
        self.interval = interval
        self.processes: Dict[str, subprocess.Popen] = {}
        self.stats: Dict[str, Dict] = {}
        self.running = False
        self._thread: Optional[threading.Thread] = None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"continuous_ping_{timestamp}.log")
        self._log_f = None
        self._prev_rtt: Dict[str, float] = {}
        self._jitter_sum: Dict[str, float] = {}
        self._jitter_count: Dict[str, int] = {}

    def start(self):
        """启动持续 ping 监控。"""
        self.running = True
        self._log_f = open(self.log_file, "w", encoding="utf-8")
        self._log_f.write(f"# Continuous ping started at {datetime.now()}\n")
        self._log_f.write(f"# Targets: {', '.join(self.targets)}\n")
        self._log_f.write(f"# Interval: {self.interval}s\n")
        self._log_f.flush()

        for target in self.targets:
            self.stats[target] = {"sent": 0, "received": 0, "rtts": []}
            self._jitter_sum[target] = 0.0
            self._jitter_count[target] = 0
            cmd = ["ping", "-i", str(self.interval), "-W", "2", target]
            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                self.processes[target] = proc
            except Exception as e:
                log.error("启动持续 ping %s 失败: %s", target, e)

        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        log.info("持续 ping 监控已启动: %s", ", ".join(self.targets))

    def _reader(self):
        """读取所有 ping 进程的输出（非阻塞，避免 readline 阻塞其他目标）。"""
        import select
        while self.running:
            # 收集还活着的进程的 stdout fd
            fd_map = {}  # fd -> target
            for target, proc in list(self.processes.items()):
                if proc.poll() is None and proc.stdout:
                    fd_map[proc.stdout.fileno()] = (target, proc.stdout)

            if not fd_map:
                time.sleep(0.1)
                continue

            # 非阻塞等待，最多等 1 秒
            try:
                ready, _, _ = select.select(list(fd_map.keys()), [], [], 1.0)
            except (ValueError, OSError):
                time.sleep(0.1)
                continue

            for fd in ready:
                target, stdout = fd_map[fd]
                try:
                    line = stdout.readline()
                    if line:
                        self._process_line(target, line)
                except Exception:
                    pass

    def _process_line(self, target: str, line: str):
        """处理单行 ping 输出。"""
        ts = datetime.now().strftime("%H:%M:%S")
        line = line.strip()
        if not line:
            return

        # 记录到日志文件
        if self._log_f:
            self._log_f.write(f"[{ts}] [{target}] {line}\n")
            self._log_f.flush()

        stats = self.stats[target]

        # "64 bytes from ... time=xx.x ms"
        m = re.search(r"time=([\d.]+)\s*ms", line)
        if m:
            rtt = float(m.group(1))
            stats["received"] += 1
            stats["sent"] += 1
            stats["rtts"].append(rtt)
            # 计算抖动
            if target in self._prev_rtt:
                jitter = abs(rtt - self._prev_rtt[target])
                self._jitter_sum[target] += jitter
                self._jitter_count[target] += 1
            self._prev_rtt[target] = rtt
            return

        # "Request timeout" 或无响应
        if re.search(r"(timeout|100% packet loss|Destination Host Unreachable)", line, re.IGNORECASE):
            stats["sent"] += 1
            return

        # 最终统计行 "rtt min/avg/max/mdev = ..."
        m = re.search(
            r"=\s*([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)\s*ms", line
        )
        if m:
            stats["final_min"] = float(m.group(1))
            stats["final_avg"] = float(m.group(2))
            stats["final_max"] = float(m.group(3))
            stats["final_mdev"] = float(m.group(4))

    def stop(self) -> Dict:
        """停止监控并返回统计结果。"""
        self.running = False

        for target, proc in self.processes.items():
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

        if self._thread:
            self._thread.join(timeout=10)

        if self._log_f:
            self._log_f.close()

        results = {}
        for target, stats in self.stats.items():
            sent = stats["sent"]
            received = stats["received"]
            rtts = stats["rtts"]
            loss = sent - received
            loss_pct = (loss / sent * 100) if sent > 0 else 0

            r: Dict = {
                "sent": sent,
                "received": received,
                "lost": loss,
                "loss_pct": round(loss_pct, 1),
            }
            if rtts:
                r["min_ms"] = round(min(rtts), 2)
                r["max_ms"] = round(max(rtts), 2)
                r["avg_ms"] = round(sum(rtts) / len(rtts), 2)
                if len(rtts) > 1:
                    mean = sum(rtts) / len(rtts)
                    r["stddev_ms"] = round(
                        (sum((x - mean) ** 2 for x in rtts) / len(rtts)) ** 0.5, 2
                    )
                # 平均抖动
                jcnt = self._jitter_count.get(target, 0)
                if jcnt > 0:
                    r["jitter_ms"] = round(self._jitter_sum[target] / jcnt, 2)

            results[target] = r

        log.info("持续 ping 监控已停止，日志: %s", self.log_file)
        return results


class NetworkTester:
    """专业网络测试套件。"""

    def __init__(self, config: Optional[NetworkTestConfig] = None):
        self.cfg = config or NetworkTestConfig()
        self.results: Dict = {}

    def run_all(self) -> Dict:
        """运行全部网络测试，返回结果字典。"""
        print_banner("🌐 专业网络测试")
        tests = [
            ("多目标 Ping 测试", self.test_multi_ping),
            ("DNS 解析测试", self.test_dns),
            ("TCP 连接测试", self.test_http_connect),
            ("下载速度测试", self.test_download_speed),
            ("上传速度测试", self.test_upload_speed),
        ]
        # iperf3 需要服务器
        if self.cfg.iperf3_server and shutil.which("iperf3"):
            tests.append(("iperf3 带宽测试", self.test_iperf3))

        for name, func in tests:
            print(f"\n  ▶ {name}...")
            try:
                result = func()
                if result:
                    self.results[name] = result
                    print(f"  ✅ {name} 完成")
                else:
                    print(f"  ⚠️  {name} 无结果")
            except Exception as e:
                log.error("%s 失败: %s", name, e)
                self.results[name] = {"error": str(e)}

        return self.results

    def test_multi_ping(self) -> Dict:
        """多目标 ping 测试，返回每个目标的统计。"""
        results = {}
        for target in self.cfg.targets:
            log.debug("ping %s x%d", target, self.cfg.ping_count)
            cmd = (
                f"ping -c {self.cfg.ping_count} "
                f"-i {self.cfg.ping_interval} "
                f"-W 2 {target}"
            )
            try:
                r = subprocess.run(cmd, shell=True, capture_output=True,
                                   text=True, timeout=30)
                output = r.stdout + r.stderr
                info = self._parse_ping_output(output)
                info["target"] = target
                results[target] = info
                # 实时显示
                status = "✅" if info.get("loss_pct", 100) < 5 else "⚠️" if info.get("loss_pct", 100) < 20 else "❌"
                print(f"    {status} {target:20s}  "
                      f"丢包={info.get('loss', '?'):>5s}  "
                      f"延迟={info.get('avg', '?'):>8s}  "
                      f"抖动={info.get('mdev', '?'):>8s}")
            except subprocess.TimeoutExpired:
                log.warning("ping %s 超时", target)
                results[target] = {"error": "timeout", "target": target}
            except Exception as e:
                results[target] = {"error": str(e), "target": target}

        # 汇总
        valid = [v for v in results.values() if "avg_ms" in v]
        if valid:
            avg_latencies = [v["avg_ms"] for v in valid]
            avg_losses = [v.get("loss_pct", 0) for v in valid]
            results["_summary"] = {
                "avg_latency_ms": sum(avg_latencies) / len(avg_latencies),
                "avg_loss_pct": sum(avg_losses) / len(avg_losses),
                "targets_ok": len(valid),
                "targets_total": len(self.cfg.targets),
            }
        return results

    def test_dns(self) -> Dict:
        """DNS 解析测试，测试多个 DNS 服务器的响应速度。"""
        results = {}
        for dns in self.cfg.dns_servers:
            times = []
            for i in range(5):
                try:
                    start = time.time()
                    r = subprocess.run(
                        f"dig @{dns} {self.cfg.dns_domain} +short +time=3 +tries=1",
                        shell=True, capture_output=True, text=True, timeout=5,
                    )
                    elapsed = (time.time() - start) * 1000  # ms
                    if r.returncode == 0 and r.stdout.strip():
                        times.append(elapsed)
                except Exception:
                    pass

            if times:
                avg_ms = sum(times) / len(times)
                results[dns] = {
                    "avg_ms": round(avg_ms, 1),
                    "min_ms": round(min(times), 1),
                    "max_ms": round(max(times), 1),
                    "success": len(times),
                    "total": 5,
                }
                icon = "✅" if avg_ms < 50 else "⚠️" if avg_ms < 150 else "❌"
                print(f"    {icon} DNS {dns:18s}  "
                      f"平均={avg_ms:>7.1f}ms  "
                      f"({len(times)}/5 成功)")
            else:
                results[dns] = {"error": "all queries failed"}

        # 如果 dig 不可用，回退到 getent
        if not results and not shutil.which("dig"):
            log.info("dig 不可用，使用 getent 回退")
            for _ in range(5):
                try:
                    start = time.time()
                    subprocess.run(f"getent hosts {self.cfg.dns_domain}",
                                   shell=True, capture_output=True, timeout=5)
                    elapsed = (time.time() - start) * 1000
                    results.setdefault("_getent", []).append(elapsed)
                except Exception:
                    pass
            if "_getent" in results:
                times = results.pop("_getent")
                results["system_resolver"] = {
                    "avg_ms": round(sum(times) / len(times), 1),
                    "samples": len(times),
                }

        return results

    def test_http_connect(self) -> Dict:
        """测试 TCP 连接 / TLS 握手 / 首字节时间。"""
        results = {}
        for url in self.cfg.http_urls:
            try:
                r = subprocess.run(
                    f"curl -o /dev/null -s -w "
                    f"'%{{time_namelookup}} %{{time_connect}} %{{time_appconnect}} "
                    f"%{{time_starttransfer}} %{{time_total}} %{{http_code}}' "
                    f"--max-time 10 {url}",
                    shell=True, capture_output=True, text=True, timeout=15,
                )
                if r.returncode == 0 and r.stdout.strip():
                    parts = r.stdout.strip().split()
                    if len(parts) >= 6:
                        dns_t = float(parts[0]) * 1000
                        conn_t = float(parts[1]) * 1000
                        tls_t = float(parts[2]) * 1000
                        ttfb = float(parts[3]) * 1000
                        total_t = float(parts[4]) * 1000
                        http_code = parts[5]
                        host = url.split("//")[1].split("/")[0]
                        results[host] = {
                            "dns_ms": round(dns_t, 1),
                            "connect_ms": round(conn_t, 1),
                            "tls_ms": round(tls_t, 1),
                            "ttfb_ms": round(ttfb, 1),
                            "total_ms": round(total_t, 1),
                            "http_code": http_code,
                        }
                        icon = "✅" if ttfb < 500 else "⚠️" if ttfb < 2000 else "❌"
                        print(f"    {icon} {host:25s}  "
                              f"TCP={conn_t:>7.1f}ms  "
                              f"TLS={tls_t:>7.1f}ms  "
                              f"TTFB={ttfb:>7.1f}ms  "
                              f"HTTP {http_code}")
            except Exception as e:
                host = url.split("//")[1].split("/")[0]
                results[host] = {"error": str(e)}

        return results

    def test_download_speed(self) -> Dict:
        """下载速度测试。"""
        results = {}
        for url in self.cfg.download_urls:
            try:
                r = subprocess.run(
                    f"curl -o /dev/null -s -w "
                    f"'%{{size_download}} %{{time_total}} %{{speed_download}}' "
                    f"--max-time {self.cfg.download_time_limit} {url}",
                    shell=True, capture_output=True, text=True,
                    timeout=self.cfg.download_time_limit + 5,
                )
                if r.returncode == 0 and r.stdout.strip():
                    parts = r.stdout.strip().split()
                    if len(parts) >= 3:
                        size_bytes = float(parts[0])
                        duration_s = float(parts[1])
                        speed_bps = float(parts[2])
                        speed_mbps = speed_bps * 8 / 1_000_000
                        size_mb = size_bytes / 1_048_576
                        name = url.split("//")[1].split("/")[0]
                        results[name] = {
                            "speed_mbps": round(speed_mbps, 2),
                            "size_mb": round(size_mb, 2),
                            "duration_s": round(duration_s, 2),
                        }
                        icon = "✅" if speed_mbps > 10 else "⚠️" if speed_mbps > 1 else "❌"
                        print(f"    {icon} {name:30s}  "
                              f"{speed_mbps:>8.2f} Mbps  "
                              f"({size_mb:.1f} MB / {duration_s:.1f}s)")
            except subprocess.TimeoutExpired:
                name = url.split("//")[1].split("/")[0]
                results[name] = {"error": "timeout"}
            except Exception as e:
                name = url.split("//")[1].split("/")[0]
                results[name] = {"error": str(e)}

        return results

    def test_upload_speed(self) -> Dict:
        """上传速度测试，通过 POST 发送随机数据到测试服务器。"""
        results = {}
        size_bytes = self.cfg.upload_size_mb * 1048576

        for url in self.cfg.upload_urls:
            try:
                # 生成临时随机数据文件
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
                tmp_path = tmp_file.name
                # 用 dd 生成随机数据（比 python 逐字节写入快得多）
                subprocess.run(
                    f"dd if=/dev/urandom of={tmp_path} bs=1M "
                    f"count={self.cfg.upload_size_mb} 2>/dev/null",
                    shell=True, check=True,
                )

                r = subprocess.run(
                    f"curl -o /dev/null -s -w "
                    f"'%{{size_upload}} %{{time_total}} %{{speed_upload}}' "
                    f"--max-time {self.cfg.upload_time_limit} "
                    f"-X POST -d @{tmp_path} {url}",
                    shell=True, capture_output=True, text=True,
                    timeout=self.cfg.upload_time_limit + 5,
                )

                # 清理临时文件
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

                if r.returncode == 0 and r.stdout.strip():
                    parts = r.stdout.strip().split()
                    if len(parts) >= 3:
                        uploaded_bytes = float(parts[0])
                        duration_s = float(parts[1])
                        speed_bps = float(parts[2])
                        speed_mbps = speed_bps * 8 / 1_000_000
                        uploaded_mb = uploaded_bytes / 1_048_576
                        name = url.split("//")[1].split("/")[0]
                        results[name] = {
                            "speed_mbps": round(speed_mbps, 2),
                            "size_mb": round(uploaded_mb, 2),
                            "duration_s": round(duration_s, 2),
                        }
                        icon = "✅" if speed_mbps > 5 else "⚠️" if speed_mbps > 0.5 else "❌"
                        print(f"    {icon} {name:30s}  "
                              f"{speed_mbps:>8.2f} Mbps  "
                              f"({uploaded_mb:.1f} MB / {duration_s:.1f}s)")
            except subprocess.TimeoutExpired:
                name = url.split("//")[1].split("/")[0]
                results[name] = {"error": "timeout"}
            except Exception as e:
                name = url.split("//")[1].split("/")[0]
                results[name] = {"error": str(e)}

        return results

    def test_iperf3(self) -> Dict:
        """iperf3 带宽测试（需要对端有 iperf3 -s 运行）。"""
        results = {}
        server = self.cfg.iperf3_server
        port = self.cfg.iperf3_port
        duration = self.cfg.iperf3_duration

        # 下行测试
        for direction, flag in [("download", "-R"), ("upload", "")]:
            cmd = (f"iperf3 -c {server} -p {port} -t {duration} "
                   f"-J {flag} 2>/dev/null")
            try:
                r = subprocess.run(cmd, shell=True, capture_output=True,
                                   text=True, timeout=duration + 15)
                if r.returncode == 0 and r.stdout.strip():
                    data = json.loads(r.stdout)
                    bits_per_sec = data["end"]["sum_received"]["bits_per_second"] \
                        if direction == "download" \
                        else data["end"]["sum_sent"]["bits_per_second"]
                    mbps = bits_per_sec / 1_000_000
                    retransmits = data["end"].get("sum_sent", {}).get("retransmits", 0)
                    results[direction] = {
                        "mbps": round(mbps, 2),
                        "retransmits": retransmits,
                    }
                    print(f"    ✅ iperf3 {direction:8s}: {mbps:>10.2f} Mbps"
                          f"  (重传: {retransmits})")
            except subprocess.TimeoutExpired:
                results[direction] = {"error": "timeout"}
            except json.JSONDecodeError:
                results[direction] = {"error": "invalid json response"}
            except Exception as e:
                results[direction] = {"error": str(e)}

        return results

    @staticmethod
    def _parse_ping_output(output: str) -> Dict:
        """解析 ping 命令输出，兼容中英文。"""
        info = {}

        # 丢包率
        m = re.search(r"(\d+(?:\.\d+)?)%\s*(?:packet loss|丢包)", output, re.IGNORECASE)
        if m:
            info["loss"] = m.group(1) + "%"
            info["loss_pct"] = float(m.group(1))

        # RTT 四元组
        m = re.search(
            r"(?:rtt|延迟|往返时间).*?=\s*([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)\s*ms",
            output, re.IGNORECASE,
        )
        if m:
            info["min"] = m.group(1) + " ms"
            info["avg"] = m.group(2) + " ms"
            info["max"] = m.group(3) + " ms"
            info["mdev"] = m.group(4) + " ms"
            info["min_ms"] = float(m.group(1))
            info["avg_ms"] = float(m.group(2))
            info["max_ms"] = float(m.group(3))
            info["mdev_ms"] = float(m.group(4))

        # 传输统计
        m = re.search(r"(\d+)\s*(?:packets|个包).*?(\d+)\s*(?:received|已接收)", output)
        if m:
            info["tx"] = int(m.group(1))
            info["rx"] = int(m.group(2))

        return info


# ============================
# 系统信息采集
# ============================

def _run(cmd, timeout=5):
    """快捷执行命令，返回 stdout（失败返回空串）。"""
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True,
                           text=True, timeout=timeout)
        if r.returncode != 0:
            log.debug("命令失败 (rc=%d): %s", r.returncode, cmd)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception as e:
        log.debug("命令异常: %s → %s", cmd, e)
        return ""


def collect_system_info() -> Dict:
    """采集系统硬件和环境信息，用于报告头部。"""
    info: Dict = {}

    # --- 基本系统 ---
    info["hostname"] = _run("hostname")
    info["os"] = _run("cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d'\"' -f2")
    if not info["os"]:
        info["os"] = _run("uname -s -r")
    info["kernel"] = _run("uname -r")
    info["arch"] = _run("uname -m")
    info["test_time"] = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

    # --- CPU ---
    info["cpu_model"] = _run("lscpu | grep 'Model name' | sed 's/.*:\\s*//' | head -1")
    if not info["cpu_model"]:
        info["cpu_model"] = _run("grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs")
    info["cpu_cores"] = _run("nproc")
    cpu_freq = _run("lscpu | grep 'CPU max MHz' | awk '{print $NF}'")
    if cpu_freq:
        info["cpu_freq"] = f"{float(cpu_freq)/1000:.2f} GHz"
    else:
        info["cpu_freq"] = _run("lscpu | grep 'Model name' | grep -oP '@\\s*\\K.*' | head -1")

    # --- 内存 ---
    mem_total_kb = _run("grep MemTotal /proc/meminfo | awk '{print $2}'")
    if mem_total_kb:
        mem_gb = int(mem_total_kb) / 1048576
        info["memory_total"] = f"{mem_gb:.1f} GB"
    else:
        info["memory_total"] = _run("free -h | awk '/^Mem:/ {print $2}'")

    # --- GPU ---
    if shutil.which("nvidia-smi"):
        # GPU 型号
        info["gpu_model"] = _run(
            "nvidia-smi --query-gpu=name --format=csv,noheader | head -1"
        )
        # GPU 数量
        gpu_count = _run("nvidia-smi --query-gpu=name --format=csv,noheader | wc -l")
        info["gpu_count"] = gpu_count if gpu_count else "1"

        # GPU SN（序列号）- nvidia-smi -q -d SERIAL（需要 root 权限）
        sn_output = ""
        if os.geteuid() == 0:
            sn_output = _run("nvidia-smi -q -d SERIAL 2>/dev/null")
        serials = re.findall(r"Serial Number\s*:\s*(\S+)", sn_output)
        if serials:
            info["gpu_sn"] = serials[0]
            if len(serials) > 1:
                info["gpu_sn_all"] = serials
        else:
            # 回退：用 UUID
            uuid = _run("nvidia-smi --query-gpu=gpu_uuid --format=csv,noheader | head -1")
            info["gpu_sn"] = uuid if uuid else "未知"

        # GPU 显存
        vram = _run(
            "nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1"
        )
        info["gpu_vram"] = vram if vram else "未知"

        # 驱动版本
        info["driver_version"] = _run(
            "nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1"
        )

        # CUDA toolkit 版本（优先从 nvcc 获取，这才是实际安装的版本）
        cuda_ver = _run("nvcc --version | grep 'release' | awk '{print $6}' | cut -d',' -f1")
        if not cuda_ver:
            # fallback: 从 /usr/local/cuda-* 目录名推断
            cuda_ver = _run("ls -d /usr/local/cuda-* 2>/dev/null | head -1 | grep -oP '[\\d.]+'")
        info["cuda_version"] = cuda_ver if cuda_ver else "未知"

        # 驱动支持的最大 CUDA 版本（仅供参考，不是实际安装版本）
        cuda_max = _run("nvidia-smi | grep -oP 'CUDA Version:\\s*\\K[\\d.]+'")
        if cuda_max:
            info["cuda_max_version"] = cuda_max

        # 所有 GPU 信息（多卡场景）
        gpu_names = _run("nvidia-smi --query-gpu=name --format=csv,noheader").split("\n")
        gpu_uuids = _run("nvidia-smi --query-gpu=gpu_uuid --format=csv,noheader").split("\n")
        if len(gpu_names) > 1:
            info["gpu_details"] = []
            for i, name in enumerate(gpu_names):
                detail = {"index": i, "name": name.strip()}
                if i < len(serials):
                    detail["serial"] = serials[i]
                elif i < len(gpu_uuids):
                    detail["uuid"] = gpu_uuids[i].strip()
                info["gpu_details"].append(detail)

    # --- 磁盘 ---
    root_disk = _run("df -h / | awk 'NR==2 {print $2}'")
    info["disk_total"] = root_disk if root_disk else "未知"

    return info


# ============================
# 报告生成
# ============================

def generate_report(data, ping_stats=None, net_results=None, output_file=None,
                    ping_monitor_results=None, periodic_net_results=None,
                    json_output_file=""):
    """
    生成压力测试报告。
    ping_stats:  旧版 ping 日志解析结果（兼容）
    net_results: NetworkTester.run_all() 返回的完整网络测试结果
    ping_monitor_results: ContinuousPingMonitor.stop() 返回的持续 ping 统计
    periodic_net_results: 周期性全量网络测试结果列表
    """
    # ---- 采集系统信息 ----
    sys_info = collect_system_info()

    def _disp_width(s):
        """计算字符串的终端显示宽度（CJK/emoji 算 2 列）。"""
        w = 0
        for ch in s:
            cp = ord(ch)
            if (
                (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF) or
                (0x3000 <= cp <= 0x303F) or (0xFF00 <= cp <= 0xFFEF) or
                (0x2E80 <= cp <= 0x2EFF) or (0x1100 <= cp <= 0x11FF) or
                (0x20000 <= cp <= 0x2A6DF) or (0x2F00 <= cp <= 0x2FDF) or
                (0xFE30 <= cp <= 0xFE4F) or (0xF900 <= cp <= 0xFAFF) or
                (0x2600 <= cp <= 0x27BF) or (0x2300 <= cp <= 0x23FF) or
                (0x2702 <= cp <= 0x27B0) or (0xFE00 <= cp <= 0xFE0F) or
                (0x1F300 <= cp <= 0x1F9FF) or (0x200D == cp) or
                (0x20E3 == cp) or (0xE0020 <= cp <= 0xE007F)
            ):
                w += 2
            else:
                w += 1
        return w

    def _pad(text, width):
        """将 text 右填充空格到指定显示宽度。"""
        deficit = width - _disp_width(text)
        return text + " " * max(0, deficit)
    ######################## 默认压测60秒 ########################
    lines = ["=" * 60, "压力测试报告", "=" * 60, ""]

    # ---- 系统 & 硬件信息 ----
    lines.append("📋 测试环境信息")
    lines.append("-" * 40)
    lines.append(f"  测试时间:      {sys_info.get('test_time', '未知')}")

    if sys_info.get("hostname"):
        lines.append(f"  主机名:        {sys_info['hostname']}")
    if sys_info.get("os"):
        lines.append(f"  系统版本:      {sys_info['os']}")
    if sys_info.get("kernel"):
        lines.append(f"  内核版本:      {sys_info['kernel']}")

    lines.append("")
    if sys_info.get("cpu_model"):
        freq_str = f" @ {sys_info['cpu_freq']}" if sys_info.get("cpu_freq") else ""
        cores_str = f"  ({sys_info['cpu_cores']}核)" if sys_info.get("cpu_cores") else ""
        lines.append(f"  CPU 型号:      {sys_info['cpu_model']}{freq_str}{cores_str}")
    if sys_info.get("memory_total"):
        lines.append(f"  内存容量:      {sys_info['memory_total']}")
    if sys_info.get("disk_total"):
        lines.append(f"  磁盘容量:      {sys_info['disk_total']} (根分区)")

    if sys_info.get("gpu_model"):
        lines.append("")
        lines.append(f"  GPU 型号:      {sys_info['gpu_model']}")
        if sys_info.get("gpu_count") and sys_info["gpu_count"] != "1":
            lines.append(f"  GPU 数量:      {sys_info['gpu_count']}")
        if sys_info.get("gpu_sn"):
            lines.append(f"  GPU SN号:      {sys_info['gpu_sn']}")
        if sys_info.get("gpu_vram"):
            lines.append(f"  GPU 显存:      {sys_info['gpu_vram']}")
        if sys_info.get("driver_version"):
            lines.append(f"  驱动版本:      {sys_info['driver_version']}")

        # 多卡详情
        if sys_info.get("gpu_details"):
            for gd in sys_info["gpu_details"]:
                sn = gd.get("serial", gd.get("uuid", "未知"))
                lines.append(f"    GPU {gd['index']}: {gd['name']}  SN: {sn}")

    # CUDA 版本独立输出（不依赖 gpu_model）
    if sys_info.get("cuda_version"):
        lines.append(f"  CUDA 版本:     {sys_info['cuda_version']}")
    if sys_info.get("cuda_max_version"):
        lines.append(f"  驱动最大CUDA:  {sys_info['cuda_max_version']} (驱动支持上限)")

    lines.append("")
    lines.append("=" * 60)

    # ---- 硬件压力测试结果 ----
    if not data:
        lines.append("没有收集到有效数据")
    else:
        def _stats(seq):
            if not seq:
                return None, None, None
            return max(seq), min(seq), sum(seq) / len(seq)

        cpu_utils = [d["cpu_util"] for d in data if "cpu_util" in d]
        mem_utils = [d["mem_util"] for d in data if "mem_util" in d]
        gpu_temps = [d["gpu_temp"] for d in data if "gpu_temp" in d]
        gpu_powers = [d["gpu_power"] for d in data if "gpu_power" in d]
        gpu_utils = [d["gpu_util"] for d in data if "gpu_util" in d]
        gpu_mem_used = [d["gpu_mem_used"] for d in data if "gpu_mem_used" in d]
        gpu_mem_total = data[0].get("gpu_mem_total") if data else None

        duration = data[-1]["timestamp"] - data[0]["timestamp"]
        lines.append(f"测试时长: {duration:.1f} 秒")
        lines.append(f"采样点数: {len(data)}")
        lines.append("")

        for label, seq, unit in [
            ("CPU 利用率", cpu_utils, "%"),
            ("内存利用率", mem_utils, "%"),
            ("GPU 温度", gpu_temps, "°C"),
            ("GPU 功耗", gpu_powers, "W"),
            ("GPU 利用率", gpu_utils, "%"),
        ]:
            mx, mn, avg = _stats(seq)
            if mx is not None:
                lines.append(f"{label} ({unit}): 平均={avg:.1f}, 最大={mx:.1f}, 最小={mn:.1f}")

        if gpu_mem_used and gpu_mem_total:
            percents = [u / gpu_mem_total * 100 for u in gpu_mem_used]
            mx, mn, avg = _stats(percents)
            lines.append(f"显存利用率 (%): 平均={avg:.1f}, 最大={mx:.1f}, 最小={mn:.1f}")
            lines.append(f"显存总量: {gpu_mem_total:.0f} MB")

    # ---- 网络测试结果 ----
    if net_results:
        lines.append("")
        lines.append("🌐 网络测试报告")
        lines.append("=" * 60)

        # 多目标 Ping
        if "多目标 Ping 测试" in net_results:
            ping_data = net_results["多目标 Ping 测试"]
            lines.append("")
            lines.append("📡 多目标 Ping 测试")
            lines.append("-" * 40)
            for key, val in ping_data.items():
                if key.startswith("_"):
                    continue
                if "error" in val:
                    lines.append(f"  {_pad(key, 24)}❌ {val['error']}")
                else:
                    lines.append(
                        f"  {_pad(key, 24)}"
                        f"丢包={val.get('loss', '?'):>5s}  "
                        f"平均={val.get('avg', '?'):>8s}  "
                        f"抖动={val.get('mdev', '?'):>8s}  "
                        f"范围={val.get('min', '?')}~{val.get('max', '?')}"
                    )
            summary = ping_data.get("_summary", {})
            if summary:
                lines.append(
                    f"  {_pad('汇总', 24)}"
                    f"平均延迟={summary.get('avg_latency_ms', 0):.1f}ms  "
                    f"平均丢包={summary.get('avg_loss_pct', 0):.1f}%  "
                    f"正常={summary.get('targets_ok', 0)}/{summary.get('targets_total', 0)}"
                )

        # DNS 测试
        if "DNS 解析测试" in net_results:
            dns_data = net_results["DNS 解析测试"]
            lines.append("")
            lines.append("🔍 DNS 解析测试")
            lines.append("-" * 40)
            for dns, val in dns_data.items():
                if "error" in val:
                    lines.append(f"  {_pad(dns, 24)}❌ {val['error']}")
                else:
                    lines.append(
                        f"  {_pad(dns, 24)}"
                        f"平均={val.get('avg_ms', '?'):>7}ms  "
                        f"范围={val.get('min_ms', '?')}~{val.get('max_ms', '?')}ms  "
                        f"({val.get('success', '?')}/{val.get('total', '?')} 成功)"
                    )

        # HTTP 连接测试
        if "TCP 连接测试" in net_results:
            http_data = net_results["TCP 连接测试"]
            lines.append("")
            lines.append("🔗 TCP 连接 / TLS 握手测试")
            lines.append("-" * 40)
            for host, val in http_data.items():
                if "error" in val:
                    lines.append(f"  {_pad(host, 24)}❌ {val['error']}")
                else:
                    lines.append(
                        f"  {_pad(host, 24)}"
                        f"DNS={val.get('dns_ms', 0):>6.1f}ms  "
                        f"TCP={val.get('connect_ms', 0):>6.1f}ms  "
                        f"TLS={val.get('tls_ms', 0):>7.1f}ms  "
                        f"TTFB={val.get('ttfb_ms', 0):>7.1f}ms  "
                        f"HTTP {val.get('http_code', '?')}"
                    )

        # 下载速度
        if "下载速度测试" in net_results:
            dl_data = net_results["下载速度测试"]
            lines.append("")
            lines.append("⬇️  下载速度测试")
            lines.append("-" * 40)
            for host, val in dl_data.items():
                if "error" in val:
                    lines.append(f"  {_pad(host, 28)}❌ {val['error']}")
                else:
                    lines.append(
                        f"  {_pad(host, 28)}"
                        f"{val.get('speed_mbps', 0):>8.2f} Mbps  "
                        f"({val.get('size_mb', 0):.1f} MB / {val.get('duration_s', 0):.1f}s)"
                    )

        # 上传速度
        if "上传速度测试" in net_results:
            ul_data = net_results["上传速度测试"]
            lines.append("")
            lines.append("⬆️  上传速度测试")
            lines.append("-" * 40)
            for host, val in ul_data.items():
                if "error" in val:
                    lines.append(f"  {_pad(host, 28)}❌ {val['error']}")
                else:
                    lines.append(
                        f"  {_pad(host, 28)}"
                        f"{val.get('speed_mbps', 0):>8.2f} Mbps  "
                        f"({val.get('size_mb', 0):.1f} MB / {val.get('duration_s', 0):.1f}s)"
                    )

        # iperf3
        if "iperf3 带宽测试" in net_results:
            iperf_data = net_results["iperf3 带宽测试"]
            lines.append("")
            lines.append("🚀 iperf3 带宽测试")
            lines.append("-" * 40)
            for direction, val in iperf_data.items():
                if "error" in val:
                    lines.append(f"  {direction:10s}  ❌ {val['error']}")
                else:
                    lines.append(
                        f"  {direction:10s}  {val.get('mbps', 0):>10.2f} Mbps  "
                        f"(重传: {val.get('retransmits', 0)})"
                    )

        lines.append("")
        lines.append("=" * 60)

    # ---- 持续 Ping 监控结果 ----
    if ping_monitor_results:
        lines.append("")
        lines.append("📡 持续 Ping 监控结果（全程）")
        lines.append("=" * 60)
        for target, val in ping_monitor_results.items():
            if val.get("sent", 0) == 0 and val.get("received", 0) == 0:
                lines.append(f"  ✅ {_pad(target, 24)}─ 无数据")
                continue
            if "avg_ms" not in val and val.get("sent", 0) > 0:
                lines.append(f"  ❌ {_pad(target, 24)}全部超时 ({val['sent']} 包无响应)")
                continue
            loss_icon = "✅" if val.get("loss_pct", 100) < 2 else "⚠️" if val.get("loss_pct", 100) < 10 else "❌"
            info = (
                f"发送={val.get('sent', 0):<4}  "
                f"接收={val.get('received', 0):<4}  "
                f"丢包={val.get('loss_pct', '?')}%"
            )
            if "avg_ms" in val:
                info += f"  延迟={val['min_ms']}~{val['avg_ms']}~{val['max_ms']}ms"
            if "jitter_ms" in val:
                info += f"  抖动={val['jitter_ms']}ms"
            if "stddev_ms" in val:
                info += f"  标准差={val['stddev_ms']}ms"
            lines.append(f"  {loss_icon} {_pad(target, 24)}{info}")
        lines.append("")
        lines.append("=" * 60)

    # ---- 周期性全量网络测试汇总 ----
    if periodic_net_results:
        lines.append("")
        lines.append(f"🔄 周期性网络测试汇总（共 {len(periodic_net_results)} 轮）")
        lines.append("=" * 60)

        for i, run_result in enumerate(periodic_net_results, 1):
            lines.append(f"\n  第 {i} 轮:")
            # Ping 汇总
            if "多目标 Ping 测试" in run_result:
                summary = run_result["多目标 Ping 测试"].get("_summary", {})
                if summary:
                    lines.append(
                        f"    Ping:  平均延迟={summary.get('avg_latency_ms', 0):.1f}ms  "
                        f"平均丢包={summary.get('avg_loss_pct', 0):.1f}%  "
                        f"正常={summary.get('targets_ok', 0)}/{summary.get('targets_total', 0)}"
                    )
            # 下载速度汇总
            if "下载速度测试" in run_result:
                speeds = [v.get("speed_mbps", 0) for v in run_result["下载速度测试"].values()
                          if "speed_mbps" in v]
                if speeds:
                    lines.append(f"    下载:  平均={sum(speeds)/len(speeds):.2f} Mbps")
            # 上传速度汇总
            if "上传速度测试" in run_result:
                speeds = [v.get("speed_mbps", 0) for v in run_result["上传速度测试"].values()
                          if "speed_mbps" in v]
                if speeds:
                    lines.append(f"    上传:  平均={sum(speeds)/len(speeds):.2f} Mbps")

        # 趋势分析
        if len(periodic_net_results) >= 2:
            lines.append("")
            lines.append("  📊 趋势分析:")
            first_ping = periodic_net_results[0].get("多目标 Ping 测试", {}).get("_summary", {})
            last_ping = periodic_net_results[-1].get("多目标 Ping 测试", {}).get("_summary", {})
            if first_ping and last_ping:
                lat_diff = last_ping.get("avg_latency_ms", 0) - first_ping.get("avg_latency_ms", 0)
                loss_diff = last_ping.get("avg_loss_pct", 0) - first_ping.get("avg_loss_pct", 0)
                lat_arrow = "↑" if lat_diff > 0 else "↓" if lat_diff < 0 else "→"
                loss_arrow = "↑" if loss_diff > 0 else "↓" if loss_diff < 0 else "→"
                lines.append(
                    f"    延迟变化: {lat_arrow} {abs(lat_diff):.1f}ms  "
                    f"({first_ping.get('avg_latency_ms', 0):.1f} → {last_ping.get('avg_latency_ms', 0):.1f})"
                )
                lines.append(
                    f"    丢包变化: {loss_arrow} {abs(loss_diff):.1f}%  "
                    f"({first_ping.get('avg_loss_pct', 0):.1f}% → {last_ping.get('avg_loss_pct', 0):.1f}%)"
                )

            # 各轮丢包对比
            all_losses = []
            for r in periodic_net_results:
                s = r.get("多目标 Ping 测试", {}).get("_summary", {})
                if s:
                    all_losses.append(s.get("avg_loss_pct", 0))
            if all_losses:
                worst = max(all_losses)
                best = min(all_losses)
                lines.append(f"    丢包范围: {best:.1f}% ~ {worst:.1f}%  (全 {len(all_losses)} 轮)")

        lines.append("")
        lines.append("=" * 60)

    # ---- 兼容旧版 ping_stats ----
    elif ping_stats:
        lines.append("")
        lines.append("网络测试结果 (ping)")
        lines.append("-" * 40)
        for key, label in [
            ("loss", "丢包率"), ("avg", "平均延迟"),
            ("min", "最小延迟"), ("max", "最大延迟"), ("mdev", "标准差"),
        ]:
            if key in ping_stats:
                lines.append(f"{label}: {ping_stats[key]}")
        if "min" in ping_stats and "max" in ping_stats:
            lines.append(f"延迟范围: {ping_stats['min']} ~ {ping_stats['max']}")
        if "raw" in ping_stats:
            lines.append("\n[原始日志最后几行]")
            lines.append(ping_stats["raw"])
        lines.append("=" * 60)

    # JSON 输出
    if json_output_file:
        try:
            json_data = {
                "system_info": sys_info,
                "hardware_stats": {
                    "duration_s": round(duration, 1) if data else 0,
                    "samples": len(data),
                    "raw_samples": data,
                },
                "network_results": net_results,
                "ping_monitor_results": ping_monitor_results,
                "periodic_net_results": periodic_net_results,
                "ping_stats_compat": ping_stats,
            }
            Path(json_output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(json_output_file, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2, default=str)
            log.info("JSON 报告已保存至: %s", json_output_file)
        except Exception as e:
            log.error("保存 JSON 报告失败: %s", e)

    # 输出
    for line in lines:
        print(line)

    if output_file:
        try:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            log.info("报告已保存至: %s", output_file)
        except Exception as e:
            log.error("保存报告失败: %s", e)


# ============================
# 压力测试主流程
# ============================

def run_stress_test(run_cpu=True, run_gpu=True, net_config=None, duration=60, json_output=""):
    """
    运行压力测试并收集数据。
    net_config: NetworkTestConfig 实例，None 则跳过网络测试。
    duration: 压测时长（秒），应用于 CPU/GPU 测试。
    """
    print_banner("运行压力测试")

    interrupted = False

    def signal_handler(sig, frame):
        nonlocal interrupted
        log.info("收到中断信号，正在停止...")
        interrupted = True

    signal.signal(signal.SIGINT, signal_handler)

    subprocess.run("clear", shell=True)
    print("=" * 60)
    print("  🖥️  Ubuntu 自动化压力测试脚本")
    print("  ⌨️  按 Ctrl+C 停止测试")
    print("=" * 60)

    # 测试配置
    print("\n📋 测试配置:")
    print(f"  压测时长:                 {duration} 秒")
    print(f"  CPU 压力测试 (stress-ng): {'✅ 运行' if run_cpu else '❌ 跳过'}")
    print(f"  GPU 压力测试 (gpu_burn):  {'✅ 运行' if run_gpu else '❌ 跳过'}")
    print(f"  网络测试:                 {'✅ 运行' if net_config else '❌ 跳过'}")
    if net_config:
        print(f"    ├─ Ping 目标:   {', '.join(net_config.targets[:3])}...")
        print(f"    ├─ Ping 次数:   {net_config.ping_count}/目标")
        print(f"    ├─ DNS 服务器:  {', '.join(net_config.dns_servers[:3])}...")
        print(f"    ├─ HTTP 测试:   {len(net_config.http_urls)} 个站点")
        print(f"    ├─ 下载测试:    {len(net_config.download_urls)} 个源")
        print(f"    └─ 上传测试:    {len(net_config.upload_urls)} 个目标 ({net_config.upload_size_mb} MB)")
    print("=" * 60)

    # 初始化所有状态变量（确保 finally 块安全引用）
    monitor_procs: list = []
    cpu_proc = None
    gpu_proc = None
    net_results = None

    # GUI + 监控窗口
    fix_gui_env()
    _, _monitors = open_monitors(None)
    if _monitors:
        monitor_procs = _monitors

    # --- CPU ---
    if run_cpu:    ######################## 修改Stress-ng测试参数 ########################
        if shutil.which("stress-ng"):
            log.info("启动 stress-ng CPU 测试...")
            cpu_cmd = [
                "stress-ng", "--cpu", "0", "--vm", "4", "--vm-bytes", "90%",
                "--hdd", "4", "--hdd-bytes", "256M", "--hdd-write-size", "4K",
                "--metrics-brief", "--timeout", f"{duration}s",
            ]
            cpu_proc = subprocess.Popen(cpu_cmd,
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL)
            log.info("CPU 测试已启动")
        else:
            log.warning("stress-ng 未安装，跳过 CPU 测试")
            run_cpu = False

    # --- GPU ---
    if run_gpu:
        if shutil.which("nvidia-smi"):
            gpu_burn_path = find_gpu_burn()
            if gpu_burn_path:
                work_dir = os.path.dirname(gpu_burn_path)
                ptx_local = os.path.join(work_dir, "compare.ptx")
                if not os.path.exists(ptx_local):
                    ptx_system = "/usr/local/bin/compare.ptx"
                    if os.path.exists(ptx_system):
                        shutil.copy(ptx_system, ptx_local)
                        log.info("已复制 compare.ptx")
                try:
                    gpu_proc = subprocess.Popen(
                        [gpu_burn_path, str(duration)], cwd=work_dir,
                        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                    )
                    log.info("GPU 测试已启动")
                except Exception as e:
                    log.error("GPU 测试启动失败: %s", e)
                    run_gpu = False
            else:
                log.warning("未找到 gpu_burn，跳过 GPU 测试")
                run_gpu = False
        else:
            log.warning("NVIDIA 驱动未安装，跳过 GPU 测试")
            run_gpu = False

    # --- 网络测试 ---
    # 1) 首轮全量网络测试（线程）
    net_thread = None
    if net_config:
        def _run_net():
            nonlocal net_results
            tester = NetworkTester(net_config)
            net_results = tester.run_all()
        net_thread = threading.Thread(target=_run_net, daemon=True)
        net_thread.start()
        log.info("首轮网络测试已在后台启动")

    # 2) 持续 ping 监控（覆盖整个压测期间）
    ping_monitor: Optional[ContinuousPingMonitor] = None
    ping_monitor_results: Dict = {}
    if net_config:
        ping_targets = net_config.targets[:4]  # 最多 4 个目标
        ping_monitor = ContinuousPingMonitor(
            targets=ping_targets,
            interval=net_config.continuous_ping_interval,
        )
        ping_monitor.start()

    # 3) 周期性全量网络测试
    periodic_net_results: List[Dict] = []
    periodic_interval = net_config.periodic_test_interval if net_config else 0
    next_periodic_time = time.time() + periodic_interval if (net_config and periodic_interval > 0 and duration > periodic_interval) else 0
    periodic_running = False

    def _run_periodic_test():
        """在后台跑一轮全量网络测试。"""
        nonlocal periodic_running, periodic_net_results
        try:
            tester = NetworkTester(net_config)
            result = tester.run_all()
            periodic_net_results.append(result)
            run_num = len(periodic_net_results)
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"  ✅ 第 {run_num} 轮周期网络测试完成 [{ts}]")
            if net_config:
                net_config.periodic_test_count = run_num
        except Exception as e:
            log.error("周期网络测试异常: %s", e)
        finally:
            periodic_running = False

    if not run_cpu and not run_gpu and not net_config:
        print("\n⚠️  所有测试都被跳过，没有测试可运行！")
        return

    print("\n✅ 所有测试已启动")
    print("=" * 60)

    # 打印各测试启动状态（仅一次）
    if cpu_proc:
        print("  🔄 CPU 压力测试: 运行中")
    if gpu_proc:
        print("  🔄 GPU 压力测试: 运行中")
    if net_config:
        print("  🔄 首轮网络测试: 运行中")
        if ping_monitor:
            targets_str = ", ".join(ping_monitor.targets)
            print(f"  🔄 持续 Ping 监控: 运行中 (间隔 {net_config.continuous_ping_interval}s, 目标: {targets_str})")
        if periodic_interval > 0 and duration > periodic_interval:
            print(f"  🔄 周期网络测试: 每 {periodic_interval // 60} 分钟一轮")
    print()

    # --- 数据采集循环 ---
    data: List[Dict] = []
    sample_interval = 5
    last_sample = time.time()
    start_time = time.time()
    test_timeout = duration + 30  # 压测时长 + 30 秒缓冲防止卡死
    cpu_done_printed = (cpu_proc is None)
    gpu_done_printed = (gpu_proc is None)
    net_done_printed = (net_config is None)

    try:
        while True:
            if interrupted:
                log.info("用户中断，终止测试...")
                break

            now = time.time()

            # 超时保护
            if now - start_time > test_timeout:
                log.warning("测试超时 (%ds)，强制终止", test_timeout)
                break

            # 采样
            if now - last_sample >= sample_interval:
                stats = collect_stats()
                if stats:
                    data.append(stats)
                last_sample = now

            # 检查压力测试进程状态，仅在状态变化时输出
            for label, proc_ref in [("CPU", "cpu"), ("GPU", "gpu")]:
                proc = cpu_proc if proc_ref == "cpu" else gpu_proc
                done_flag = cpu_done_printed if proc_ref == "cpu" else gpu_done_printed
                if proc is None or done_flag:
                    continue
                ret = proc.poll()
                if ret is None:
                    continue
                if ret == 0:
                    print(f"  ✅ {label} 压力测试完成")
                else:
                    print(f"  ⚠️  {label} 压力测试异常退出 (rc={ret})")
                if proc_ref == "cpu":
                    cpu_done_printed = True
                    cpu_proc = None
                else:
                    gpu_done_printed = True
                    gpu_proc = None

            # 网络测试是否完成（首轮）
            if not net_done_printed and net_thread and not net_thread.is_alive():
                print("  ✅ 首轮网络测试完成")
                net_done_printed = True

            # 触发周期性全量网络测试
            if next_periodic_time > 0 and now >= next_periodic_time and not periodic_running:
                periodic_running = True
                run_num = len(periodic_net_results) + 1
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"  🔄 第 {run_num} 轮周期网络测试开始 [{ts}]")
                t = threading.Thread(target=_run_periodic_test, daemon=True)
                t.start()
                next_periodic_time = now + periodic_interval

            # 判断是否所有测试完成
            net_done = net_done_printed or (net_thread is None)
            cpu_gpu_done = (cpu_proc is None) and (gpu_proc is None)
            periodic_done = (next_periodic_time == 0) or (now >= start_time + duration - 5)

            if cpu_gpu_done and net_done and not periodic_running and periodic_done:
                print("\n🏁 所有测试均已结束")
                break

            time.sleep(1)

    except Exception as e:
        log.error("测试异常: %s", e)

    finally:
        # 终止压力测试子进程
        for proc in (cpu_proc, gpu_proc):
            if proc and proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()

        # 终止监控窗口
        for proc in monitor_procs:
            if proc and proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except Exception:
                    proc.kill()
        log.info("监控窗口已关闭")

        # 等待首轮网络测试线程完成（最多再等 30 秒）
        if net_thread and net_thread.is_alive():
            log.info("等待首轮网络测试完成...")
            net_thread.join(timeout=30)

        # 等待周期网络测试完成（最多再等 60 秒）
        if periodic_running:
            log.info("等待周期网络测试完成...")
            for _ in range(60):
                if not periodic_running:
                    break
                time.sleep(1)

        # 停止持续 ping 监控并收集结果
        if ping_monitor:
            log.info("停止持续 Ping 监控...")
            ping_monitor_results = ping_monitor.stop()

        # 生成报告
        report_path = os.path.join(
            cfg.desktop_path,
            f"压力测试报告_{time.strftime('%Y%m%d_%H%M%S')}.txt",
        )
        generate_report(
            data,
            net_results=net_results,
            output_file=report_path,
            ping_monitor_results=ping_monitor_results,
            periodic_net_results=periodic_net_results,
            json_output_file=json_output,
        )
        sys.exit(0)


# ============================
# 安装功能模块
# ============================

def update_system():
    """更新系统并安装基础依赖。"""
    print_banner("1. 更新系统")
    run_command("sudo apt update", fatal=True)
    run_command("sudo apt upgrade -y")
    run_command(
        "sudo apt install -y build-essential git wget curl "
        "software-properties-common xterm sysstat",
        fatal=True,
    )


def detect_gpu():
    """检测 GPU 类型。"""
    print_banner("2. 检测 GPU 硬件")
    gpu_info = {"nvidia": False, "amd": False, "intel": False}
    r = subprocess.run("lspci | grep -i nvidia", shell=True,
                       capture_output=True, text=True)
    if r.stdout:
        log.info("找到 NVIDIA 显卡: %s", r.stdout.strip())
        gpu_info["nvidia"] = True
    return gpu_info


def install_nvidia_driver():
    """安装 NVIDIA 驱动并可选重启。"""
    print_banner("安装 NVIDIA 驱动")
    run_command("sudo add-apt-repository ppa:graphics-drivers/ppa -y")
    run_command("sudo apt update")
    stdout, _ = run_command("ubuntu-drivers devices", capture_output=True)
    if "recommended" in stdout:
        run_command("sudo ubuntu-drivers autoinstall") ######################## 修改NVIDIA驱动 ########################
    else:
        run_command("sudo apt install -y nvidia-driver-590") ######################## 修改NVIDIA驱动 ########################

    if yes_no_prompt("是否安装 CUDA 工具包？(推荐用于 GPU 计算)", "y"):
        log.info("安装 CUDA 工具包...")
        run_command(
            "wget https://developer.download.nvidia.com/compute/cuda/"
            "repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
        )
        run_command("sudo dpkg -i cuda-keyring_1.1-1_all.deb") ######################## 修改CUDA驱动 ########################
        run_command("sudo apt update")
        run_command("sudo apt install -y cuda-toolkit-13-1")

    print("✓ NVIDIA 驱动安装完成")
    if yes_no_prompt("是否立即重启系统？", "y"):
        for i in range(10, 0, -1):
            print(f"\r{i}秒后重启...", end="", flush=True)
            time.sleep(1)
        print()
        run_command("sudo reboot", fatal=True)
    else:
        print("请手动重启系统以完成驱动安装")


def install_stress_ng():
    """安装 stress-ng。"""
    print_banner("3. 安装 stress-ng")
    run_command("sudo apt install -y stress-ng htop", fatal=True)
    stdout, _ = run_command("stress-ng --version", capture_output=True)
    if stdout:
        log.info("stress-ng 版本: %s", stdout.split()[1] if len(stdout.split()) > 1 else "未知")


def install_gpu_burn():
    """安装 GPU-burn。"""
    print_banner("4. 安装 GPU-Burn")
    work_dir = os.path.join(cfg.home_dir, "gpu-burn")

    if os.path.exists(work_dir):
        log.info("gpu-burn 目录已存在，跳过 clone")
    else:
        run_command(f"git clone https://github.com/wilicc/gpu-burn {work_dir}")

    run_command(f"cd {work_dir} && make")
    run_command(f"sudo cp {work_dir}/gpu_burn /usr/local/bin/")
    run_command("sudo chmod +x /usr/local/bin/gpu_burn")
    run_command(f"sudo cp {work_dir}/compare.ptx /usr/local/bin/ 2>/dev/null || true")
    log.info("GPU-burn 安装完成")


def full_install():
    """完整安装流程。"""
    print_banner("Ubuntu 自动压力测试安装程序")

    if os.geteuid() != 0:
        print("错误：请使用 sudo 运行此脚本")
        print(f"正确命令: sudo python3 {sys.argv[0]}")
        sys.exit(1)

    try:
        update_system()
        gpu_info = detect_gpu()

        # NVIDIA 驱动
        if gpu_info["nvidia"]:
            print_banner("检测到 NVIDIA 显卡")
            if check_nvidia_driver_installed():
                print("✓ NVIDIA 驱动已安装")
            elif yes_no_prompt("是否安装 NVIDIA 驱动？", "y"):
                install_nvidia_driver()
            else:
                print("跳过 NVIDIA 驱动安装")

        install_stress_ng()

        if gpu_info["nvidia"] and yes_no_prompt("是否安装 GPU-burn?", "y"):
            install_gpu_burn()

        print_banner("安装完成")
        print("✓ 所有必要的软件已安装完成\n")

        # 询问是否运行测试
        if yes_no_prompt("是否立即运行压力测试？", "y"):
            run_cpu, run_gpu, net_config, duration = _ask_test_options()
            if not run_cpu and not run_gpu and not net_config:
                print("\n⚠️  你没有选择任何测试项目，测试已取消。")
                print(f"  可以稍后手动运行: sudo python3 {sys.argv[0]} --test")
                sys.exit(0)
            run_stress_test(run_cpu=run_cpu, run_gpu=run_gpu, net_config=net_config, duration=duration)
        else:
            print(f"\n📋 你可以稍后手动运行测试:")
            print(f"  1. 运行全部测试:  sudo python3 {sys.argv[0]} --test")
            print(f"  2. 只运行 CPU:    sudo python3 {sys.argv[0]} --test --cpu")
            print(f"  3. 只运行 GPU:    sudo python3 {sys.argv[0]} --test --gpu")
            print(f"  4. 只运行网络:    sudo python3 {sys.argv[0]} --test --net")
            print(f"  5. 查看 GPU 状态: nvidia-smi")
            print(f"  6. 运行 CPU 测试: stress-ng --cpu 0 --timeout 60s")
            print(f"  7. 指定压测时长:  sudo python3 {sys.argv[0]} --test --duration 300")

    except KeyboardInterrupt:
        print("\n安装被用户中断")
        sys.exit(1)
    except Exception as e:
        log.error("安装错误: %s", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _ask_test_options():
    """交互式询问要运行哪些测试，返回 (run_cpu, run_gpu, net_config, duration)。"""
    run_cpu = False
    run_gpu = False
    net_config = None
    duration = 60

    if shutil.which("stress-ng"):
        run_cpu = yes_no_prompt("是否运行 CPU 压力测试 (stress-ng)?", "y")
    else:
        print("⚠️  stress-ng 未安装，跳过 CPU 测试")

    nvidia_ok = shutil.which("nvidia-smi") or os.path.exists("/usr/bin/nvidia-smi")
    gpu_burn_ok = find_gpu_burn() is not None
    if nvidia_ok and gpu_burn_ok:
        run_gpu = yes_no_prompt("是否运行 GPU 压力测试 (gpu_burn)?", "y")
    else:
        print("⚠️  GPU 测试条件不满足，跳过")
        if not nvidia_ok:
            print("   - 未找到 nvidia-smi")
        if not gpu_burn_ok:
            print("   - 未找到 gpu_burn")

    # 压测时间设置（CPU/GPU 共用）
    if run_cpu or run_gpu:
        try:
            dur_input = input(f"  请输入压测时间（秒）[{duration}]: ").strip()
            if dur_input:
                duration = max(1, int(dur_input))
        except ValueError:
            pass

    if yes_no_prompt("是否运行网络测试？（Ping/DNS/HTTP/下载）", "n"):
        net_config = NetworkTestConfig()

        # 自定义 Ping 目标
        if yes_no_prompt(f"  使用默认 Ping 目标？({', '.join(net_config.targets[:3])}...)", "y"):
            pass  # 保持默认
        else:
            targets_input = input("  请输入 Ping 目标（逗号分隔）: ").strip()
            if targets_input:
                net_config.targets = [t.strip() for t in targets_input.split(",") if t.strip()]

        # Ping 次数
        try:
            cnt = input(f"  每个目标 Ping 次数 [{net_config.ping_count}]: ").strip()
            if cnt:
                net_config.ping_count = max(1, int(cnt))
        except ValueError:
            pass

        # iperf3 服务器（可选）
        if shutil.which("iperf3"):
            iperf_server = input("  iperf3 服务器地址（留空跳过）: ").strip()
            if iperf_server:
                net_config.iperf3_server = iperf_server
                try:
                    dur = input(f"  iperf3 测试时长（秒）[{net_config.iperf3_duration}]: ").strip()
                    if dur:
                        net_config.iperf3_duration = max(1, int(dur))
                except ValueError:
                    pass

        # 周期性全量网络测试间隔（压测时长 > 测试间隔时才有意义）
        if duration > 60:
            try:
                default_min = net_config.periodic_test_interval // 60
                pi = input(f"  周期性全量网络测试间隔（分钟）[{default_min}]: ").strip()
                if pi:
                    net_config.periodic_test_interval = max(60, int(pi) * 60)
                elif duration <= net_config.periodic_test_interval:
                    print(f"  ⚠️  压测时长({duration}s) ≤ 测试间隔({net_config.periodic_test_interval}s)，将禁用周期测试")
                    net_config.periodic_test_interval = 0
            except ValueError:
                pass

    return run_cpu, run_gpu, net_config, duration


# ============================
# 主入口
# ============================

def main():
    """主函数。"""
    parser = argparse.ArgumentParser(
        description="Ubuntu 自动化压力测试安装脚本（优化版）",
    )
    parser.add_argument("--test", action="store_true",
                        help="直接运行压力测试（默认运行 CPU+GPU+网络；可用 --cpu/--gpu/--net 精确控制）")
    parser.add_argument("--cpu", action="store_true",
                        help="运行 CPU 压力测试 (stress-ng)")
    parser.add_argument("--gpu", action="store_true",
                        help="运行 GPU 压力测试 (gpu_burn)")
    parser.add_argument("--net", action="store_true",
                        help="运行专业网络测试（Ping/DNS/HTTP/下载/iperf3）")
    parser.add_argument("--net-targets", type=str, default="",
                        help="Ping 目标（逗号分隔），例如: 223.5.5.5,baidu.com")
    parser.add_argument("--net-ping-count", type=int, default=20,
                        help="每个 Ping 目标的测试次数，默认 20")
    parser.add_argument("--download-urls", type=str, default="",
                        help="下载速度测试 URL（逗号分隔），例如: https://example.com/file.bin,https://other.com/test.bin")
    parser.add_argument("--upload-urls", type=str, default="",
                        help="上传速度测试 URL（逗号分隔），默认使用 Cloudflare")
    parser.add_argument("--upload-size", type=int, default=10,
                        help="上传测试数据大小（MB），默认 10")
    parser.add_argument("--iperf3-server", type=str, default="",
                        help="iperf3 服务器地址（指定后自动运行带宽测试）")
    parser.add_argument("--iperf3-duration", type=int, default=10,
                        help="iperf3 测试时长（秒），默认 10")
    parser.add_argument("--duration", type=int, default=60,
                        help="CPU/GPU 压测时长（秒），默认 60")
    parser.add_argument("--periodic-interval", type=int, default=1800,
                        help="周期性全量网络测试间隔（秒），默认 1800（30分钟）；设为 0 禁用")
    parser.add_argument("--ping-interval", type=float, default=5.0,
                        help="持续 Ping 间隔（秒），默认 5")
    parser.add_argument("--monitor", action="store_true",
                        help="只运行命令行监控")
    parser.add_argument("--json-output", type=str, default="",
                        help="将测试结果以 JSON 格式保存到指定路径")
    parser.add_argument("--install", action="store_true",
                        help="只运行安装流程（不测试）")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="显示调试日志")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 初始化全局配置
    global cfg
    cfg = Config()

    log.info("当前用户: %s  HOME: %s", os.getlogin(), cfg.home_dir)

    if args.monitor:
        command_line_monitor()
    elif args.install:
        full_install()
    elif args.test:
        # 根据用户指定的参数决定运行什么
        has_explicit = args.cpu or args.gpu or args.net
        run_cpu = args.cpu if has_explicit else True
        run_gpu = args.gpu if has_explicit else True

        net_config = None
        if args.net or (not has_explicit):
            # 默认就跑网络测试；或用户显式指定 --net
            net_config = NetworkTestConfig()
            if args.net_targets:
                net_config.targets = [t.strip() for t in args.net_targets.split(",") if t.strip()]
            if args.download_urls:
                net_config.download_urls = [u.strip() for u in args.download_urls.split(",") if u.strip()]
            if args.upload_urls:
                net_config.upload_urls = [u.strip() for u in args.upload_urls.split(",") if u.strip()]
            if args.upload_size:
                net_config.upload_size_mb = max(1, args.upload_size)
            net_config.ping_count = args.net_ping_count
            if args.iperf3_server:
                net_config.iperf3_server = args.iperf3_server
                net_config.iperf3_duration = args.iperf3_duration
            net_config.periodic_test_interval = args.periodic_interval
            net_config.continuous_ping_interval = args.ping_interval
        # 如果用户只指定了 --cpu / --gpu，没指定 --net，则不跑网络
        if has_explicit and not args.net:
            net_config = None

        run_stress_test(run_cpu=run_cpu, run_gpu=run_gpu, net_config=net_config, duration=args.duration, json_output=args.json_output)
    else:
        full_install()


if __name__ == "__main__":
    main()
