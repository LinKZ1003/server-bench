# 🔥 autoOCMIMO — Ubuntu 服务器压力测试 & 网络质量一体化脚本

一站式服务器稳定性验证工具：CPU/GPU 压力测试 + 实时监控 + 专业网络质量检测，压测期间全程监控网络状态，自动生成综合报告。

## ✨ 功能特性

### 🖥️ 压力测试
- **CPU 压测**：stress-ng 全核满载 + 内存压力
- **GPU 压测**：gpu_burn CUDA 烧机测试（需 NVIDIA 显卡）
- **可配置时长**：自由控制压测时间

### 📊 实时监控
- **系统资源采集**：CPU 占用、内存使用、GPU 状态
- **GUI 监控窗口**：自动打开 htop / nvidia-smi / ping 监控终端
- **命令行模式**：纯终端实时刷新，无需桌面环境

### 🌐 网络测试套件
- **Ping**：丢包率、延迟统计
- **MTR**：路由追踪、逐跳丢包分析
- **TCPing**：TCP 连接响应时间
- **iperf3**：带宽、抖动、重传率
- **DNS**：DNS 解析速度测试
- **HTTP**：状态码、TTFB、各阶段耗时
- **持续 Ping 监控**：压测全程后台运行，记录网络波动
- **周期性网络测试**：压测期间定时全量网络检测

### 🛠️ 自动化
- **依赖自动安装**：stress-ng、gpu_burn、NVIDIA 驱动
- **硬件信息采集**：CPU、内存、磁盘、GPU、系统版本自动识别
- **报告生成**：文本报告 + JSON 报告，自动保存到桌面

## 📋 环境要求

- **系统**：Ubuntu 20.04+ / Debian 11+
- **Python**：3.8+
- **权限**：`sudo`（安装依赖、运行压测）
- **GPU 测试**：NVIDIA 显卡 + CUDA 驱动（可选，无 GPU 自动跳过）
- **可选依赖**：`psutil`（更精确的系统监控，没有也能跑）

## 🚀 快速开始

### 一键全装 + 全测

```bash
git clone https://github.com/LinKZ1003/server-bench.git
cd server-bench
sudo python3 autoOCMIMO.py
```

### 交互模式（推荐新手）

```bash
sudo python3 autoOCMIMO.py
```

按提示选择要测试的项目和参数。

### 仅安装依赖

```bash
sudo python3 autoOCMIMO.py --install
```

### 仅运行网络测试（不压测）

```bash
sudo python3 autoOCMIMO.py --no-cpu --no-gpu --duration 30
```

## 💻 使用方式

### CLI 模式

```bash
# 全部测试，压测 120 秒
sudo python3 autoOCMIMO.py --all --duration 120

# 只压 CPU
sudo python3 autoOCMIMO.py --cpu --no-gpu --duration 60

# 只压 GPU
sudo python3 autoOCMIMO.py --gpu --no-cpu --duration 60

# 纯网络测试（不压硬件）
sudo python3 autoOCMIMO.py --no-cpu --no-gpu

# 指定 iperf3 服务器
sudo python3 autoOCMIMO.py --all --iperf3-server 192.168.1.100

# 输出 JSON 报告
sudo python3 autoOCMIMO.py --all --json-output /tmp/report.json

# 命令行监控模式（不压测，只看状态）
sudo python3 autoOCMIMO.py --monitor
```

## 📋 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--all` | 运行所有测试 | - |
| `--install` | 安装所有依赖 | - |
| `--monitor` | 纯命令行监控模式 | - |
| `--cpu` | 运行 CPU 压测 | - |
| `--no-cpu` | 跳过 CPU 压测 | - |
| `--gpu` | 运行 GPU 压测 | - |
| `--no-gpu` | 跳过 GPU 压测 | - |
| `--duration N` | 压测时长（秒） | `60` |
| `--iperf3-server` | iperf3 服务端地址 | - |
| `--iperf3-duration` | iperf3 测试时长（秒） | `10` |
| `--json-output` | JSON 报告输出路径 | - |

## 📊 报告示例

```
============================================================
 压力测试报告
============================================================

📋 测试环境信息
  测试时间:      2026-04-16 14:30:00
  系统:          Ubuntu 22.04.3 LTS
  内核:          5.15.0-91-generic
  CPU:           Intel(R) Xeon(R) E5-2680 v4 @ 2.40GHz (28 核)
  内存:          64 GB
  GPU:           NVIDIA GeForce RTX 3090 (24 GB)

🖥️ CPU 压力测试
  测试工具:      stress-ng
  测试时长:      60s
  最高温度:      78°C
  平均负载:      27.5

🎮 GPU 压力测试
  测试工具:      gpu_burn
  测试时长:      60s
  最高温度:      83°C
  最高功耗:      350W

🌐 网络测试报告
  📡 Ping → 8.8.8.8
    丢包率: 0%
    平均延迟: 35.2ms ✅

  📡 持续 Ping 监控（全程）
    运行时长: 62s
    总包数: 12 | 丢包: 0 | 丢包率: 0.0% ✅

  🌐 HTTP → https://www.baidu.com
    HTTP 200 ✅
    TTFB: 45ms ✅

✅ 所有测试通过
```

## 🏗️ 项目结构

```
autoOCMIMO.py          # 主脚本（全部功能）
README.md              # 本文件
LICENSE                # MIT 开源协议
.gitignore             # Git 忽略规则
```

## ⚠️ 注意事项

- CPU/GPU 压测会产生**高负载**，生产环境慎用
- GPU 测试需要 **NVIDIA 显卡 + 驱动**，无 GPU 会自动跳过
- iperf3 测试需要**对端有 iperf3 服务端**运行
- 建议在**测试环境**或**新购服务器验货**时使用
- 报告默认保存到 `~/Desktop/`

## 🤝 贡献

欢迎 Issue 和 PR！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/xxx`)
3. 提交更改 (`git commit -m 'Add xxx'`)
4. 推送分支 (`git push origin feature/xxx`)
5. 创建 Pull Request

## 📄 许可证

[MIT](LICENSE)
