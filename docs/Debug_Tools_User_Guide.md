# MOE-FedCL Debug Tools User Guide

## 📖 Overview

This document provides comprehensive usage instructions for the debug tools in the MOE-FedCL project, including `debug_tools.sh` and `simple_debug.sh` scripts, specifically designed for debugging and troubleshooting federated learning systems.

## 🔧 Tools Overview

### 1. `debug_tools.sh` - Complete Debug Toolkit
- **Function**: Comprehensive debugging and monitoring capabilities
- **Features**: Real-time monitoring, deep analysis, loop checking
- **Use Cases**: Deep debugging, real-time monitoring, problem diagnosis

### 2. `simple_debug.sh` - Simplified Debug Tool
- **Function**: Basic one-time check capabilities
- **Features**: No loop output, quick diagnosis, safe and reliable
- **Use Cases**: Daily checks, quick diagnosis, avoiding terminal blocking

## 📋 Feature Comparison

| Feature | debug_tools.sh | simple_debug.sh | Description |
|---------|----------------|-----------------|-------------|
| System Diagnosis | ✅ Detailed | ✅ Quick | Analyze logs and component status |
| Process Monitoring | ✅ Real-time Loop | ✅ Single Check | Check test process status |
| Log Analysis | ✅ Deep Analysis | ✅ Summary View | Analyze errors and warnings |
| Warning Analysis | ✅ | ❌ | Detailed warning cause analysis |
| Blocking Analysis | ✅ | ❌ | Deep analysis of test blocking |
| Process Cleanup | ✅ | ✅ | Force cleanup processes |
| Timeout Testing | ✅ | ❌ | Run tests with timeout |
| Log Monitoring | ✅ Real-time | ❌ | Real-time log change monitoring |
| Emergency Stop | ✅ | ❌ | Stop all loop processes |

## 🚀 Usage Instructions

### debug_tools.sh Usage

#### 1. Interactive Menu Mode
```bash
# Launch interactive menu
./scripts/debug_tools.sh

# Select corresponding function number
1. System Diagnosis
2. Real-time Process Monitoring
3. Analyze Test Logs
4. Run Tests with Timeout
5. Kill All Test Processes
6. Monitor Log File Changes
7. Detailed Warning Analysis
8. Analyze Test Blocking Root Cause
9. Emergency Stop All Debug Scripts
0. Exit
```

#### 2. Direct Command Mode
```bash
# System diagnosis
./scripts/debug_tools.sh fedcl_diagnose

# Analyze warning details
./scripts/debug_tools.sh analyze_warning_details

# Analyze blocking root cause
./scripts/debug_tools.sh analyze_blocking_root_cause

# Monitor test processes (real-time)
./scripts/debug_tools.sh monitor_test_processes

# Cleanup test processes
./scripts/debug_tools.sh kill_test_processes

# Emergency stop all loops
./scripts/debug_tools.sh emergency_stop
```

### simple_debug.sh Usage

#### 1. Interactive Menu Mode
```bash
# Launch simplified menu
./scripts/simple_debug.sh

# Select function
1. Quick System Diagnosis
2. View Log Summary
3. Check Process Status (Single)
4. Cleanup All Processes
0. Exit
```

#### 2. Direct Command Mode
```bash
# Quick diagnosis
./scripts/simple_debug.sh quick_diagnose

# View log summary
./scripts/simple_debug.sh show_log_summary

# Single process check
./scripts/simple_debug.sh check_processes_once

# Cleanup all processes
./scripts/simple_debug.sh cleanup_all
```

## 🎯 Detailed Function Description

### 1. System Diagnosis (fedcl_diagnose)
**Purpose**: Analyze overall federated learning system status
```bash
./scripts/debug_tools.sh fedcl_diagnose
```
**Output Content**:
- Latest experiment log directory
- Error and warning statistics
- Component status (communication components, clients, servers)
- Recent error information

### 2. Detailed Warning Analysis (analyze_warning_details)
**Purpose**: Deep analysis of warning information in logs
```bash
./scripts/debug_tools.sh analyze_warning_details
```
**Analysis Content**:
- Summary of all warning information
- Detailed analysis of round timeout warnings
- Client training activity checks
- Root cause analysis
- Solution recommendations

### 3. Blocking Analysis (analyze_blocking_root_cause)
**Purpose**: Analyze root causes of integration test blocking
```bash
./scripts/debug_tools.sh analyze_blocking_root_cause
```
**Analysis Content**:
- Current active process checks
- Heartbeat thread infinite loop issues
- Incomplete thread cleanup issues
- Test framework problem analysis
- State synchronization issues
- Solution recommendations

### 4. Real-time Monitoring (monitor_test_processes)
**Purpose**: Real-time monitoring of test process status
```bash
./scripts/debug_tools.sh monitor_test_processes
```
**Monitoring Content**:
- Pytest process status
- Python test process status
- Memory usage
- Auto-exit conditions (3 consecutive times with no processes)

**Safety Mechanisms**:
- Ctrl+C signal handling
- Input 'q' or 'quit' to exit
- Automatic timeout exit

### 5. Log Monitoring (monitor_logs)
**Purpose**: Real-time monitoring of log file changes
```bash
./scripts/debug_tools.sh monitor_logs
```
**Features**:
- Monitor all related log files
- Limit display to latest 20 lines
- 60-second automatic timeout
- Safe Ctrl+C exit

### 6. Timeout Testing (timeout_test)
**Purpose**: Run tests with timeout protection
```bash
./scripts/debug_tools.sh timeout_test
```
**Usage Flow**:
1. Input timeout duration (default 30 seconds)
2. Input test command
3. Automatic execution and timeout monitoring

### 7. Process Cleanup Functions
**Normal Cleanup**:
```bash
./scripts/debug_tools.sh kill_test_processes
```

**Emergency Stop**:
```bash
./scripts/debug_tools.sh emergency_stop
```

## 🛡️ Safety Mechanisms

### 1. Loop Control
- **Signal Handling**: Support safe Ctrl+C exit
- **Timeout Mechanism**: Automatic timeout exit from loops
- **User Input**: Support 'q' command exit
- **Auto Exit**: Automatic exit when conditions are met

### 2. Process Management
- **Gentle Termination**: Use TERM signal first
- **Force Termination**: Use KILL signal after delay
- **Process Check**: Confirm process status
- **Cleanup Verification**: Verify cleanup effectiveness

### 3. Error Handling
- **File Check**: Check log file existence
- **Permission Verification**: Check script execution permissions
- **Path Handling**: Use absolute paths to avoid errors

## 🔍 Troubleshooting

### Common Issues and Solutions

#### 1. Terminal Continuous Loop Output
**Problem**: Monitoring functions cause continuous terminal output
**Solutions**:
```bash
# Method 1: Press Ctrl+C
# Method 2: Type 'q' to exit
# Method 3: Emergency stop
./scripts/debug_tools.sh emergency_stop
```

#### 2. Test Processes Cannot Be Cleaned
**Problem**: kill_test_processes ineffective
**Solutions**:
```bash
# Use emergency stop
./scripts/debug_tools.sh emergency_stop

# Or manually find and cleanup
ps aux | grep pytest
kill -9 <pid>
```

#### 3. Permission Issues
**Problem**: Scripts cannot execute
**Solutions**:
```bash
# Add execute permissions
chmod +x ./scripts/debug_tools.sh
chmod +x ./scripts/simple_debug.sh
```

#### 4. Log Directory Not Found
**Problem**: Experiment log directory not found
**Solutions**:
```bash
# Check logs directory
ls -la logs/

# Run a test once to generate logs
python -m pytest tests/test_federation_framework_fixed.py -v -s
```

## 📊 Usage Recommendations

### 1. Daily Use Recommendations
```bash
# Recommended: Daily quick check
./scripts/simple_debug.sh quick_diagnose

# Recommended: View log summary
./scripts/simple_debug.sh show_log_summary
```

### 2. Deep Debugging Recommendations
```bash
# System problem diagnosis
./scripts/debug_tools.sh fedcl_diagnose

# Detailed warning analysis
./scripts/debug_tools.sh analyze_warning_details

# Blocking problem analysis
./scripts/debug_tools.sh analyze_blocking_root_cause
```

### 3. Real-time Monitoring Recommendations
```bash
# Monitor test processes (short-term)
./scripts/debug_tools.sh monitor_test_processes

# Monitor log changes (short-term)
./scripts/debug_tools.sh monitor_logs
```

## 🔧 Custom Extensions

### Adding New Features
1. Add functions to respective scripts
2. Add options to menu
3. Add case branches to main function

### Modifying Timeout Duration
```bash
# Modify log monitoring timeout (default 60 seconds)
# In monitor_logs function
timeout 120 tail -n 20 -f ...

# Modify process monitoring auto-exit condition (default 3 times)
# In monitor_test_processes function
if [[ $count -gt 5 ]]; then
```

## 📝 Log File Description

### Log Directory Structure
```
logs/
└── experiment_YYYYMMDD_HHMMSS/
    ├── federated_training.log     # Main training log
    ├── server_test_server.log     # Server log
    └── clients/
        ├── client_0.log           # Client 0 log
        ├── client_1.log           # Client 1 log
        └── client_2.log           # Client 2 log
```

### Important Log Content
- **ERROR**: System errors, require attention
- **WARNING**: Warning information, usually expected behavior like timeouts
- **heartbeat**: Heartbeat messages, used to check connection status
- **registration**: Client registration messages
- **Round timeout**: Round timeout, normal federated learning behavior

## 🎯 Summary

1. **Daily Use**: Prioritize using `simple_debug.sh` for quick checks
2. **Deep Debugging**: Use specialized analysis functions in `debug_tools.sh`
3. **Real-time Monitoring**: Use real-time monitoring functions for short periods, avoid long-running
4. **Safe Exit**: Always remember Ctrl+C and emergency_stop functions
5. **Problem Diagnosis**: Follow the sequence: System Diagnosis → Warning Analysis → Blocking Analysis

By properly using these debug tools, you can effectively diagnose and resolve various issues in the MOE-FedCL federated learning system.
