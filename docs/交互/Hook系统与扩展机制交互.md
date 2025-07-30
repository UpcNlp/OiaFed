```mermaid
sequenceDiagram
    participant TE as TrainingEngine
    participant HE as HookExecutor
    participant CR as ComponentRegistry
    participant EC as ExecutionContext
    participant MH as MetricsHook
    participant CH as CheckpointHook
    participant CustomH as CustomHook
    participant ML as MetricsLogger
    participant CM as CheckpointManager
    
    Note over TE,CM: Hook注册阶段（系统启动时）
    
    CR->>HE: register_hooks_from_registry()
    HE->>CR: get_hooks("before_task")
    CR-->>HE: [MetricsHook(priority=1), CustomHook(priority=2)]
    
    HE->>CR: get_hooks("after_task") 
    CR-->>HE: [CheckpointHook(priority=1), MetricsHook(priority=2)]
    
    HE->>CR: get_hooks("before_batch")
    CR-->>HE: [MetricsHook(priority=1)]
    
    Note over TE,CM: 训练任务开始
    
    TE->>HE: execute_hooks("before_task", context, task_id=1, task_data=data)
    
    Note over HE,CustomH: before_task Hook执行（按优先级）
    HE->>HE: sort_hooks_by_priority()
    
    loop 按优先级执行Hook
        HE->>MH: execute(context, task_id=1, task_data=data)
        MH->>EC: get_state("experiment_config")
        EC-->>MH: config_data
        MH->>ML: log_task_start(task_id, timestamp)
        ML-->>MH: logged
        MH-->>HE: hook_result_1
        
        HE->>CustomH: execute(context, task_id=1, task_data=data)
        CustomH->>EC: get_state("custom_metrics")
        EC-->>CustomH: custom_data
        CustomH->>CustomH: perform_custom_logic()
        CustomH-->>HE: hook_result_2
    end
    
    HE-->>TE: [hook_result_1, hook_result_2]
    
    Note over TE,CM: 训练循环中的Hook
    
    loop 训练批次
        TE->>HE: execute_hooks("before_batch", context, batch_data=batch)
        
        HE->>MH: execute(context, batch_data=batch)
        MH->>EC: set_state("current_batch", batch_info)
        MH->>ML: log_batch_metrics(batch_size, learning_rate)
        MH-->>HE: batch_hook_result
        
        HE-->>TE: [batch_hook_result]
        
        TE->>TE: process_batch(batch)
        
        Note over TE: 批次处理完成
    end
    
    Note over TE,CM: 任务完成后的Hook
    
    TE->>HE: execute_hooks("after_task", context, task_id=1, results=task_results)
    
    loop 按优先级执行Hook
        HE->>CH: execute(context, task_id=1, results=task_results)
        CH->>EC: get_state("checkpoint_config")
        EC-->>CH: checkpoint_settings
        
        alt 需要保存检查点
            CH->>CM: save_checkpoint(task_id, model_state, results)
            CM->>CM: create_checkpoint_file()
            CM-->>CH: checkpoint_saved
        else 跳过检查点
            CH->>CH: log_checkpoint_skipped()
        end
        
        CH-->>HE: checkpoint_hook_result
        
        HE->>MH: execute(context, task_id=1, results=task_results)
        MH->>ML: log_task_completion(task_id, metrics)
        MH->>EC: set_state("task_1_metrics", results.metrics)
        MH-->>HE: metrics_hook_result
    end
    
    HE-->>TE: [checkpoint_hook_result, metrics_hook_result]
    
    Note over TE,CM: 动态Hook注册场景
    
    rect rgb(240, 240, 255)
        Note over TE,CustomH: 运行时添加新Hook
        TE->>HE: register_runtime_hook("custom_phase", new_hook)
        HE->>CR: add_hook("custom_phase", new_hook)
        CR-->>HE: hook_registered
        
        TE->>HE: execute_hooks("custom_phase", context, custom_data=data)
        HE->>CustomH: execute(context, custom_data=data)
        CustomH-->>HE: custom_result
        HE-->>TE: [custom_result]
    end
    
    Note over TE,CM: 错误处理Hook
    
    rect rgb(255, 240, 240)
        Note over HE,CustomH: Hook执行失败处理
        HE->>CustomH: execute(context, invalid_data=None)
        CustomH-xCustomH: execution_error
        CustomH-->>HE: HookExecutionError
        
        HE->>HE: handle_hook_error(CustomHook, error)
        HE->>ML: log_hook_error(hook_name, error_details)
        
        alt 错误策略：继续执行
            HE->>HE: continue_with_remaining_hooks()
            HE->>MH: execute(context, ...)
            MH-->>HE: success_result
        else 错误策略：停止执行
            HE-->>TE: HookExecutionFailed([error_details])
        end
    end
    
    Note over TE,CM: Hook通信场景
    
    rect rgb(240, 255, 240)
        Note over MH,CM: Hook间通信
        MH->>EC: set_state("metrics_computed", True)
        EC->>EC: notify_state_change("metrics_computed")
        
        CH->>EC: get_state("metrics_computed")
        EC-->>CH: True
        CH->>CH: trigger_conditional_checkpoint()
        CH->>CM: save_metrics_triggered_checkpoint()
        CM-->>CH: checkpoint_complete
    end
    
    Note over TE,CM: 训练任务完全结束