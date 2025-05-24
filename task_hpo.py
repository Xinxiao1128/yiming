from clearml import Task
from clearml.automation import DiscreteParameterRange, HyperParameterOptimizer
from clearml.automation.optuna import OptimizerOptuna
import logging
import sys
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

task = Task.init(
    project_name='Agri-Pest-Detection',
    task_name='Step 4 - HPO: Train Model',
    task_type=Task.TaskTypes.optimizer
)

# Step 2 + Step 3 Task ID
STEP2_TASK_ID = '97b92047cc62415e860d96e531e9dbd4'
STEP3_TASK_ID = '8a99258673cd4e02a2b788d5b9437d9a'  

# 支持从命令行参数传入
for arg in sys.argv:
    if arg.startswith("--step3_id="):
        STEP3_TASK_ID = arg.split("=")[1]
if not STEP3_TASK_ID:
    raise ValueError("❌ Missing Step 3 base training task ID")

params = task.connect({
    'processed_dataset_id': STEP2_TASK_ID,
    'base_train_task_id': STEP3_TASK_ID,
    'test_queue': 'pipeline',
    'num_trials': 5,
    'time_limit_minutes': 30
})

# agent
if __name__ == "__main__" and params.get("test_queue") and Task.running_locally():
    task.execute_remotely(queue_name=params["test_queue"])
    
hyper_parameters = [
    DiscreteParameterRange('General/learning_rate', values=[0.0001, 0.0005, 0.001, 0.005]),
    DiscreteParameterRange('General/batch_size', values=[16, 32, 64]),
    DiscreteParameterRange('General/weight_decay', values=[1e-6, 1e-5, 1e-4]),
    DiscreteParameterRange('General/num_epochs', values=[5, 10, 15]),
    DiscreteParameterRange('General/dropout_rate', values=[0.3, 0.4, 0.5, 0.6])
]

optimizer = HyperParameterOptimizer(
    base_task_id=params['base_train_task_id'],
    hyper_parameters=hyper_parameters,
    objective_metric_title='accuracy',       
    objective_metric_series='validation',    
    objective_metric_sign='max',
    optimizer_class=OptimizerOptuna,
    max_number_of_concurrent_tasks=1,
    optimization_time_limit=params['time_limit_minutes'] * 60,
    compute_time_limit=60 * 60,
    total_max_jobs=params['num_trials'],
    min_iteration_per_job=0,
    max_iteration_per_job=9999
)

logger.info("🚀 Starting HPO...")
optimizer.start_locally()
optimizer.wait()

top_experiments = optimizer.get_top_experiments(top_k=5)
logger.info("✅ HPO Finished. Processing results...")

results = []
for task in top_experiments:
    try:
        metrics = task.get_last_scalar_metrics()
        accuracy = metrics.get('accuracy', {}).get('validation', {}).get('last', 0)
        results.append({
            'id': task.id,
            'name': task.name,
            'accuracy': accuracy,
            'params': task.get_parameters().get('General', {})
        })
    except Exception as e:
        logger.warning(f"⚠️ Failed to extract result from task {task.id}: {e}")

results.sort(key=lambda x: x['accuracy'], reverse=True)

from clearml import Task as ClearMLTask

if results:
    best = results[0]
    logger.info(f"🏆 Best Model: {best['accuracy']:.4f} ({best['name']})")

    upload_task = ClearMLTask.create(
        task_name='Step 4 - HPO Artifact Upload',
        project_name='Agri-Pest-Detection',
        task_type=ClearMLTask.TaskTypes.custom
    )

    upload_task.set_comment("Automatically uploading HPO results (best model info)")
    upload_task.connect(best)
    upload_task.mark_started()

    try:
        upload_task.upload_artifact("best_experiment_id", best['id'])
        upload_task.upload_artifact("best_accuracy", best['accuracy'])
        upload_task.upload_artifact("best_hyperparameters", best['params'])
        logger.info("✅ Artifacts uploaded successfully in subtask.")
    except Exception as e:
        logger.warning(f"❌ Failed to upload artifacts: {e}")

    upload_task.close()  
    logger.info(f"📦 Artifacts saved in subtask: {upload_task.artifacts}")

logger.info("✅ Step 4 - HPO completed.")
