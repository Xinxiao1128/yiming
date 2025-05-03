from clearml import Task
from clearml.automation import PipelineController

def run_pipeline():
    pipe = PipelineController(
        name="Agri Pest Detection Pipeline",
        project="Agri-Pest-Detection",
        version="1.0",
        add_pipeline_tags=True
    )

    pipe.set_default_execution_queue("default")

    pipe.add_step(
        name="stage_data",
        base_task_project="Agri-Pest-Detection",
        base_task_name="Step 1 - Upload archive.zip (Colab)",
        parameter_override={},
    )

    pipe.add_step(
        name="stage_process",
        parents=["stage_data"],
        base_task_project="Agri-Pest-Detection",
        base_task_name="Step 2 - Preprocessing",
        parameter_override={
            "dataset_task_id": "${stage_data.id}",
            "image_size": 256,
            "test_size": 0.2,
            "random_state": 42
        },
    )

    pipe.add_step(
        name="stage_train",
        parents=["stage_process"],
        base_task_project="Agri-Pest-Detection",
        base_task_name="Step 3 - Model Training",
        parameter_override={"dataset_task_id": "${stage_process.id}"},
    )

    pipe.start_locally()  # 或用 pipe.start(queue="default") 在线运行

    print("✅ Pipeline launched successfully.")

if __name__ == '__main__':
    run_pipeline()
