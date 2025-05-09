from clearml import Task
from clearml.automation import PipelineController


def pre_execute_callback_example(a_pipeline, a_node, current_param_override):
    # type (PipelineController, PipelineController.Node, dict) -> bool
    print(
        "Cloning Task id={} with parameters: {}".format(
            a_node.base_task_id, current_param_override
        )
    )
    # if we want to skip this node (and subtree of this node) we return False
    # return True to continue DAG execution
    return True


def post_execute_callback_example(a_pipeline, a_node):
    # type (PipelineController, PipelineController.Node) -> None
    print("Completed Task id={}".format(a_node.executed))
    # if we need the actual executed Task: Task.get_task(task_id=a_node.executed)
    return


def run_pipeline():
    # Connecting ClearML with the current pipeline,
    # from here on everything is logged automatically
    pipe = PipelineController(
        name="Pipeline demo", project="examples", version="0.0.1", add_pipeline_tags=False
    )

    pipe.set_default_execution_queue("pipeline")

    pipe.add_step(
        name="stage_data",
        base_task_project="examples",
        base_task_name="Step 1 - Load Uncompressed Pest Image Dataset",
    )

    pipe.add_step(
        name="stage_process",
        parents=["stage_data"],
        base_task_name="Step 2 - Preprocessing (artifact version)",
        base_task_project="Agri-Pest-Detection",
        parameter_override={
            "General/dataset_task_id": "${stage_data.id}",
            "General/test_size": 0.25,
            "General/random_state": 42
        },
    )

    pipe.add_step(
        name="stage_train",
        parents=["stage_process"],
        base_task_project="Agri-Pest-Detection",
        base_task_name="Step 3 - Model Training",
        parameter_override={"General/dataset_task_id": "${stage_process.id}"},
    )

    # for debugging purposes use local jobs
    pipe.start_locally()

    # Starting the pipeline (in the background)
    # pipe.start(queue="pipeline")  # already set pipeline queue

    print("done")
