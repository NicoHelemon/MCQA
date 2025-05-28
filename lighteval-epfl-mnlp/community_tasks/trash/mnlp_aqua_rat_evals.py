from lighteval.tasks.requests import Doc
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig

def aqua_rat_harness(line, task_name=None):
    prompt = (
        "The following are multiple choice questions (with answers) "
        "about mathematics problem-solving.\n\n"
        f"{line['question']}\n"
    )
    # A–E
    for key, opt in zip(LETTER_INDICES, line["options"]):
        prompt += f"{key}. {opt}\n"
    prompt += "Answer:"
    gold_ix = LETTER_INDICES.index(line["correct"])
    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[f" {l}" for l in LETTER_INDICES[: len(line["options"])]],
        gold_index=gold_ix,
        instruction="The following are multiple choice questions (with answers) about mathematics problem-solving.\n\n"
    )

task = LightevalTaskConfig(
    name="mnlp_aqua_rat_evals",
    prompt_function=aqua_rat_harness,
    suite=["community"],
    hf_subset="",
    hf_repo="deepmind/aqua_rat",  # Change the repo name to the evaluation dataset that you compiled
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
    generation_size=-1,
    stop_sequence=None,
    trust_dataset=True,
    limited_num_samples=0,  # Set to 0 to use all samples, specify a number to limit the number of samples for debugging purposes
)

# STORE YOUR EVALS
TASKS_TABLE = [task]
