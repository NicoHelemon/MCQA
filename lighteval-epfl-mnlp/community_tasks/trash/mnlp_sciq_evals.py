from lighteval.tasks.requests import Doc
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig


def sciq_harness(line, task_name: str = None):
    """
    Prompt function for the SciQ dataset.
    Each example contains a question with one correct answer and three distractors.
    """
    topic = "science exam questions"
    # build prompt header
    prompt = (
        f"The following are multiple choice questions (with answers) about {topic}.\n\n"
        f"{line['question']}\n"
    )
    # collect all answer options: correct first, then distractors
    options = [
        line["correct_answer"],
        line.get("distractor1"),
        line.get("distractor2"),
        line.get("distractor3"),
    ]
    # append choices A. ... D.
    for key, choice in zip(LETTER_INDICES, options):
        prompt += f"{key}. {choice}\n"
    prompt += "Answer:"

    # correct answer is always the first in our list
    gold_ix = 0

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[f" {l}" for l in LETTER_INDICES[: len(options)]],
        gold_index=gold_ix,
        instruction=(
            f"The following are multiple choice questions (with answers) about {topic}.\n\n"
        ),
    )


task = LightevalTaskConfig(
    name="mnlp_sciq_evals",
    prompt_function=sciq_harness,
    suite=["community"],
    hf_subset="",  # not used
    hf_repo="allenai/sciq",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
    generation_size=-1,
    stop_sequence=None,
    trust_dataset=True,
    limited_num_samples=0,  # Set to 0 to use all test examples
)

# STORE YOUR EVALS
TASKS_TABLE = [task]
