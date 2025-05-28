from lighteval.tasks.requests import Doc
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig


def openbookqa_harness(line, task_name: str = None):
    """
    Prompt function for the OpenBookQA dataset (main/test split).
    Each example has a question_stem, a list of four answer choices, and the correct answer key.
    """
    topic = "science exam questions"
    # question stem
    question = line['question_stem']
    prompt = (
        f"The following are multiple choice questions (with answers) about {topic}.\n\n"
        f"{question}\n"
    )

    # extract choice texts (list of strings)
    choices = line['choices']

    # append labeled options A. ... D.
    for key, choice in zip(LETTER_INDICES, choices):
        prompt += f"{key}. {choice}\n"
    prompt += "Answer:"

    # answerKey is something like 'A', 'B', 'C', or 'D'
    gold_ix = LETTER_INDICES.index(line['answerKey'])

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[f" {l}" for l in LETTER_INDICES[: len(choices)]],
        gold_index=gold_ix,
        instruction=(
            f"The following are multiple choice questions (with answers) about {topic}.\n\n"
        ),
    )


task = LightevalTaskConfig(
    name="mnlp_openbookqa_evals",
    prompt_function=openbookqa_harness,
    suite=["community"],
    hf_subset="main",
    hf_repo="allenai/openbookqa",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
    generation_size=-1,
    stop_sequence=None,
    trust_dataset=True,
    limited_num_samples=0,  # 0 to use all examples in main/test
)

# STORE YOUR EVALS
TASKS_TABLE = [task]
