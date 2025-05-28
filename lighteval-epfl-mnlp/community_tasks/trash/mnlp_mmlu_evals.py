from lighteval.tasks.requests import Doc
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig


def mmlu_harness(line, task_name: str = None):
    """
    Prompt function for the MMLU dataset ("all" config, "test" split).
    Each example has:
      - question: the question string
      - choices: a list of 4 answer strings
      - answer: the correct answer letter ("A"|"B"|"C"|"D")
    """
    topic = "knowledge and skills across diverse academic subjects"
    prompt = (
        f"The following are multiple choice questions (with answers) about {topic}.\n\n"
        f"{line['question']}\n"
    )
    for key, choice in zip(LETTER_INDICES, line['choices']):
        prompt += f"{key}. {choice}\n"
    prompt += "Answer:"

    # 'answer' is a class-label integer 0–3
    gold_ix = int(line['answer'])

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[f" {l}" for l in LETTER_INDICES[: len(line['choices'])]],
        gold_index=gold_ix,
        instruction=(
            f"The following are multiple choice questions (with answers) about {topic}.\n\n"
        ),
    )


task = LightevalTaskConfig(
    name="mnlp_mmlu_evals",
    prompt_function=mmlu_harness,
    suite=["community"],
    hf_subset="all",           # use the 'all' subset
    hf_repo="cais/mmlu",
    hf_avail_splits=["test"],  # the test split of the 'all' subset
    evaluation_splits=["test"],
    metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
    generation_size=-1,
    stop_sequence=None,
    trust_dataset=True,
    limited_num_samples=0,      # use the entire test split
)

# STORE YOUR EVALS
TASKS_TABLE = [task]
