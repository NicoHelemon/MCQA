from lighteval.tasks.requests import Doc
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig

def mmlu_harness_legacy(line, task_name: str = None):
    topic = "knowledge and skills in advanced master-level STEM courses"
    prompt = f"The following are multiple choice questions (with answers) about {topic.replace('_', ' ')}.\n\n"
    prompt += line["question"] + "\n"
    prompt += "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, line["choices"])])
    prompt += "Answer:"
    gold_ix = LETTER_INDICES.index(line["answer"]) if isinstance(line["answer"], str) else line["answer"]

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[" A", " B", " C", " D"],
        gold_index=gold_ix,
        instruction=f"The following are multiple choice questions (with answers) about {topic.replace('_', ' ')}.\n\n",
    )

def mmlu_harness(line, task_name: str = None):
    topic = "knowledge and skills in advanced master-level STEM courses"
    prompt = f"The following are multiple choice questions (with answers) about {topic.replace('_', ' ')}.\n\n"
    prompt += line["question"] + "\n"
    prompt += "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, line["choices"])])
    prompt += "Answer:"
    gold_idx = LETTER_INDICES.index(line["answer"]) if isinstance(line["answer"], str) else line["answer"]

    return Doc(
        task_name=task_name,
        query=prompt,
        # For single token continuation
        # choices = [" A", " B", " C", " D"]
        # For multi-token continuation
        choices = [f" {key}. {choice}" for key, choice in zip(LETTER_INDICES, line["choices"])],
        gold_index=gold_idx,
        instruction=f"The following are multiple choice questions (with answers) about {topic.replace('_', ' ')}.\n\n",
    )

def MNLP_M3_mcqa_dataset_harness(line, task_name: str = None):
    topic = "knowledge and skills in advanced master-level STEM courses"
    prompt = f"The following are multiple choice questions (with answers) about {topic.replace('_', ' ')}.\n\n"
    prompt += line["question"] + "\n"
    prompt += "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, line["options"])])
    prompt += "Answer:"
    gold_ix = line["label_idx"]

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[" A", " B", " C", " D"],
        gold_index=gold_ix,
        instruction=f"The following are multiple choice questions (with answers) about {topic.replace('_', ' ')}.\n\n",
    )

def med_qa(line, task_name: str = None):
    query = f"Give a letter answer among A, B, C or D.\nQuestion: {line['question']}\n"
    query += "".join([f"{option['key']}. {option['value']}\n" for option in line["options"]])
    query += "Answer:"
    return Doc(
        task_name=task_name,
        query=query,
        choices=[opt["key"] for opt in line["options"]],
        gold_index=LETTER_INDICES.index(line["answer_idx"]),
        instruction="Give a letter answer among A, B, C or D.\n",
    )

def commonsense_qa(line, task_name: str = None):
    query = f"The following are multiple choice questions (with answers) about common sense.\nQuestion: {line['question']}\n"
    query += "".join(
        [f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, [f" {c}" for c in line["choices"]["text"]])]
    )
    query += "Answer:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=LETTER_INDICES[: len(line["choices"]["text"])],
        gold_index=LETTER_INDICES.index(line["answerKey"].strip()),
        instruction="The following are multiple choice questions (with answers) about common sense.\n",
    )

def arc_with_options_letters_predict(line, task_name: str = None):
    query = f"Question: {line['question']}\n"
    query += "".join([f"\n{key}. {choice}" for key, choice in zip(LETTER_INDICES, line["choices"]["text"])])
    query += "\nAnswer:"
    return Doc(
        task_name=task_name,
        query=query,
        choices=LETTER_INDICES[: len(line["choices"]["text"])],
        gold_index=line["choices"]["label"].index(line["answerKey"]),
    )

####################################

task_MNLP_STEM_mcqa_demo_legacy = LightevalTaskConfig(
    name="mnlp_mcqa_evals_legacy",
    prompt_function=mmlu_harness_legacy,
    suite=["community"],
    hf_subset="",
    hf_repo="zechen-nlp/MNLP_STEM_mcqa_demo",  # Change the repo name to the evaluation dataset that you compiled
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
    generation_size=-1,
    stop_sequence=None,
    trust_dataset=True,
    limited_num_samples=0,  # Set to 0 to use all samples, specify a number to limit the number of samples for debugging purposes
)

task_MNLP_STEM_mcqa_demo = LightevalTaskConfig(
    name="mnlp_mcqa_evals",
    prompt_function=mmlu_harness,
    suite=["community"],
    hf_subset="",
    hf_repo="zechen-nlp/MNLP_STEM_mcqa_demo",  # Change the repo name to the evaluation dataset that you compiled
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
    generation_size=-1,
    stop_sequence=None,
    trust_dataset=True,
    limited_num_samples=0,  # Set to 0 to use all samples, specify a number to limit the number of samples for debugging purposes
)

####################################

task_MNLP_M3_mcqa_dataset = LightevalTaskConfig(
    name="MNLP_M3_mcqa_dataset",
    prompt_function=MNLP_M3_mcqa_dataset_harness,
    suite=["community"],
    hf_subset="",
    hf_repo="NicoHelemon/MNLP_M3_mcqa_dataset",  # Change the repo name to the evaluation dataset that you compiled
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test", "validation"],
    metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
    generation_size=-1,
    stop_sequence=None,
    trust_dataset=True,
    limited_num_samples=0,  # Set to 0 to use all samples, specify a number to limit the number of samples for debugging purposes
)

task_mmlu_STEM_mmlu_harness = LightevalTaskConfig(
    name="mmlu:stem",
    prompt_function=mmlu_harness_legacy,
    suite=["community"],
    hf_repo="NicoHelemon/mmlu_STEM",
    hf_subset="",  # using the full dataset
    hf_avail_splits=["test", "validation", "dev"],
    evaluation_splits=["test"],
    metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
    generation_size=-1,
    stop_sequence=None,
    trust_dataset=True,
    limited_num_samples=0,  # Set to 0 to use all samples, specify a number to limit the number of samples for debugging purposes
)

####################################

med_qa_helm = LightevalTaskConfig(
    name="med_qa",
    suite=["helm"],
    prompt_function=med_qa,
    hf_repo="bigbio/med_qa",
    hf_subset="med_qa_en_source",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
    stop_sequence=["\n"],
    trust_dataset=True,
    version=0,
)

commonsenseqa_helm = LightevalTaskConfig(
    name="commonsenseqa",
    suite=["helm", "commonsense_scenario"],
    prompt_function=commonsense_qa,
    hf_repo="commonsense_qa",
    hf_subset="default",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
    stop_sequence=["\n"],
    trust_dataset=True,
    version=0,
)

arc_c_letters_original = LightevalTaskConfig(
    name="arc:c:letters",
    suite=["original", "arc"],
    prompt_function=arc_with_options_letters_predict,
    hf_repo="ai2_arc",
    hf_subset="ARC-Challenge",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
    stop_sequence=["\n"],
    trust_dataset=True,
    version=0,
)

# STORE YOUR EVALS
TASKS_TABLE = [task_MNLP_STEM_mcqa_demo_legacy,
               task_MNLP_STEM_mcqa_demo,
               task_MNLP_M3_mcqa_dataset,
               task_mmlu_STEM_mmlu_harness,
               med_qa_helm,
               commonsenseqa_helm,
               arc_c_letters_original]
