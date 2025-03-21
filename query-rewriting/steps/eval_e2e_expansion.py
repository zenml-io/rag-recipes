# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import warnings
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Annotated, Callable, Dict, List, Tuple

# Suppress the specific FutureWarning about clean_up_tokenization_spaces
warnings.filterwarnings(
    "ignore",
    message=".*clean_up_tokenization_spaces.*",
    category=FutureWarning,
    module="transformers.tokenization_utils_base",
)

from datasets import load_dataset
from litellm import completion
from pydantic import BaseModel, conint
from structures import TestResult
from utils.llm_utils import process_rewritten_queries
from utils.openai_utils import get_openai_api_key
from zenml import step
from zenml.logger import get_logger
from constants import OPENAI_MODEL

logger = get_logger(__name__)
logging.getLogger().setLevel(logging.WARNING)

# Test data sets from the original eval_e2e.py
bad_answers = [
    {
        "question": "What orchestrators does ZenML support?",
        "bad_words": ["AWS Step Functions", "Flyte", "Prefect", "Dagster"],
    },
    {
        "question": "What is the default orchestrator in ZenML?",
        "bad_words": ["Flyte", "AWS Step Functions"],
    },
]

bad_immediate_responses = [
    {
        "question": "Does ZenML support the Flyte orchestrator out of the box?",
        "bad_words": ["Yes"],
    },
]

good_responses = [
    {
        "question": "What are the supported orchestrators in ZenML? Please list as many of the supported ones as possible.",
        "good_words": ["Kubeflow", "Airflow"],
    },
    {
        "question": "What is the default orchestrator in ZenML?",
        "good_words": ["local"],
    },
]


def test_content_for_bad_words_with_expansion(
    item: dict, model: str = OPENAI_MODEL
) -> TestResult:
    """
    Test if responses contain bad words using query expansion.

    Args:
        item (dict): The item to test.
        model (str): The model to use for query expansion. Defaults to OPENAI_MODEL.

    Returns:
        TestResult: A TestResult object containing the test result information.
    """
    question = item["question"]
    bad_words = item["bad_words"]
    response = process_rewritten_queries(
        query=question,
        model=model,
    )
    for word in bad_words:
        if word in response:
            return TestResult(
                success=False,
                question=question,
                keyword=word,
                response=response,
            )
    return TestResult(success=True, question=question, response=response)


def test_response_starts_with_bad_words_with_expansion(
    item: dict, model: str = OPENAI_MODEL
) -> TestResult:
    """
    Test if responses improperly start with bad words using query expansion.

    Args:
        item (dict): The item to test.
        model (str): The model to use for query expansion. Defaults to OPENAI_MODEL.

    Returns:
        TestResult: A TestResult object containing the test result information.
    """
    question = item["question"]
    bad_words = item["bad_words"]
    response = process_rewritten_queries(
        query=question,
        model=model,
    )
    for word in bad_words:
        if response.lower().startswith(word.lower()):
            return TestResult(
                success=False,
                question=question,
                keyword=word,
                response=response,
            )
    return TestResult(success=True, question=question, response=response)


def test_content_contains_good_words_with_expansion(
    item: dict, model: str = OPENAI_MODEL
) -> TestResult:
    """
    Test if responses properly contain good words using query expansion.

    Args:
        item (dict): The item to test, containing the question and expected good words.
        model (str): The model to use for query expansion. Defaults to OPENAI_MODEL.

    Returns:
        TestResult: A TestResult object containing the test result information.
    """
    question = item["question"]
    good_words = item["good_words"]
    response = process_rewritten_queries(
        query=question,
        model=model,
    )
    for word in good_words:
        if word not in response:
            return TestResult(
                success=False,
                question=question,
                keyword=word,
                response=response,
            )
    return TestResult(success=True, question=question, response=response)


class LLMJudgedTestResult(BaseModel):
    """A class to represent the results of the LLM judged test.

    All scores are on a scale of 1 to 5.

    Attributes:
        toxicity: Score for the toxicity of the response.
        faithfulness: Score for the factual consistency taking the given context
            into account.
        helpfulness: How helpful the response is.
        relevance: How relevant the response is to the question.
    """

    toxicity: conint(ge=1, le=5)
    faithfulness: conint(ge=1, le=5)
    helpfulness: conint(ge=1, le=5)
    relevance: conint(ge=1, le=5)


def llm_judged_test_e2e_with_expansion(
    question: str,
    context: str,
    model: str = OPENAI_MODEL,
) -> LLMJudgedTestResult:
    """E2E tests judged by an LLM using query expansion.

    Args:
        question (str): The question to test.
        context (str): The context to use for evaluation.
        model (str): The model to use for query expansion. Defaults to OPENAI_MODEL.

    Returns:
        LLMJudgedTestResult: The result of the test.
    """
    logging.debug("Starting LLM judged test with expansion...")
    response = process_rewritten_queries(
        query=question,
        model=model,
    )
    logging.debug("Input processed with retrieval and query expansion.")
    prompt = f"""
    Please analyze the following text and provided context to provide scores
    for toxicity, faithfulness, helpfulness, and relevance. The toxicity score 
    should be an integer between 1 and 5, where 1 is not toxic and 5 is very
    toxic. The faithfulness score should be an integer between 1 and 5, where 1
    is not faithful to the provided context and 5 is very faithful. 
    The helpfulness score should be an integer between 1 and 5, where 1 is not
    helpful and 5 is very helpful. The relevance score should be an integer
    between 1 and 5, where 1 is not relevant given the context and 5 is very relevant.

    **Text:** {response}

    **Context:** {context}

    **Output format:**
    {{
        "toxicity": int,
        "faithfulness": int,
        "helpfulness": int,
        "relevance": int
    }}
    """
    logging.debug("Prompt created.")
    response = completion(
        model="gpt-4-turbo",
        messages=[{"content": prompt, "role": "user"}],
        api_key=get_openai_api_key(),
    )

    json_output = response["choices"][0]["message"]["content"].strip()
    logging.info("Received response from model.")
    logging.debug(json_output)
    try:
        return LLMJudgedTestResult(**json.loads(json_output))
    except json.JSONDecodeError as e:
        logging.error(f"JSON bad output: {json_output}")
        raise e


def process_single_llm_test_with_expansion(
    item: Dict, test_function: Callable, model: str = OPENAI_MODEL
) -> Tuple[float, float, float, float]:
    """Process a single LLM test with query expansion.

    Args:
        item (Dict): The dataset item to test.
        test_function (Callable): The test function to use.
        model (str): The model to use for query expansion. Defaults to OPENAI_MODEL.

    Returns:
        Tuple[float, float, float, float]: Tuple containing toxicity, faithfulness, helpfulness, and relevance scores.
    """
    try:
        question = item["generated_questions"][0]
        context = item["content"]
        result = test_function(question, context, model)
        return (
            result.toxicity,
            result.faithfulness,
            result.helpfulness,
            result.relevance,
        )
    except Exception as e:
        logger.error(f"Error processing item: {str(e)}")
        return None


def run_llm_judged_tests_with_expansion(
    test_function: Callable,
    sample_size: int = 10,
    model: str = OPENAI_MODEL,
) -> Tuple[
    Annotated[float, "average_toxicity_score"],
    Annotated[float, "average_faithfulness_score"],
    Annotated[float, "average_helpfulness_score"],
    Annotated[float, "average_relevance_score"],
]:
    """E2E tests judged by an LLM using query expansion.

    Args:
        test_function (Callable): The test function to run.
        sample_size (int): The sample size to run the tests on. Defaults to 10.
        model (str): The model to use for query expansion. Defaults to OPENAI_MODEL.

    Returns:
        Tuple: The average toxicity, faithfulness, helpfulness, and relevance scores.
    """
    # Load the dataset from the Hugging Face Hub
    dataset = load_dataset("zenml/rag_qa_embedding_questions", split="train")

    # Shuffle the dataset and select a random sample
    sampled_dataset = dataset.shuffle(seed=42).select(range(sample_size))

    n_processes = max(1, cpu_count() // 2)
    worker = partial(
        process_single_llm_test_with_expansion, 
        test_function=test_function,
        model=model,
    )

    try:
        logger.info(f"Running LLM judged tests with expansion on {sample_size} samples...")
        results = []
        with Pool(processes=n_processes) as pool:
            for i, result in enumerate(pool.imap(worker, sampled_dataset), 1):
                results.append(result)
                logger.info(f"Completed {i}/{len(sampled_dataset)} tests")

        # Filter out None results (failed tests)
        valid_results = [r for r in results if r is not None]
        total_tests = len(valid_results)

        if total_tests == 0:
            logger.error("All tests failed!")
            return 0.0, 0.0, 0.0, 0.0

        # Calculate totals
        total_toxicity = sum(r[0] for r in valid_results)
        total_faithfulness = sum(r[1] for r in valid_results)
        total_helpfulness = sum(r[2] for r in valid_results)
        total_relevance = sum(r[3] for r in valid_results)

        # Calculate averages
        average_toxicity_score = total_toxicity / total_tests
        average_faithfulness_score = total_faithfulness / total_tests
        average_helpfulness_score = total_helpfulness / total_tests
        average_relevance_score = total_relevance / total_tests

        logger.info("\nTest Results Summary:")
        logger.info(f"Total valid tests: {total_tests}")
        logger.info(f"Average toxicity: {average_toxicity_score:.3f}")
        logger.info(f"Average faithfulness: {average_faithfulness_score:.3f}")
        logger.info(f"Average helpfulness: {average_helpfulness_score:.3f}")
        logger.info(f"Average relevance: {average_relevance_score:.3f}")

        return (
            round(average_toxicity_score, 3),
            round(average_faithfulness_score, 3),
            round(average_helpfulness_score, 3),
            round(average_relevance_score, 3),
        )

    except Exception as e:
        logger.error(f"Error during parallel processing: {str(e)}")
        raise


def run_simple_tests_with_expansion(
    test_data: list, test_function: Callable, model: str = OPENAI_MODEL
) -> float:
    """Run simple tests with query expansion.

    Args:
        test_data (list): The test data.
        test_function (Callable): The test function to run.
        model (str): The model to use for query expansion. Defaults to OPENAI_MODEL.

    Returns:
        float: The failure rate.
    """
    failures = 0
    for item in test_data:
        result = test_function(item, model=model)
        if not result.success:
            failures += 1
            logger.error(
                f"Failed test for question: '{result.question}'\n"
                f"Expected keyword: {result.keyword}\n"
                f"Got response: {result.response}"
            )
        else:
            logger.info(f"Passed test for question: '{result.question}'")

    failure_rate = (failures / len(test_data)) * 100
    logger.info(
        f"\nTest Summary:\n"
        f"Total tests: {len(test_data)}\n"
        f"Failures: {failures}\n"
        f"Failure rate: {failure_rate}%"
    )
    return failure_rate


@step
def e2e_evaluation_with_expansion(
    model: str = OPENAI_MODEL,
) -> Tuple[
    Annotated[float, "failure_rate_bad_answers_expansion"],
    Annotated[float, "failure_rate_bad_immediate_responses_expansion"],
    Annotated[float, "failure_rate_good_responses_expansion"],
]:
    """Executes the E2E evaluation step with query expansion.

    Args:
        model (str): The model to use for query expansion. Defaults to OPENAI_MODEL.

    Returns:
        Tuple: The failure rates for bad answers, bad immediate responses, and good responses.
    """
    logger.info("Starting E2E evaluation with query expansion...")
    
    failure_rate_bad_answers = run_simple_tests_with_expansion(
        bad_answers, test_content_for_bad_words_with_expansion, model=model
    )
    
    failure_rate_bad_immediate_responses = run_simple_tests_with_expansion(
        bad_immediate_responses, 
        test_response_starts_with_bad_words_with_expansion,
        model=model
    )
    
    failure_rate_good_responses = run_simple_tests_with_expansion(
        good_responses, 
        test_content_contains_good_words_with_expansion,
        model=model
    )
    
    logger.info(
        f"E2E Evaluation Results with Query Expansion:\n"
        f"Bad Answers Failure Rate: {failure_rate_bad_answers}%\n"
        f"Bad Immediate Responses Failure Rate: {failure_rate_bad_immediate_responses}%\n"
        f"Good Responses Failure Rate: {failure_rate_good_responses}%"
    )
    
    return (
        round(failure_rate_bad_answers, 2),
        round(failure_rate_bad_immediate_responses, 2),
        round(failure_rate_good_responses, 2),
    )


@step
def e2e_evaluation_llm_judged_with_expansion(
    sample_size: int = 10,
    model: str = OPENAI_MODEL,
) -> Tuple[
    Annotated[float, "average_toxicity_score_expansion"],
    Annotated[float, "average_faithfulness_score_expansion"],
    Annotated[float, "average_helpfulness_score_expansion"],
    Annotated[float, "average_relevance_score_expansion"],
]:
    """Executes the E2E evaluation step with LLM judging and query expansion.

    Args:
        sample_size (int): The sample size to run the tests on. Defaults to 10.
        model (str): The model to use for query expansion. Defaults to OPENAI_MODEL.

    Returns:
        Tuple: The average toxicity, faithfulness, helpfulness, and relevance scores.
    """
    logger.info("Starting E2E evaluation with LLM judging and query expansion...")
    
    return run_llm_judged_tests_with_expansion(
        llm_judged_test_e2e_with_expansion,
        sample_size=sample_size,
        model=model,
    ) 