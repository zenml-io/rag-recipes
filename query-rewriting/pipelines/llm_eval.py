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
from pathlib import Path
from typing import Optional

import click
from steps.eval_e2e import e2e_evaluation, e2e_evaluation_llm_judged
from steps.eval_e2e_expansion import (
    e2e_evaluation_with_expansion,
    e2e_evaluation_llm_judged_with_expansion,
)
from steps.eval_retrieval import (
    retrieval_evaluation_full,
    retrieval_evaluation_full_with_reranking,
    retrieval_evaluation_small,
    retrieval_evaluation_small_with_reranking,
    retrieval_evaluation_small_with_expansion,
    retrieval_evaluation_full_with_expansion,
)
from steps.eval_visualisation import visualize_evaluation_results
from steps.eval_query_rewriting import (
    evaluate_query_information_retention,
    evaluate_query_length_quality,
)
from zenml import pipeline


@pipeline(enable_cache=False)
def llm_eval(after: Optional[str] = None) -> None:
    """Executes the pipeline to evaluate a RAG pipeline."""
    # Query rewriting evals
    query_retention_score, query_retention_results = evaluate_query_information_retention(after=after)
    query_length_score, query_length_results = evaluate_query_length_quality(after=after)

    # Retrieval evals
    failure_rate_retrieval = retrieval_evaluation_small(after=after)
    full_retrieval_answers = retrieval_evaluation_full(after=after)
    failure_rate_retrieval_reranking = (
        retrieval_evaluation_small_with_reranking(after=after)
    )
    full_retrieval_answers_reranking = (
        retrieval_evaluation_full_with_reranking(after=after)
    )
    
    # Query expansion retrieval evals
    failure_rate_retrieval_expansion = (
        retrieval_evaluation_small_with_expansion(after=after)
    )
    full_retrieval_answers_expansion = (
        retrieval_evaluation_full_with_expansion(after=after)
    )

    # # E2E evals
    (
        failure_rate_bad_answers,
        failure_rate_bad_immediate_responses,
        failure_rate_good_responses,
    ) = e2e_evaluation(after=after)

    (
        average_toxicity_score,
        average_faithfulness_score,
        average_helpfulness_score,
        average_relevance_score,
    ) = e2e_evaluation_llm_judged(after=after)
    
    # E2E evals with query expansion
    (
        failure_rate_bad_answers_expansion,
        failure_rate_bad_immediate_responses_expansion,
        failure_rate_good_responses_expansion,
    ) = e2e_evaluation_with_expansion()
    
    (
        average_toxicity_score_expansion,
        average_faithfulness_score_expansion,
        average_helpfulness_score_expansion,
        average_relevance_score_expansion,
    ) = e2e_evaluation_llm_judged_with_expansion()

    visualize_evaluation_results(
        failure_rate_retrieval,
        failure_rate_retrieval_reranking,
        full_retrieval_answers,
        full_retrieval_answers_reranking,
        failure_rate_bad_answers,
        failure_rate_bad_immediate_responses,
        failure_rate_good_responses,
        average_toxicity_score,
        average_faithfulness_score,
        average_helpfulness_score,
        average_relevance_score,
        query_retention_score,
        query_length_score,
        failure_rate_retrieval_expansion,
        full_retrieval_answers_expansion,
        failure_rate_bad_answers_expansion,
        failure_rate_bad_immediate_responses_expansion,
        failure_rate_good_responses_expansion,
        average_toxicity_score_expansion,
        average_faithfulness_score_expansion,
        average_helpfulness_score_expansion,
        average_relevance_score_expansion
    )


@click.option(
    "--no-cache",
    "no_cache",
    is_flag=True,
    default=False,
    help="Disable cache.",
)
@click.option(
    "--config",
    "config",
    default="rag_local_dev.yaml",
    help="Specify a configuration file",
)
def main(no_cache: bool = False, config: Optional[str] = "rag_eval.yaml"):
    """
    Executes the pipeline to train a basic RAG model.

    Args:
        no_cache (bool): If `True`, cache will be disabled.
        config (str): The path to the configuration file.
    """
    config_path = Path(__file__).parent.parent / "configs" / config

    llm_eval.with_options(
        config_path=str(config_path), enable_cache=not no_cache
    )()


if __name__ == "__main__":
    main()
