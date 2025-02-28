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

import logging
from typing import Dict, List, Optional, Tuple
from typing_extensions import Annotated

import numpy as np
from constants import OPENAI_MODEL
from utils.llm_utils import generate_rewritten_queries, get_completion_from_messages
from zenml import step

logger = logging.getLogger(__name__)

TEST_QUERIES = [
    "How do I set up authentication in ZenML?",
    "What are the different types of stack components?",
    "How can I deploy my model using ZenML?",
    "What is the difference between a pipeline and a step?",
    "How do I configure secrets in ZenML?",
]

@step
def evaluate_query_information_retention(
    model: str = OPENAI_MODEL,
) -> Tuple[Annotated[float, "query_retention_score"], Annotated[List[Dict], "query_retention_results"]]:
    """Evaluates if rewritten queries retain key information from original queries.

    Args:
        after (Optional[str], optional): Filter for model versions after this date. Defaults to None.
        model (str, optional): Model to use for evaluation. Defaults to OPENAI_MODEL.

    Returns:
        Tuple[float, List[dict]]: Average retention score and detailed results.
    """
    results = []
    retention_scores = []

    for original_query in TEST_QUERIES:
        # Generate rewritten queries
        rewritten_queries = generate_rewritten_queries(
            original_query,
            model=model,
        )

        # For each rewritten query, check if it maintains key information
        system_message = """
        You are an expert at analyzing search queries.
        Your task is to evaluate if a rewritten query maintains the key information and intent of the original query.
        Score from 0 to 1, where:
        0 = Completely lost the original meaning/key concepts
        0.5 = Maintained some key concepts but lost important context
        1 = Perfectly maintained all key information and intent

        Respond with just the score as a float between 0 and 1.
        """

        query_scores = []
        for rewritten_query in rewritten_queries:
            if not rewritten_query:  # Skip empty queries
                continue

            messages = [
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": f"Original query: {original_query}\nRewritten query: {rewritten_query}",
                },
            ]

            score_str = get_completion_from_messages(
                messages,
                model=model,
                temperature=0,
            )

            try:
                score = float(score_str)
                query_scores.append(score)
            except ValueError:
                logger.error(f"Failed to parse score: {score_str}")
                query_scores.append(0.0)

        avg_score = np.mean(query_scores) if query_scores else 0.0
        retention_scores.append(avg_score)

        results.append({
            "original_query": original_query,
            "rewritten_queries": rewritten_queries,
            "retention_scores": query_scores,
            "average_retention_score": avg_score,
        })

    overall_retention_score = np.mean(retention_scores)
    return overall_retention_score, results

@step
def evaluate_query_length_quality(
    model: str = OPENAI_MODEL,
    max_length: int = 150,  # Maximum acceptable length in characters
) -> Tuple[Annotated[float, "query_length_score"], Annotated[List[Dict], "query_length_results"]]:
    """Evaluates if rewritten queries maintain a reasonable length.

    Args:
        after (Optional[str], optional): Filter for model versions after this date. Defaults to None.
        model (str, optional): Model to use for evaluation. Defaults to OPENAI_MODEL.
        max_length (int, optional): Maximum acceptable length in characters. Defaults to 150.

    Returns:
        Tuple[float, List[dict]]: Success rate and detailed results.
    """
    results = []
    success_rates = []

    for original_query in TEST_QUERIES:
        # Generate rewritten queries
        rewritten_queries = generate_rewritten_queries(
            original_query,
            model=model,
        )

        # Check length of each rewritten query
        query_results = []
        success_count = 0

        for rewritten_query in rewritten_queries:
            if not rewritten_query:  # Skip empty queries
                continue

            is_good_length = len(rewritten_query) <= max_length
            if is_good_length:
                success_count += 1

            query_results.append({
                "query": rewritten_query,
                "length": len(rewritten_query),
                "is_good_length": is_good_length,
            })

        success_rate = success_count / len(rewritten_queries) if rewritten_queries else 0
        success_rates.append(success_rate)

        results.append({
            "original_query": original_query,
            "query_results": query_results,
            "success_rate": success_rate,
        })

    overall_success_rate = np.mean(success_rates)
    return overall_success_rate, results 