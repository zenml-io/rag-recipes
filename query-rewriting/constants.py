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
#
import os

# Vector Store constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 50
EMBEDDING_DIMENSIONALITY = (
    384  # Update this to match the dimensionality of the new model
)

# ZenML constants
ZENML_CHATBOT_MODEL = "zenml-docs-qa-chatbot"
ZENML_CHATBOT_MODEL_NAME = "zenml-docs-qa-chatbot"
ZENML_CHATBOT_MODEL_VERSION = "0.75.0-dev"

# Scraping constants
RATE_LIMIT = 5  # Maximum number of requests per second

# LLM Utils constants
OPENAI_MODEL = "gpt-4o"
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
MODEL_NAME_MAP = {
    "gpt4o": "gpt-4o",
    "claude3": "claude-3-7-sonnet-latest",
    "claudehaiku": "claude-3-5-haiku-latest",
}

# Markdown Loader constants
FILES_TO_IGNORE = [
    "toc.md",
]

SECRET_NAME = os.getenv("ZENML_PROJECT_SECRET_NAME", "llm-complete")
SECRET_NAME_ELASTICSEARCH = "elasticsearch-zenml"
