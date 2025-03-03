# RAG Recipes: Advanced Patterns with Robust Evaluation Pipelines
Welcome to RAG Recipes, a repository dedicated to showcasing advanced Retrieval-Augmented Generation (RAG) patterns, each accompanied by comprehensive evaluation pipelines. Built with ZenML, this repository emphasizes not just implementation, but rigorous evaluation—ensuring your RAG systems are reliable, accurate, and production-ready.

## 🚀 Why RAG Recipes?
Retrieval-Augmented Generation (RAG) has revolutionized how we build intelligent, context-aware AI applications. However, while many tutorials and repositories cover the basics of RAG, few address the critical aspect of evaluation. Without proper evaluation, even the most advanced RAG patterns can silently degrade user experience and trust.
RAG Recipes bridges this gap by providing:
- Advanced RAG Patterns: Implementations of cutting-edge techniques like query rewriting, hybrid retrieval, embeddings optimization, and more.
- Robust Evaluation Pipelines: Automated, reproducible evaluation pipelines built with ZenML, ensuring each pattern is rigorously tested and validated.
- ZenML Integration: Leveraging ZenML's powerful MLOps capabilities—automation, tracking, caching, visualization, and artifact management—to streamline your workflow.

## 🎯 What You'll Find Here
Each sub-project in this repository focuses on a specific advanced RAG pattern, complete with:
- Implementation Code: Ready-to-use code examples demonstrating the RAG pattern.
- Evaluation Pipelines: ZenML pipelines that systematically evaluate the effectiveness of each pattern.
- Detailed Documentation: Clear explanations, setup instructions, and insights into the evaluation results.

## 📦 Current Recipes
**Query Rewriting**: Improve retrieval accuracy by reformulating ambiguous user queries, with comprehensive evaluation to ensure intent preservation and relevance.
(More recipes coming soon!)

## 🛠️ Getting Started
Each recipe is self-contained with its own README, providing detailed instructions on setup, running pipelines, and interpreting evaluation results. Here's how you can quickly get started:
1. Clone the repository:
```
git clone https://github.com/<your-username>/rag-recipes.git
cd rag-recipes
```
2. Navigate to a recipe:
```
cd query-rewriting
```
3. Follow the recipe-specific README to set up your environment, run pipelines, and explore evaluation results.

## 📊 Why ZenML?
ZenML simplifies the complexity of building, deploying, and evaluating machine learning pipelines. With ZenML, each RAG recipe benefits from:
Automation: Easily automate ingestion, deployment, and evaluation workflows.
- Tracking & Versioning: Keep track of pipeline runs, artifacts, and model versions effortlessly.
- Caching: Speed up experimentation by caching pipeline steps.
- Visualization: Interactive dashboards and visualizations to quickly interpret evaluation metrics.

Learn more about ZenML at zenml.io.

## 🤝 Contributing
We welcome contributions! Whether it's adding new RAG patterns, improving existing evaluation pipelines, or enhancing documentation—your input is valuable.
Please open an issue or submit a pull request to get involved.

## 📖 License
This project is licensed under the MIT License. See the LICENSE file for details.