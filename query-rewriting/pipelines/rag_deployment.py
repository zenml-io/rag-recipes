from steps.rag_deployment import gradio_rag_deployment
from zenml import pipeline, Model


@pipeline(enable_cache=False, model=Model())
def rag_deployment():
    gradio_rag_deployment()
