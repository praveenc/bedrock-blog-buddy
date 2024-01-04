import boto3
from langchain.embeddings.bedrock import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock


def get_langchain_bedrock_llm(
    model_id: str = "anthropic.claude-v2:1", region: str = "us-west-2"
):
    bedrock_client = boto3.client("bedrock-runtime", region_name=region)
    model_kwargs = get_inference_parameters("anthropic")

    llm = Bedrock(
        client=bedrock_client,
        model_kwargs=model_kwargs,
        model_id=model_id,
        region_name=region,
    )
    return llm


def get_langchain_bedrock_embeddings(
    model_id: str = "cohere.embed-english-v3", region: str = "us-west-2"
):
    bedrock_client = boto3.client("bedrock-runtime", region_name=region)
    embeddings = BedrockEmbeddings(
        client=bedrock_client, model_id=model_id, region_name=region
    )
    return embeddings


def get_model_ids(
    provider: str = "Anthropic",
    output_modality: str = "TEXT",
    region: str = "us-west-2",
):
    """
    Fetch model IDs from AWS Bedrock for specified provider and output modality.

    Args:
    provider (str): The provider of the model.
    output_modality (str): The output modality of the model.

    Returns:
    list: A list of model IDs that match the criteria.
    """
    bedrock_client = boto3.client("bedrock", region_name=region)
    models = bedrock_client.list_foundation_models()["modelSummaries"]
    model_ids = [
        model["modelId"]
        for model in models
        if model["providerName"] == provider
        and model["outputModalities"] == [output_modality]
    ]

    return model_ids


def get_inference_parameters(
    model,
):  # return a default set of parameters based on the model's provider
    bedrock_model_provider = model.split(".")[
        0
    ]  # grab the model provider from the first part of the model id

    if bedrock_model_provider == "anthropic":  # Anthropic model
        return {  # anthropic
            "max_tokens_to_sample": 1000,
            "temperature": 0.0,
            "top_k": 250,
            "top_p": 0.999,
            "stop_sequences": ["\n\nHuman:"],
        }

    elif bedrock_model_provider == "ai21":  # AI21
        return {  # AI21
            "maxTokens": 512,
            "temperature": 0,
            "topP": 0.5,
            "stopSequences": [],
            "countPenalty": {"scale": 0},
            "presencePenalty": {"scale": 0},
            "frequencyPenalty": {"scale": 0},
        }

    elif bedrock_model_provider == "cohere":  # COHERE
        return {
            "max_tokens": 512,
            "temperature": 0,
            "p": 0.01,
            "k": 0,
            "stop_sequences": [],
            "return_likelihoods": "NONE",
        }

    else:  # Amazon
        # For the LangChain Bedrock implementation, these parameters will be added to the
        # textGenerationConfig item that LangChain creates for us
        return {
            "maxTokenCount": 512,
            "stopSequences": [],
            "temperature": 0,
            "topP": 0.9,
        }
