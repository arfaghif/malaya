import tensorflow as tf
from malaya.torch_model.huggingface import (
    Generator,
    Prefix,
    Paraphrase,
    Summarization,
    Similarity,
    ZeroShotClassification,
    ZeroShotNER,
    ExtractiveQA,
)
from transformers import AutoTokenizer
from malaya_boilerplate.utils import check_tf2


@check_tf2
def load_automodel(model, model_class, huggingface_class=None, **kwargs):
    try:
        from transformers import TFAutoModel, AutoTokenizer
    except BaseException:
        raise ModuleNotFoundError(
            'transformers not installed. Please install it by `pip3 install transformers` and try again.'
        )

    tokenizer = AutoTokenizer.from_pretrained(model)
    if huggingface_class is None:
        huggingface_class = TFAutoModel
    model = huggingface_class.from_pretrained(model)
    return model_class(model=model, tokenizer=tokenizer, **kwargs)


def load_generator(model, initial_text, **kwargs):
    return Generator(model, initial_text)


def load_prefix(model, **kwargs):
    return Prefix(model)


def load_paraphrase(model, initial_text, **kwargs):
    return Paraphrase(model, initial_text)


def load_summarization(model, initial_text, **kwargs):
    return Summarization(model, initial_text)


def load_similarity(model, **kwargs):
    return Similarity(model=model)


def load_zeroshot_classification(model, **kwargs):
    return ZeroShotClassification(model=model)


def load_zeroshot_ner(model, **kwargs):
    return ZeroShotNER(model=model)


def load_extractive_qa(model, **kwargs):
    return ExtractiveQA(model=model)
