import streamlit as st
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, pipeline


@st.cache(allow_output_mutation=True)
def get_classifier():
    name = "facebook/convnext-tiny-224"
    extractor = AutoFeatureExtractor.from_pretrained(name)
    model = AutoModelForImageClassification.from_pretrained(name)
    classifier = pipeline(
        "image-classification", model=model, feature_extractor=extractor
    )

    return classifier


def classify(image):
    classifier = get_classifier()
    raw_results = classifier(image)
    results = list(
        map(lambda item: dict.fromkeys([item["label"]], item["score"]), raw_results)
    )

    return results
