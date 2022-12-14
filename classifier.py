from transformers import AutoFeatureExtractor, AutoModelForImageClassification, pipeline


class Classifier:
    def __init__(self):
        name = "facebook/convnext-tiny-224"
        extractor = AutoFeatureExtractor.from_pretrained(name)
        model = AutoModelForImageClassification.from_pretrained(name)
        self.__classifier = pipeline(
            "image-classification", model=model, feature_extractor=extractor
        )

    def classify(self, image):
        raw_results = self.__classifier(image)
        results = list(
            map(lambda item: dict.fromkeys([item["label"]], item["score"]), raw_results)
        )

        return results
