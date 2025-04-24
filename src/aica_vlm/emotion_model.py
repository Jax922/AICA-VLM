class EmotionModel:
    def __init__(self, model_name="6_basic"):
        self.models = {
            "6_basic": ["Joy", "Surprise", "Anger", "Disgust", "Fear", "Sadness"],
            "8_expanded": [
                "Amusement",
                "Awe",
                "Contentment",
                "Excitement",
                "Anger",
                "Disgust",
                "Fear",
                "Sadness",
            ],
            "26_fine_grained": [
                "Affection",
                "Anger",
                "Annoyance",
                "Anticipation",
                "Aversion",
                "Confidence",
                "Disapproval",
                "Disconnection",
                "Disquietment",
                "Doubt/Confusion",
                "Embarrassment",
                "Engagement",
                "Esteem",
                "Excitement",
                "Fatigue",
                "Fear",
                "Happiness",
                "Pain",
                "Peace",
                "Pleasure",
                "Sadness",
                "Sensitivity",
                "Suffering",
                "Surprise",
                "Sympathy",
                "Yearning",
            ],
            "VA": ["Valence", "Arousal"],
        }
        self.model_name = model_name

    def get_labels(self):
        return self.models.get(self.model_name, [])

    def list_models(self):
        return list(self.models.keys())
