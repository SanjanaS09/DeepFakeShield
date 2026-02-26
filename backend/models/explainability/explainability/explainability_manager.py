from models.explainability.shap_explainer import ShapExplainer


class ExplainabilityManager:

    def __init__(self, model):
        """
        model: trained detection model
        """

        self.shap_explainer = ShapExplainer(model)


    def explain_image(self, input_tensor):
        """
        Generate explanation for image input
        """

        result = self.shap_explainer.explain(input_tensor)

        return {
            "shap_explanation": result
        }
