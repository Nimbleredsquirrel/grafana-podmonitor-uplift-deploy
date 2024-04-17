import logging

from server_ml import MLService, data_types
from server_ml.utils import fetch_model_path
from server_ml.encoders import TextEncoder
import joblib
import numpy as np
import json

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


class UpliftPredictor(MLService):
    def __init__(self):
        super().__init__()
        self.uplift_model = None
        self.is_ready = False

    async def setup_model(self):
        model_path = await fetch_model_path(self._configurations)
        self.uplift_model = joblib.load(model_path)
        self.is_ready = True
        return self.is_ready

    def _parse(self, request):
        inputs = {}
        for sht in payload.inputs:
            inputs[sht.name] = json.loads(
                "".join(self.decode(sht, default_codec=StringCodec))
            )

        return inputs

    async def make_prediction(self, request):
        try:
            decoded_request = self._parse(request)\
            .get("prediction_request", {})
            feature_array = np.array(decoded_request.get("features", []))
            prediction = self.uplift_model.predict(feature_array)
            prediction_result = {"success": True, 
                                "prediction": prediction}

        except Exception as e:
            prediction_result = {"success": False, 
                                 "prediction": None, 
                                 "error": str(e)}

        response_data = json.dumps(prediction_result)

        return data_types.Response(
            id=request.id,
            model_name=self.name,
            model_version=self.version,
            outputs=[
                data_types.Output(
                    name="prediction_result",
                    shape=[len(response_data)],
                    dtype="BYTES",
                    data=[response_data.encode("UTF-8")],
                    params=data_types.\
                    Parameters(content_type="application/json"),
                )
            ],
        )