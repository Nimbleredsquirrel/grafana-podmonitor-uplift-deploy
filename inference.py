import logging

import joblib
import numpy as np
from mlserver import MLModel
from mlserver.codecs import NumpyCodec
from mlserver.types import InferenceRequest, InferenceResponse, ResponseOutput

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


class UpliftPredictor(MLModel):
    async def load(self) -> bool:
        model_uri = self.settings.parameters.uri
        self.uplift_model = joblib.load(model_uri)
        self.ready = True
        return self.ready

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        input_data = NumpyCodec.decode_input(payload.inputs[0])
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)

        predictions = self.uplift_model.predict(input_data)

        return InferenceResponse(
            id=payload.id,
            model_name=self.name,
            model_version=self.version,
            outputs=[
                ResponseOutput(
                    name="predictions",
                    shape=list(predictions.shape),
                    datatype="FP64",
                    data=predictions.tolist(),
                )
            ],
        )
