import numpy as np

import bentoml
from bentoml.io import NumpyNdarray


model_ref = bentoml.sklearn.get("mlzoomcamp_homework:jsi67fslz6txydu5")

model_runner = model_ref.to_runner()

svc = bentoml.Service("mlzoomcamp_homework", runners=[model_runner])


@svc.api(input=NumpyNdarray(shape=(-1, 4), dtype=np.float32, enforce_shape=True, enforce_dtype=True), output=NumpyNdarray())
async def classify(vector):
    prediction = await model_runner.predict.async_run(vector)
    result = prediction[0]

    return result