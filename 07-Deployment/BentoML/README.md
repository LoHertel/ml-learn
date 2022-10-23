# Deploy ML model using BentoML

## Training

Train a model (e.g. xgboost) and save it as BentoML model:
```py
bentoml.xgboost.save_model(
    'credit_risk_model',
    model,
    custom_objects={
        'dictVectorizer': dv
    })
```

## Deployment

Start BentoML server in development mode (auto-reload on file changes)

```bash
bentoml serve service.py:svc --reload
```


## Use Model

Open http://127.0.0.1:3000/ and use the /classify endpoint for a prediction.

Use this dictionary as input data:
```json
{
  "seniority": 3,
  "home": "owner",
  "time": 36,
  "age": 26,
  "marital": "single",
  "records": "no",
  "job": "freelance",
  "expenses": 35,
  "income": 0.0,
  "assets": 60000.0,
  "debt": 3000.0,
  "amount": 800,
  "price": 1000
}
```

The response is:
```json
{
  "status": "MAYBE"
}
```


List all BentoML models:
```bash
bentoml models list
```
Output:
```
 Tag                                 Module           Size        Creation Time       
 credit_risk_model:pxusypssuglwoaav  bentoml.xgboost  197.77 KiB  2022-10-23 09:08:27 
```

Show specific dependencies for a model:
```bash
bentoml models get credit_risk_model:pxusypssuglwoaav
```

```yaml
name: credit_risk_model                                        
version: pxusypssuglwoaav
module: bentoml.xgboost
labels: {}
options:
  model_class: Booster
metadata: {}
context:
  framework_name: xgboost
  framework_versions:
    xgboost: 1.6.2
  bentoml_version: 1.0.7
  python_version: 3.10.6
signatures:
  predict:
    batchable: false
api_version: v2
creation_time: '2022-10-23T07:08:27.400420+00:00'
```

## Build BentoML deployment

Source: bentofile.yaml

```bash
bentoml build
```