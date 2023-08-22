::: {.cell .markdown}
## Implementation
:::

::: {.cell .markdown}
#### Importing relevent Library
:::

::: {.cell .code}
```python
import json

```
:::

::: {.cell .markdown}
#### Setting up the training arguments
:::

::: {.cell .code}
```python
args = {
    "output_dir": "./output",
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "num_train_epochs": 10,
    "warmup_ratio": 0.1,
    "weight_decay": 0.001,
    "load_best_model_at_end": True,
    "metric_for_best_model": "accuracy",
    "save_total_limit"=1,
    "logging_dir": "./logs",
    "optimizer": "sgd"
}


```
:::

::: {.cell .markdown}
#### Storing the dictionary to a json file
:::

::: {.cell .code}
```python

with open('args.json', 'w') as args_file:
    json.dump(args, args_file, indent=4)

```
:::