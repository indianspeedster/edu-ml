::: {.cell .markdown}
## Setting up Training parameters and Hyperparameters

If you have selected this notebook then it is assumed that you have selected to go ahead with AdamW (Adam with weight decay) as the optimizer.
:::


::: {.cell .markdown}

### Training Parameters and Hyperparameters Explanation

1.  **`output_dir`**: Specifies the directory where model checkpoints and training logs will be saved.

2.  **`evaluation_strategy`**: Defines the strategy for evaluating the model during training. 

3.  **`save_strategy`**: Specifies when to save model checkpoints. 

4.  **`learning_rate`**: Determines the step size at which the optimizer adjusts model weights during training.

5.  **`per_device_train_batch_size`**: Specifies the batch size for training data per GPU, impacting memory usage and computational efficiency.

6.  **`per_device_eval_batch_size`**: Sets the batch size for evaluation data per GPU, affecting memory and computation during evaluation.

7.  **`num_train_epochs`**: Indicates the total number of training epochs, which are complete passes through the training dataset.

8.  **`warmup_ratio`**: Determines the ratio of warmup steps to the total number of training steps, helping the optimizer to smoothly adapt in the initial stages of training.

9.  **`weight_decay`**: Introduces L2 regularization to the optimizer, helping to prevent overfitting by penalizing large model weights.

10. **`load_best_model_at_end`**: Specifies whether to load the best model based on the chosen evaluation metric at the end of training.

11. **`metric_for_best_model`**: Specifies the evaluation metric used to determine the best model, which is set to \"accuracy\" in this case.

12. **`save_total_limit`**: Sets the maximum number of model checkpoints to keep, preventing excessive storage usage.

13. **`logging_dir`**: Designates the directory where training logs, such as training progress and performance metrics, will be stored.

14. **`optimizers`**: Specifies the optimizers used for model parameter updates during training, allowing for gradient-based optimization algorithms like AdamW, SGD and many more.
:::

::: {.cell .markdown}
## Implementation
:::

::: {.cell .markdown}
### Importing relevent Library
:::

::: {.cell .code}
```python
import json

```
:::

::: {.cell .markdown}
### Setting up the training arguments
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
    "save_total_limit": 1,
    "logging_dir": "./logs",
    "optimizer": "AdamW"
}


```
:::

::: {.cell .markdown}
### Storing the dictionary to a json file
:::

::: {.cell .code}
```python

with open('args.json', 'w') as args_file:
    json.dump(args, args_file, indent=4)

```
:::


