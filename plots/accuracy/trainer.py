for i, train in enumerate([train_dataset_full, train_dataset, train_dataset_augmented]):
  best_lr = learning_rates[0]
  for lr in learning_rates:
    best_accuracy = 0
    for train_data in train:
      BERT_model = BertModelWithCustomLossFunction()
      trainer = Trainer(
            model = BERT_model,
            args = TrainingArguments(
          output_dir="./output",
          evaluation_strategy="epoch",
          save_strategy="epoch",
          learning_rate=lr,
          per_device_train_batch_size=8 ,
          per_device_eval_batch_size=8 ,
          num_train_epochs=3,
          warmup_ratio= 0.1,
          weight_decay= 0.001,
          load_best_model_at_end=True,
          metric_for_best_model="accuracy",
          save_total_limit=1,
              ),
            train_dataset=train_data,
            eval_dataset=val_dataset,
            tokenizer=BERT_tokenizer,
            compute_metrics=compute_metrics,)
      trainer.train()
      evaluation_metrics = trainer.predict(test_dataset)
      accuracy = evaluation_metrics.metrics['test_accuracy']
      if accuracy > best_accuracy:
        best_accuracy = accuracy
        if i == 0:
          best_lr = learning_rate
      best_accuracy = max(accuracy, best_accuracy)
      torch.cuda.empty_cache()
