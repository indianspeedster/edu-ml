all: clean intro dataset dataprep args train read_me reserve

dataprep:
	pandoc --resource-path=markdown/ --embed-resources --standalone --wrap=none -i markdown/data_preprocessing/3_data_preprocessing_1.md \
	markdown/data_preprocessing/footer.md \
		 -o notebooks/3_data_preprocessing_1.ipynb
	pandoc --resource-path=markdown/ --embed-resources --standalone --wrap=none -i markdown/data_preprocessing/3_data_preprocessing_2.md \
	markdown/data_preprocessing/footer.md \
		 -o notebooks/3_data_preprocessing_2.ipynb

intro: 
	pandoc --resource-path=markdown/ --embed-resources --standalone --wrap=none -i markdown/intro/1_introduction.md \
	markdown/intro/footer.md \
		 -o notebooks/1_introduction.ipynb

dataset:
	pandoc --resource-path=markdown/ --embed-resources --standalone --wrap=none -i markdown/dataset/2_dataset_1.md \
	markdown/dataset/footer.md \
		 -o notebooks/2_dataset_1.ipynb
	pandoc --resource-path=markdown/ --embed-resources --standalone --wrap=none -i markdown/dataset/2_dataset_2.md \
	markdown/dataset/footer.md \
		 -o notebooks/2_dataset_2.ipynb
args:
	pandoc --resource-path=markdown/ --embed-resources --standalone --wrap=none -i markdown/train_args/4_train_args_1.md \
	markdown/train_args/footer.md \
		 -o notebooks/4_train_args_1.ipynb
	pandoc --resource-path=markdown/ --embed-resources --standalone --wrap=none -i markdown/train_args/4_train_args_2.md \
	markdown/train_args/footer.md \
		 -o notebooks/4_train_args_2.ipynb

train:
	pandoc --resource-path=markdown/ --embed-resources --standalone --wrap=none -i markdown/model_train/5_model_training_1.md \
	markdown/model_train/footer.md \
		 -o notebooks/5_model_training_1.ipynb
	pandoc --resource-path=markdown/ --embed-resources --standalone --wrap=none -i markdown/model_train/5_model_training_2.md \
	markdown/model_train/footer.md \
		 -o notebooks/5_model_training_2.ipynb
read_me:
	pandoc --resource-path=/ --embed-resources --standalone --wrap=none -i README.md -o README.ipynb

reserve:
	pandoc --resource-path=/ --embed-resources --standalone --wrap=none -i reserve.md -o reserve.ipynb


clean: 
	rm -f notebooks/*.ipynb
	rm -f *.ipynb