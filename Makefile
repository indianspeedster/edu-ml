SOURCES := $(wildcard *.md)

NBS := $(patsubst %.md,%.ipynb,$(SOURCES))

%.ipynb: %.md
	pandoc  --self-contained --wrap=none  -i notebooks/title.md $^ -o $@

all: clean intro dataset dataprep args train start_here

notebooks: $(NBS)

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
start_here:
	pandoc --resource-path=/ --embed-resources --standalone --wrap=none -i start_here.md -o start_here.ipynb


clean: 
	rm -f notebooks/*.ipynb
	rm -f *.ipynb