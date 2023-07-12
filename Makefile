all: dataset_1.ipynb dataset_2.ipynb introduction.ipynb

clean: 
	rm final_notebook_augmentation.ipynb

final_notebook.ipynb: markdown/*.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
                -i markdown/notebook.md \
                -o final_notebook.ipynb  
	sed -i 's/attachment://g' final_notebook.ipynb

introduction.ipynb: markdown/notebooks/introduction.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
                -i markdown/notebooks/introduction.md \
                -o notebooks/introduction.ipynb  
	sed -i 's/attachment://g' notebooks/introduction.ipynb

dataset_1.ipynb: markdown/notebooks/dataset_1.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
                -i markdown/notebooks/dataset_1.md \
                -o notebooks/dataset_1.ipynb  
	sed -i 's/attachment://g' notebooks/dataset_1.ipynb

dataset_2.ipynb: markdown/notebooks/dataset_2.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
                -i markdown/notebooks/dataset_2.md \
                -o notebooks/dataset_2.ipynb  
	sed -i 's/attachment://g' notebooks/dataset_2.ipynb