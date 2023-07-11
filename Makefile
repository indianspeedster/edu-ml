all: introduction.ipynb

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