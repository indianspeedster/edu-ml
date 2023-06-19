all: final_notebook.ipynb

clean: 
	rm final_notebook.ipynb

final_notebook.ipynb: markdown/*.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
                -i markdown/notebook.md \
                -o final_notebook.ipynb  
	sed -i 's/attachment://g' final_notebook.ipynb