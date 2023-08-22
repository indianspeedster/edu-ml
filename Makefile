MARKDOWN_FOLDER = markdown_md\data_preprocessing
NOTEBOOK_FOLDER = notebook_md

MARKDOWN_FILES = $(wildcard $(MARKDOWN_FOLDER)/*.md)
NOTEBOOK_FILES = $(patsubst $(MARKDOWN_FOLDER)/%.md, $(NOTEBOOK_FOLDER)/%.ipynb, $(MARKDOWN_FILES))

convert: $(NOTEBOOK_FILES)

$(NOTEBOOK_FOLDER)/%.ipynb: $(MARKDOWN_FOLDER)/%.md
	pandoc --embed-resources --standalone --wrap=none $< -o $@ 
	

clean:
	rm -f  $(NOTEBOOK_FILES)
