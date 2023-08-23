MARKDOWN_FOLDER = markdown/data_preprocessing
NOTEBOOK_FOLDER = notebooks

MARKDOWN_FILES = $(wildcard $(MARKDOWN_FOLDER)/*.md)
NOTEBOOK_FILES = $(patsubst $(MARKDOWN_FOLDER)/%.md, $(NOTEBOOK_FOLDER)/%.ipynb, $(MARKDOWN_FILES))

convert: $(NOTEBOOK_FILES)

$(NOTEBOOK_FOLDER)/%.ipynb: $(MARKDOWN_FOLDER)/%.md
	pandoc --embed-resources --standalone --wrap=none $< -o $@ 
	

clean:
	rm -f  $(NOTEBOOK_FILES)
