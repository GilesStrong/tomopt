
Commands assume you are inside the docs folder.
Build the docs .rst source files with: `sphinx-apidoc -l -o ./source/ ../tomopt -f -d 1`
And then remove all the "Module contents" sections, since these create copies of everything.
To build the docs, run `make html`