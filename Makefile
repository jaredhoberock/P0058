.SUFFIXES: .md .html .pdf .docx

%.pdf: %.md; pandoc -o $@ $< --number-sections -H header.tex

%.html: %.md template.html; pandoc -t html5 -o $@ $< --smart --template=template.html

%.docx: %.md; pandoc -o $@ $< --smart --number-sections

default: P0058_an_interface_for_abstracting_execution.pdf

all: P0058_an_interface_for_abstarcting_execution.pdf P0058_an_interface_for_abstracting_execution.html P0058_an_interface_for_abstracting_execution.docx
