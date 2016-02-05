This is the source of the ISO C++ paper P0058.

Build the paper with

    $ pandoc --number-sections -H header.tex P0058_an_interface_for_abstracting_execution.md -o P0058_an_interface_for_abstracting_execution.pdf

or simply

    $ make

Alternate Word and HTML formats can be generated with:

    $ make P0058_an_interface_for_abstracting_execution.docx
    $ make P0058_an_interface_for_abstracting_execution.html
