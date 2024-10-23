#!/bin/bash

DIR="$(pwd)"

yes |  sh -c "$(curl -fsSL https://ftp.ncbi.nlm.nih.gov/entrez/entrezdirect/install-edirect.sh)"

export PATH=${HOME}/edirect:${PATH}


#TODO: Run a simple test command before running the final command
#TODO: Take query as input, with for ex: --query flag
#TODO: Take file name as input, with for ex: --output flag

#TODO: inlcude file name, file path in .gitignore 

#TODO: Prints statetemnt for verbosity, status code for final command
esearch -db pubmed -query "(intelligence[Title/Abstract]) AND (("2013"[Date - Publication] : "2023"[Date - Publication]))" | efetch -format xml | xtract -pattern PubmedArticle -element MedlineCitation/PMID -block Abstract -sep " " -element AbstractText -block PubMedPubDate -if @PubStatus -equals "medline"  -sep "/" -element Year,Month,Day -block Author -sep " " -tab ", " -element ForeName,LastName >> "$DIR/output.txt"
