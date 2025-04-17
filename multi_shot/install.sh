#!/bin/bash
# installing python libraries

##***note***: libraries cannot be installed with python3.13
##successfull with python 3.11

pip install requests;
pip install dotenv;
pip install langchain langchain-google-genai langchain-community chromadb;
pip install pyarrow --only-binary :all:	#unable to be indirectly installed when called from streamlit install
pip install streamlit
 
