# RAG-Chatbot

### What is RAG?
Retrieval-Augmented Generation (RAG) is the process of optimizing the output of a large language model, so it references an authoritative knowledge base outside of its training data sources before generating a response. RAG extends the already powerful capabilities of LLMs to specific domains or an organization's internal knowledge base, all without the need to retrain the model. It is a cost-effective approach to improving LLM output so it remains relevant, accurate, and useful in various contexts. Primarily focussed on text generation tasks - text summarisation, text generation, question answering etc

### How to run?

##### Install the requirements

pip install -r requirements.txt

##### Create the vector db locally

python vector.py

##### Run the streamlit app for the bot
python -m streamlit run chatbot.py

### Note
Please add your open api key when testing locally in chatbot.py
