# RAG-Chatbot

### What is RAG?
Retrieval-Augmented Generation (RAG) enhances the output of large language models by incorporating information from an authoritative knowledge base beyond their initial training data. This approach leverages the existing strengths of LLMs, extending their capabilities to specific domains or an organization's internal knowledge base without requiring model retraining. RAG offers a cost-effective method for improving the relevance, accuracy, and usefulness of LLM outputs in various contexts, particularly in text generation tasks such as summarization, generation, and question answering.

### How to run?

##### Install the requirements

pip install -r requirements.txt

##### Create the vector db locally

python vector.py

##### Run the streamlit app for the bot
python -m streamlit run chatbot.py

### Note
Please add your open api key when testing locally in chatbot.py
