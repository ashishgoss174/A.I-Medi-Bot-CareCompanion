import os
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.language_models import BaseLLM
from pydantic import Field, PrivateAttr
from dotenv import load_dotenv, find_dotenv
from langchain_core.outputs import LLMResult

load_dotenv(find_dotenv())

# Step 1: Setup LLM (Mistral with HuggingFace)
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

class CustomLLM(BaseLLM):
    model_id: str = Field(default=HUGGINGFACE_REPO_ID)
    token: str = Field(default=HF_TOKEN)
    temperature: float = Field(default=0.5)
    max_new_tokens: int = Field(default=512)

    #Define client as a private attribute
    _client: InferenceClient = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = InferenceClient(model=self.model_id, token=self.token)

    def _generate(self, prompts, stop=None) -> LLMResult:
        """Implements required `_generate()` method."""
        responses = []
        for prompt in prompts:
            response = self._client.text_generation(
                prompt, max_new_tokens=self.max_new_tokens, temperature=self.temperature
            )
            responses.append(response)
        
        return LLMResult(generations=[[{"text": r}] for r in responses])

    @property
    def _identifying_params(self):
        return {"model_id": self.model_id, "temperature": self.temperature}

    @property
    def _llm_type(self):
        return "custom_huggingface"

# Load LLM
llm = CustomLLM()

# Step 2: Connect LLM with FAISS and Create chain
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk, please.
"""

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# Load Database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Now invoke with a single query
user_query = input("Write Query Here: ")
response = qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])