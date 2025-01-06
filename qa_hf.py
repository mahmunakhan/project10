# Use a pipeline as a high-level helper
from transformers import pipeline
import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from get_embedding_function import get_embedding_function

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("riotu-lab/ArabianGPT-0.3B-QA")
model = AutoModelForCausalLM.from_pretrained("riotu-lab/ArabianGPT-0.3B-QA")

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
أجب عن السؤال بناءً فقط على السياق التالي:

{context}

---

أجب عن السؤال بناءً على السياق أعلاه: {question}
"""

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="نص الاستفسار.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Use the pipeline to get the response
    response = pipe(question=query_text, context=context_text)

    response_text = response['answer']  # Extract the answer from the response

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"الرد: {response_text}\nالمصادر: {sources}"
    print(formatted_response)
    return response_text

if __name__ == "__main__":
    main()
