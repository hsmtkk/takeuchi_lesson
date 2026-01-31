# GitHub: https://github.com/naotaka1128/llm_app_codes/chapter_010/build_qa_vectorstore.py


import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv

load_dotenv()


def main():
    # CSVファイルから「よくある質問」を読み込む
    # qa_df = pd.read_csv("./data/bearmobile_QA.csv")  # question,answer
    qa_df = pd.read_csv("./data/bearmobile_QA_small.csv")  # question,answer

    # ベクトルDBに書き込むデータを作る
    qa_texts = []
    for _, row in qa_df.iterrows():
        qa_texts.append(f"question: {row['question']}\nanswer: {row['answer']}")

    # 上記のデータをベクトルDBに書き込む
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    db = FAISS.from_texts(qa_texts, embeddings)
    db.save_local("./vectorstore/qa_vectorstore")


if __name__ == "__main__":
    main()
    print("done")
