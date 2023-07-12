from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import WebBaseLoader
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector
import os
from bs4 import BeautifulSoup
import requests

urls = ['https://www.keypress.co.in']
# all_urls = collect_urls(root_url='https://www.keypress.co.in')
os.environ['OPENAI_API_KEY'] = 'sk-**********'
# loaders = WebBaseLoader(urls)
# data = loaders.load()
response = requests.get('https://www.terabits.xyz/')
content_type = response.headers.get('Content-Type') or ''
if 'text/html' in content_type:
    soup = BeautifulSoup(response.content, "html.parser")
    text = soup.text.replace('\t','')
    clean_text = '\n'.join(line for line in text.split('\n') if line)
    clean_text = ' '.join(line for line in clean_text.split(' ') if line)

text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1536, chunk_overlap=20)
a = """ - The evaluation may need other threads running while it's running:
    In this case, it's possible to set the PYDEVD_UNBLOCK_THREADS_TIMEOUT
    environment variable so that if after a given timeout an evaluation doesn't finish,
    other threads are unblocked or you can manually resume all threads.

    Alternatively, it's also possible to skip breaking on a particular thread by setting a
    `pydev_do_not_trace = True` attribute in the related threading.Thread instance
    (if some thread should always be running and no breakpoints are expected to be hit in it).

- The evaluation is deadlocked:
    In this case you may set the PYDEVD_THREAD_DUMP_ON_WARN_EVALUATION_TIMEOUT
    environment variable to true so that a thread dump is shown along with this message and
    optionally, set the PYDEVD_INTERRUPT_THREAD_TIMEOUT to some value so that the debugger
    tries to interrupt the evaluation (if possible) when this happens."""
docs = text_splitter.create_documents([a])

embeddings = OpenAIEmbeddings()

CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver=os.environ.get("PGVECTOR_DRIVER", "psycopg"),
    host=os.environ.get("PGVECTOR_HOST", "192.168.29.15"),
    port=int(os.environ.get("PGVECTOR_PORT", "6432")),
    database=os.environ.get("PGVECTOR_DATABASE", "matastore"),
    user=os.environ.get("PGVECTOR_USER", "postgres"),
    password=os.environ.get("PGVECTOR_PASSWORD", "admin"),
)

db = PGVector.from_documents(
    embedding=embeddings,
    documents=docs,
    collection_name="state_of_the_union",
    connection_string=CONNECTION_STRING,
)

llm = OpenAI(temperature=0)

chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=db.as_retriever())

# while True:
#     qns = input('Question: ')
#     ans = chain({"question": qns }, return_only_outputs=True)
#     print(ans['answer'])