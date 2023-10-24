import sys
import interpreter
from langchain.embeddings import OpenAIEmbeddings
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer

embeddings = OpenAIEmbeddings(openai_api_key="")
model = SentenceTransformer('sentence-transformers/allenai-specter', device='cpu')
query = sys.argv[1]
depth = int(sys.argv[2])
name = sys.argv[3]

EMBEDDING_DIM = 1536
def get_embeddings_for_text(text):
    return embeddings.embed_query(text)
def load_index_map():
    index_map = {}
    with open('annoys/' + name + '_index_map' + '.txt', 'r') as f:
        for line in f:
            idx, path = line.strip().split('	')
            index_map[int(idx)] = path
    return index_map

def query_top_files(query, top_n=4):
    # Load annoy index and index map
    t = AnnoyIndex(EMBEDDING_DIM, 'angular')
    t.load('annoys/' + name + '_ada.ann')
    index_map = load_index_map()
    # Get embeddings for the query
    query_embedding = get_embeddings_for_text(query)
    # Search in the Annoy index
    indices, distances = t.get_nns_by_vector(query_embedding, top_n, include_distances=True)
    # Fetch file paths for these indices
    files = [(index_map[idx], dist) for idx, dist in zip(indices, distances)]
    return files

def query_top_files_specter(query, top_n=4):
    # Load annoy index and index map
    t = AnnoyIndex(768, 'angular')
    t.load('annoys/' + name + '_specter.ann')
    index_map = load_index_map()
    # Get embeddings for the query
    query_embedding = model.encode(query)
    # Search in the Annoy index
    indices, distances = t.get_nns_by_vector(query_embedding, top_n, include_distances=True)
    # Fetch file paths for these indices
    files = [(index_map[idx], dist) for idx, dist in zip(indices, distances)]
    return files

def get_file_contents(path):
    with open(path, 'r') as f:
        return f.read()
results = query_top_files(query, depth)
results_specter = query_top_files_specter(query, depth)
results = results + results_specter

file_content = ""
s = set()

print ("Files you might want to read:")
for path, dist in results:

    content = get_file_contents(path)
    file_content += "Path: "
    file_content += path
    if (path not in s):
        print (path)
        s.add(path)
    file_content += f"\n```\n{content}\n```\n\n"

print("open interpreter's recommendation")
language = "Japanese"

# print("file_content:", file_content)

if (language == "Japanese"):
    message = "あなたは完了すべきタスクがあります。 以下のタスクとそのパスを読んで、質問に答えてください。 タスク：以下のファイルを読んで、質問に答えてください。質問：" + query + "\n" + file_content + "\n" + "日本語で答えてください。"
else:
    message = "You have a task to complete. Please help with the task below and answer my question. Task : READ THE FILE content below and their paths and answer " + query + "\n" + file_content

interpreter.auto_run = True

count = 0
while True:
    if (count == 0):
        interpreter.chat(message)
    else:
        message = input("Enter your message: ")
        interpreter.chat(message)
    count += 1

# print ("interpreter's recommendation done. (Risk: LLMs are known to hallucinate)")
