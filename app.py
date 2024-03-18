from flask import Flask, request, session, jsonify
from werkzeug.utils import secure_filename
import os
import os
import uuid
import torch
import transformers
from transformers import (
AutoTokenizer,
AutoModelForCausalLM,
BitsAndBytesConfig,
pipeline
)
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_transformers import Html2TextTransformer
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain, RetrievalQA

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
ALLOWED_EXTENSIONS = {'pdf'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

#Loading the Llama-2 Model
model_name='meta-llama/Llama-2-7b-chat-hf'
model_config = transformers.AutoConfig.from_pretrained(model_name,)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

#################################################################
# bitsandbytes parameters
#################################################################

# Activate 4-bit precision base model loading
use_4bit = True
# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"
# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"
# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

#################################################################
# Set up quantization config
#################################################################
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
load_in_4bit=use_4bit,
bnb_4bit_quant_type=bnb_4bit_quant_type,
bnb_4bit_compute_dtype=compute_dtype,
bnb_4bit_use_double_quant=use_nested_quant,
)
# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

#################################################################
# Load pre-trained config
#################################################################
model = AutoModelForCausalLM.from_pretrained(
model_name,
quantization_config=bnb_config,
)

# Building a LLM QNA chain
text_generation_pipeline = transformers.pipeline(
model=model,
tokenizer=tokenizer,
task="text-generation",
temperature=0.2,
repetition_penalty=1.1,
return_full_text=True,
max_new_tokens=300,
)

llama_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
file_id = None
retrieval_chain = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    global file_id
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_id = str(uuid.uuid4())
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file_id)
        file.save(filepath)

        # Placeholder for your PDF processing logic
        process_pdf(filepath)

        return 'File uploaded & processed successfully. You can begin querying now', 200

def process_pdf(filepath):
    global retrieval_chain
    # Loading the splitting the document #
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file_id)
    loader = PyPDFLoader(filepath)
    docs = loader.load_and_split()
    # Chunk text
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunked_documents = text_splitter.split_documents(docs)

    # Load chunked documents into the Qdrant index
    db = Qdrant.from_documents(chunked_documents, HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'), location= ":memory:")
    retriever = db.as_retriever()
    retrieval_chain = RetrievalQA.from_llm(llm= llama_llm, retriever= retriever)

@app.route('/query', methods=['POST'])
def query():
    global retrieval_chain
    data = request.json
    query = data.get('query')

    return jsonify(retrieval_chain.run(query)), 200

if __name__ == '__main__':
    app.run(debug=True)