"""Create a ChatVectorDBChain for question/answering."""
import os

from langchain.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ChatVectorDBChain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.vectorstores.base import VectorStore

from prompt import prompt

# load text-generation webgui API (local, mimic openai)
os.environ["OPENAI_API_KEY"] = 'dummy'
os.environ["OPENAI_API_BASE"] = 'http://localhost:5001/v1'


kwargs = {"model_path": "C:/tuanshu/languege_models/gpt4all/gpt4all-lora-quantized-ggml.bin",
          # "model_path": "C:/tuanshu/languege_models/vicuna-13B-1.1-GPTQ-4bit-128g-GGML/vicuna-13B-1.1-GPTQ-4bit-128g.GGML.bin",
          "verbose": True,
          "n_ctx": 2048,
          "max_tokens": 1024,
          "seed": 17,
          "streaming": True}


def get_chain(
    vectorstore: VectorStore, question_handler, stream_handler, tracing: bool = False
) -> ChatVectorDBChain:
    """Create a ChatVectorDBChain for question/answering."""
    # Construct a ChatVectorDBChain with a streaming llm for combine docs
    # and a separate, non-streaming llm for question generation
    manager = AsyncCallbackManager([])
    question_manager = AsyncCallbackManager([question_handler])
    stream_manager = AsyncCallbackManager([stream_handler])
    if tracing:
        tracer = LangChainTracer()
        tracer.load_default_session()
        manager.add_handler(tracer)
        question_manager.add_handler(tracer)
        stream_manager.add_handler(tracer)

    question_gen_llm = OpenAI(
        streaming=True,
        temperature=0,
        verbose=True,
        callback_manager=question_manager,
    )

    question_generator = LLMChain(
        llm=question_gen_llm, prompt=prompt, callback_manager=manager,
    )
    # doc_chain = load_qa_chain(
    #     streaming_llm, chain_type="stuff", prompt=prompt, callback_manager=manager
    # )

    # qa = ChatVectorDBChain(
    #     vectorstore=vectorstore,
    #     combine_docs_chain=doc_chain,
    #     question_generator=question_generator,
    #     callback_manager=manager,
    # )
    return question_generator
