base_generator.py
│       │   │   ├── llm_generator.py
│       │   │   └── rag_generator.py


Install editable
    pip install -e .

faiss-cpu==1.7.4.post2 - fails on mac so,
    brew install swig cmake     
    pip install --no-binary :all: faiss-cpu  #compiles C++