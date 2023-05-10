from executor import PDFIndexer

docs = DocumentArray.from_files("../Data/*.pdf", recursive=True)

flow = Flow().add(uses=PDFIndexer, name="indexer")

with flow:
    indexed_docs = flow.index(docs)
    print(indexed_docs[0].chunks)
    for chunk in indexed_docs[0].chunks:
        print(chunk)