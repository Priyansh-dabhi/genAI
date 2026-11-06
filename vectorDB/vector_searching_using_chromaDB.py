from certifi import where
import chromadb
from chromadb.utils import embedding_functions

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MINILM-L6-v2"
)



client = chromadb.Client()

collection = client.create_collection(
            name= "filer_demo",
            metadata= {
                "description": "used to demo filtering  in chromaDB",
                "hnsw:space": "cosine"
                },
            configuration= {
                "embedding_function": ef
            },
        )
print(f"collection created: {collection.name}")

def main():
    try:

        # Array of grocery-related text items
        texts = [
            'fresh red apples',
            'organic bananas',
            'ripe mangoes',
            'whole wheat bread',
            'farm-fresh eggs',
            'natural yogurt',
            'frozen vegetables',
            'grass-fed beef',
            'free-range chicken',
            'fresh salmon fillet',
            'aromatic coffee beans',
            'pure honey',
            'golden apple',
            'red fruit'
        ]

        ids = [f"food_{index+1}" for index, _ in enumerate(texts)]
        
        collection.add(
            documents=texts,
            metadatas= [{"source": "grocery_store", "category":"food"} for _ in texts],
            ids= ids
        )
        all_items = collection.get()

        print("Collection Contents: ")
        print(f"Number of docs: {len(all_items['documents'])}")
        # print(f"Number of docs: {all_items}")

    except Exception as error:
        print(f"Got Error: {error}")


all_items = collection.get()


def similarity_search(collection, all_items):
    try:
        query_term = 'apple'

        result = collection.query(
            query_texts= [query_term],
            n_results = 3
        )
        print(f"Query results for '{query_term}':")
        print(result)

        if not result or not result['ids'] or len(result['ids'][0]) == 0:
            print(f'No documents found similar to "{query_term}"')
            return

        print(f'Top 3 similar documents to "{query_term}":')
        # Access the nested arrays in 'results["ids"]' and 'results["distances"]'

        for i in range(min(3, len(result['ids'][0]))):
            doc_id = result['ids'][0][i]
            score = result['distances'][0][i]
            # Retrieve text data from the results
            text = result['documents'][0][i]
            if not text:
                print(f' - ID: {doc_id}, Text: "Text not available", Score: {score:.4f}')
            else:
                print(f' - ID: {doc_id}, Text: "{text}", Score: {score:.4f}')
        
    except Exception as error:
        print(f"Similarity search error: {error}")


if __name__ == "__main__":
    main()
    similarity_search(collection, all_items)

