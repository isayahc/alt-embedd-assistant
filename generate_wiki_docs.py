from llama_index.readers import SimpleWebPageReader #as of version 0.9.13
from llama_index.readers import WikipediaReader


cities = [
    "Los Angeles", "Houston", "Honolulu", "Tucson", "Mexico City", 
    "Cincinatti", "Chicago"
]

wiki_docs = []
for city in cities:
    try:
        doc = WikipediaReader().load_data(pages=[city])
        wiki_docs.extend(doc)
    except Exception as e:
        print(f"Error loading page for city {city}: {e}")
