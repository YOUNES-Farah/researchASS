from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document
import concurrent.futures

# ✅ Singleton
_search_tool = DuckDuckGoSearchRun()

def web_search(query: str) -> list[Document]:
    try:
        # ✅ Timeout agressif : si DuckDuckGo est lent, on coupe à 5s
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_search_tool.invoke, query)
            results = future.result(timeout=5)

        return [Document(
            page_content=results[:1000],   # ✅ encore réduit (1500 → 1000)
            metadata={"source": "duckduckgo"}
        )]

    except Exception as e:
        print(f"   ⚠️ DuckDuckGo erreur/timeout : {e}")
        return [Document(
            page_content=f"Recherche web indisponible. Question : {query}",
            metadata={"source": "fallback"}
        )]