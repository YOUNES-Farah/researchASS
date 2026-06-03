import asyncio
import time
from graph.graph import graph

async def main():
    print("\n" + "="*50)
    print("  ResearchASS — 100% Free AI Research Agent")
    print("="*50)
    print("Tape 'exit' pour quitter\n")

    session_id = "session-1"

    while True:
        question = input("❓ Ta question : ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            print("À bientôt !")
            break

        if not question:
            continue

        print("\n⏳ Recherche en cours...\n")

        try:
            # ✅ START TIMER
            start = time.time()

            result = await graph.ainvoke(
                {"question": question},
                config={"configurable": {"thread_id": session_id}}
            )

            # ✅ END TIMER
            elapsed = time.time() - start

            print("\n" + "─"*50)
            print("💡 RÉPONSE :\n")
            print(result["generation"])

            print("\n" + "─"*50)
            print(f"Source : {result['source']}")

            # ✅ affichage temps
            print(f"\n⏱️ Temps: {elapsed:.2f}s\n")

        except Exception as e:
            print(f"❌ Erreur : {e}\n")


if __name__ == "__main__":
    asyncio.run(main())