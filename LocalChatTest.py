import requests
import json

def list_models():
    """Fetch and return installed models from Ollama"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        else:
            print("⚠️ Could not fetch models:", response.text)
            return []
    except Exception as e:
        print("⚠️ Error connecting to Ollama:", e)
        return []

def ollama_chat_stream():
    models = list_models()
    if not models:
        print("❌ No models found. Try running: `ollama pull gemma3:4b`")
        return

    print("📦 Installed models:")
    for i, m in enumerate(models, 1):
        print(f"  {i}. {m}")

    choice = input("\n👉 Select a model by number (default 1): ").strip()
    try:
        idx = int(choice) - 1 if choice else 0
        model = models[idx]
    except (ValueError, IndexError):
        model = models[0]

    print(f"\n🤖 Chatbot running with {model}. Type 'exit' to quit.\n")

    # Store conversation history
    messages = []

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        # Add user message to history
        messages.append({"role": "user", "content": user_input})

        # Stream response from Ollama
        with requests.post(
            "http://localhost:11434/api/chat",
            json={"model": model, "messages": messages, "stream": True},
            stream=True,
        ) as response:
            if response.status_code != 200:
                print("⚠️ Error:", response.text)
                break

            print(f"{model}: ", end="", flush=True)
            assistant_reply = ""

            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    token = data.get("message", {}).get("content", "")
                    print(token, end="", flush=True)
                    assistant_reply += token

                    if data.get("done", False):
                        break
            print("\n")

        # Save assistant’s reply to history
        messages.append({"role": "assistant", "content": assistant_reply})


if __name__ == "__main__":
    ollama_chat_stream()
