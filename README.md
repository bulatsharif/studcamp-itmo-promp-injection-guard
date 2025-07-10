# studcamp-itmo-promp-injection-guard

## How does it work?

This project is designed to detect prompt injection, toxicity, and spam in user prompts for LLMs (Large Language Models). It provides a FastAPI backend with endpoints for analyzing user input using both local models and external LLMs via OpenRouter. The backend can:
- Detect prompt injection using regex, ML models, and LLMs.
- Detect toxicity and spam in both English and Russian.
- Provide a unified API for message defense and LLM-based analysis.
- Evaluate and benchmark different LLMs on curated test sets for prompt injection, toxicity, and spam detection.

The project also includes scripts to benchmark various LLMs (via OpenRouter) on detection tasks, measuring both accuracy and response time.

## How to test everythingg

### 1. Clone the repository
```bash
git clone <your-fork-url>
cd studcamp-itmo-promp-injection-guard
```

### 2. Set up the environment
- Install Python 3.10+ (recommended: 3.12)
- (Optional) Create and activate a virtual environment:
  ```bash
  python -m venv venv
  source venv/bin/activate
  ```
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### 3. Configure OpenRouter API key
- Create a `.env` file in the project root with your OpenRouter API key:
  ```
  OPEN_ROUTER_API_KEY=sk-<your-key-here>
  ```

### 4. Run the backend
- From the project root, run:
  ```bash
  uvicorn backend.basic:app --reload
  ```
- The API docs will be available at [http://localhost:8000/docs](http://localhost:8000/docs)

### 5. Test the OpenRouter endpoint
- Use the `/openrouter-analyze` endpoint in the docs or via curl:
  ```bash
  curl -X POST "http://localhost:8000/openrouter-analyze" \
    -H "Content-Type: application/json" \
    -d '{"user_prompt": "Your test prompt here"}'
  ```

### 6. Run LLM evaluation script
- Go to the backend folder:
  ```bash
  cd backend
  ```
- Run the test script:
  ```bash
  python llms_test.py
  ```
- The script will test several LLMs on prompt injection, toxicity, and spam detection, printing accuracy, average confidence, and response time for each model.

### 7. (Optional) Docker
- You can run the backend in Docker using the provided Dockerfile and docker-compose.yml. Make sure to mount your `.env` file and expose the correct ports.

---

**If you have any issues or questions, feel free to open an issue or ask for help!**
