from dotenv import load_dotenv
load_dotenv()
from graph.graph import app

if __name__ == "__main__":
    res = app.invoke(input={'question':'What is agent memory?'})
    print(res)