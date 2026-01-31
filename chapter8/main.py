from operator import itemgetter
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import JsonOutputToolsParser
import dotenv

dotenv.load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

@tool(description="Add two integers")
def add(first_int:int, second_int:int) -> int:
    return first_int + second_int

@tool(description="Multiply two integers")
def multiply(first_int:int, second_int:int)->int:
    return first_int * second_int

@tool(description="Exponentiate the base to the exponent power")
def exponentiate(base:int, exponent:int) -> int:
    return base ** exponent


tools = [multiply, add, exponentiate]
tool_map = {tool.name: tool for tool in tools}

def call_tool(tool_invocation:dict) -> str | Runnable:
    tool = tool_map[tool_invocation["type"]]
    return RunnablePassthrough.assign(output=itemgetter("args") | tool)

llm_with_tools = llm.bind_tools([add, exponentiate, multiply])

chain = (
    llm_with_tools
    | JsonOutputToolsParser()
    | RunnableLambda(call_tool).map()
)

result = chain.invoke(""""
以下の計算を行って
- 100 たす 1000
- 1241 x 21314
- 4 ** 10
""")

print(result)