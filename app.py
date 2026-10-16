import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.llm import LLMChain
from langchain_classic.chains.llm_math.base import LLMMathChain
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.agents import AgentType, initialize_agent, Tool
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_openai import ChatOpenAI

st.set_page_config(page_title="Math Problem Solver", page_icon="🧮")
st.title("🧮 Text to Math Problem Solver and Data Search Assistant")

openai_api_key = st.sidebar.text_input("OpenAI API KEY", type="password")

if not openai_api_key:
    st.info("Please enter your OpenAI API Key in the sidebar.")
    st.stop()

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=openai_api_key)

wikipedia_wrapper = WikipediaAPIWrapper()

wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="Search Wikipedia for factual information about different topics."
)

math_chain = LLMMathChain.from_llm(llm=llm)

calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="Solve mathematical calculations. Input a mathematical expression."
)

prompt_template = PromptTemplate(
    input_variables=["question"],
    template="""
    You are an assistant that solves mathematical and reasoning questions.
    Explain your answer step by step.

    Question: {question}

    Answer:
    """
)

chain = LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool = Tool(
    name="Reasoning Tool",
    func=chain.run,
    description="Answer logic-based and reasoning questions with explanations."
)

assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! Ask me a maths, reasoning, or general knowledge question."
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Ask your question here...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    try:
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(
                st.container(),
                expand_new_thoughts=False
            )

            with st.spinner("Generating answer..."):
                response = assistant_agent.invoke(
                    {"input": question},
                    config={"callbacks": [st_cb]}
                )

                answer = response["output"]

            st.write(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

    except Exception as e:
        st.error(f"Error generating answer: {e}")