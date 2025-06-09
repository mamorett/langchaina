import os
import streamlit as st
from typing import Dict, Any, List
from dataclasses import dataclass
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

@dataclass
class AgentState:
    query: str = ""
    context: str = ""
    analysis: str = ""
    response: str = ""
    next_action: str = ""
    iteration: int = 0
    max_iterations: int = 3

class GraphAIAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7,
        )
        self.analyzer = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3,
        )

    def run_with_reasoning(self, chat_history: List[Dict[str, str]], st_container):
        # The latest user message is the query
        query = chat_history[-1]["content"]
        state = AgentState(query=query)

        # Prepare conversation history as LangChain messages (INCLUDING the latest message)
        lc_messages = []
        for msg in chat_history:  # Include ALL messages
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            else:
                lc_messages.append(AIMessage(content=msg["content"]))

        # Router
        st_container.markdown("🔀 **Router:** Analyzing query and providing context...")
        router_out = self._router_node(state, lc_messages)
        state = AgentState(**{**vars(state), **router_out})
        st_container.markdown(f"**Context:**\n{state.context}")

        # Analyzer
        st_container.markdown("🧐 **Analyzer:** Determining if research is needed or direct response is possible...")
        analyzer_out = self._analyzer_node(state, lc_messages)
        state = AgentState(**{**vars(state), **analyzer_out})
        st_container.markdown(f"**Analysis:**\n{state.analysis}")

        # Decide next step
        if analyzer_out["next_action"] == "research":
            st_container.markdown("🔬 **Researcher:** Gathering additional information...")
            researcher_out = self._researcher_node(state, lc_messages)
            state = AgentState(**{**vars(state), **researcher_out})
            st_container.markdown(f"**Research:**\n{state.context.split('Research:',1)[-1].strip()}")

        # Responder (streamed)
        responder_prompt = f"""You are a helpful AI assistant. Provide a comprehensive, accurate,
and well-structured response based on the analysis and context provided.

Conversation history:
{self._format_history(chat_history)}

Current query: {state.query}
Context: {state.context}
Analysis: {state.analysis}

Provide a complete and helpful response that takes into account the conversation history.
"""
        st_container.markdown("💬 **Responder:** Generating answer...")
        messages = [HumanMessage(content=responder_prompt)]
        full_response = ""
        response_placeholder = st_container.empty()
        for chunk in self.llm.stream(messages):
            full_response += chunk.content
            response_placeholder.markdown(full_response + "▌")
        response_placeholder.markdown(full_response)
        state.response = full_response

        # Validator
        st_container.markdown("✅ **Validator:** Checking if the answer is complete...")
        validator_out = self._validator_node(state, lc_messages, full_response)
        state = AgentState(**{**vars(state), **validator_out})
        validation = state.context.split("Validation:",1)[-1].strip()
        st_container.markdown(f"**Validation:**\n{validation}")

        return full_response

    def _format_history(self, chat_history):
        # Format ALL chat history for prompt (including context)
        formatted = ""
        for msg in chat_history:
            if msg["role"] == "user":
                formatted += f"User: {msg['content']}\n"
            else:
                formatted += f"Assistant: {msg['content']}\n"
        return formatted.strip()

    def _router_node(self, state: AgentState, lc_messages: List) -> Dict[str, Any]:
        system_msg = """You are a query router. Analyze the user's query in the context of the conversation history. 
        Provide context about the query type and any relevant information from the conversation history that should be considered."""
        messages = [SystemMessage(content=system_msg)] + lc_messages
        response = self.llm.invoke(messages)
        return {
            "context": response.content,
            "iteration": state.iteration + 1
        }

    def _analyzer_node(self, state: AgentState, lc_messages: List) -> Dict[str, Any]:
        system_msg = """Analyze the query and context, considering the conversation history. 
        Determine if additional research is needed or if you can provide a direct response based on the conversation context."""
        messages = [SystemMessage(content=system_msg)] + lc_messages + [
            HumanMessage(content=f"""
            Context: {state.context}
            Previous Analysis: {state.analysis}
            
            Consider the conversation history when determining the next action.
            """)
        ]
        response = self.analyzer.invoke(messages)
        analysis = response.content
        if "research" in analysis.lower() or "more information" in analysis.lower():
            next_action = "research"
        else:
            next_action = "respond"
        return {
            "analysis": analysis,
            "next_action": next_action
        }

    def _researcher_node(self, state: AgentState, lc_messages: List) -> Dict[str, Any]:
        system_msg = """You are a research assistant. Based on the analysis and conversation history, 
        gather relevant information and insights to help answer the query comprehensively."""
        messages = [SystemMessage(content=system_msg)] + lc_messages + [
            HumanMessage(content=f"""
            Analysis: {state.analysis}
            Research focus: Provide detailed information relevant to the current query while considering the conversation context.
            """)
        ]
        response = self.llm.invoke(messages)
        updated_context = f"{state.context}\n\nResearch: {response.content}"
        return {"context": updated_context}

    def _validator_node(self, state: AgentState, lc_messages: List, response: str) -> Dict[str, Any]:
        system_msg = """Evaluate if the response adequately answers the query in the context of the conversation. 
        Return 'COMPLETE' if satisfactory, or 'NEEDS_IMPROVEMENT' if more work is needed."""
        messages = [SystemMessage(content=system_msg)] + lc_messages + [
            HumanMessage(content=f"""
            Response: {response}

            Is this response complete and satisfactory given the conversation context?
            """)
        ]
        response = self.analyzer.invoke(messages)
        validation = response.content
        return {"context": f"{state.context}\n\nValidation: {validation}"}

# --- Streamlit UI ---

st.set_page_config(page_title="Graph AI Agent", page_icon="🤖")
st.title("🤖 Graph AI Agent (LangGraph + Gemini)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

agent = GraphAIAgent()

# Display existing chat history FIRST
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

# Handle new user input
user_input = st.chat_input("Type your prompt and press Enter...")

if user_input:
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Agent reasoning and streaming output
    with st.chat_message("assistant"):
        response = agent.run_with_reasoning(st.session_state.chat_history, st)
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
