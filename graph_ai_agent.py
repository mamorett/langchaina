
import os
from typing import Dict, Any
from dataclasses import dataclass
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage

@dataclass
class AgentState:
    """State shared across all nodes in the graph"""
    query: str = ""
    context: str = ""
    analysis: str = ""
    response: str = ""
    next_action: str = ""
    iteration: int = 0
    max_iterations: int = 3

class GraphAIAgent:
    def __init__(self, api_key: str = None):
        # Do NOT set os.environ["GOOGLE_API_KEY"] here!
        # The environment variable should be set externally.
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7
        )

        self.analyzer = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3
        )

        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("router", self._router_node)
        workflow.add_node("analyzer", self._analyzer_node)
        workflow.add_node("researcher", self._researcher_node)
        workflow.add_node("responder", self._responder_node)
        workflow.add_node("validator", self._validator_node)

        workflow.set_entry_point("router")
        workflow.add_edge("router", "analyzer")
        workflow.add_conditional_edges(
            "analyzer",
            self._decide_next_step,
            {
                "research": "researcher",
                "respond": "responder"
            }
        )
        workflow.add_edge("researcher", "responder")
        workflow.add_edge("responder", "validator")
        workflow.add_conditional_edges(
            "validator",
            self._should_continue,
            {
                "continue": "analyzer",
                "end": END
            }
        )

        return workflow.compile()

    def _router_node(self, state: AgentState) -> Dict[str, Any]:
        system_msg = """You are a query router. Analyze the user's query and provide context.
        Determine if this is a factual question, creative request, problem-solving task, or analysis."""

        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=f"Query: {state.query}")
        ]

        response = self.llm.invoke(messages)

        return {
            "context": response.content,
            "iteration": state.iteration + 1
        }

    def _analyzer_node(self, state: AgentState) -> Dict[str, Any]:
        system_msg = """Analyze the query and context. Determine if additional research is needed
        or if you can provide a direct response. Be thorough in your analysis."""

        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=f"""
            Query: {state.query}
            Context: {state.context}
            Previous Analysis: {state.analysis}
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

    def _researcher_node(self, state: AgentState) -> Dict[str, Any]:
        system_msg = """You are a research assistant. Based on the analysis, gather relevant
        information and insights to help answer the query comprehensively."""

        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=f"""
            Query: {state.query}
            Analysis: {state.analysis}
            Research focus: Provide detailed information relevant to the query.
            """)
        ]

        response = self.llm.invoke(messages)

        updated_context = f"{state.context}\n\nResearch: {response.content}"

        return {"context": updated_context}

    def _responder_node(self, state: AgentState) -> Dict[str, Any]:
        system_msg = """You are a helpful AI assistant. Provide a comprehensive, accurate,
        and well-structured response based on the analysis and context provided."""

        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=f"""
            Query: {state.query}
            Context: {state.context}
            Analysis: {state.analysis}

            Provide a complete and helpful response.
            """)
        ]

        response = self.llm.invoke(messages)

        return {"response": response.content}

    def _validator_node(self, state: AgentState) -> Dict[str, Any]:
        system_msg = """Evaluate if the response adequately answers the query.
        Return 'COMPLETE' if satisfactory, or 'NEEDS_IMPROVEMENT' if more work is needed."""

        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=f"""
            Original Query: {state.query}
            Response: {state.response}

            Is this response complete and satisfactory?
            """)
        ]

        response = self.analyzer.invoke(messages)
        validation = response.content

        return {"context": f"{state.context}\n\nValidation: {validation}"}

    def _decide_next_step(self, state: AgentState) -> str:
        return state.next_action

    def _should_continue(self, state: AgentState) -> str:
        if state.iteration >= state.max_iterations:
            return "end"
        if "COMPLETE" in state.context:
            return "end"
        if "NEEDS_IMPROVEMENT" in state.context:
            return "continue"
        return "end"

    def run(self, query: str) -> str:
        initial_state = AgentState(query=query)
        result = self.graph.invoke(initial_state)
        return result["response"]

def main():
    agent = GraphAIAgent()

    test_queries = [
        "Explain quantum computing and its applications",
        "What are the best practices for machine learning model deployment?",
        "Create a story about a robot learning to paint"
    ]

    print("🤖 Graph AI Agent with LangGraph and Gemini")
    print("=" * 50)

    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Query {i}: {query}")
        print("-" * 30)

        try:
            response = agent.run(query)
            print(f"🎯 Response: {response}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

        print("\n" + "="*50)

if __name__ == "__main__":
    main()
