class LLMNode(WorkflowNode):
    def __init__(self, node_id: str, model_name: str, prompt_template: str):
        super().__init__(node_id, "llm")
        self.model_name = model_name
        self.prompt_template = prompt_template

    async def process(self, context: dict) -> dict:
        prompt = self.prompt_template.format(**context)

        response = await call_llm(self.model_name, prompt)

        return {"output": response, "model_name": self.model_name}


class SummaryNode(WorkflowNode):
    def __init__(self, node_id: str):
        super().__init__(node_id, "summary")

    async def process(self, context: dict) -> dict:
        text = context.get("text", "")
        summary = await summarize_text(text)
        return {"summary": summary}
