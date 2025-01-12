class WorkflowNode:
    def __init__(self, node_id: str, node_type: str):
        self.node_id = node_id
        self.node_type = node_type
        self.inputs = {}  # 输入参数
        self.outputs = {}  # 输出参数
        self.next_nodes = []  # 下一个节点列表

    async def process(self, context: dict) -> dict:
        """
        处理节点逻辑
        :param context: 工作流上下文，包含之前节点的输出
        :return: 处理结果
        """
        raise NotImplementedError
