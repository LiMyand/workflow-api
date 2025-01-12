from typing import Dict
from datetime import datetime


class SubWorkflow:
    def __init__(self, config: Dict):
        self.nodes = {}
        self.edges = []
        self.execution_manager = ExecutionManager()

    async def execute(self, context: "IterationContext") -> Dict:
        execution_order = self.execution_manager.get_execution_order()
        results = {}

        for level in execution_order:
            if len(level) > 1:
                # 并行执行同层节点
                level_results = await self._execute_parallel_nodes(level, context)
            else:
                # 串行执行单个节点
                level_results = await self._execute_sequential_node(level[0], context)

            results.update(level_results)
            context.update_level_results(level_results)

        return results


class IterationContext:
    def __init__(self):
        self.current_iteration = 0
        self.results_history = []
        self.current_context = {}
        self.global_context = {}

    def update_level_results(self, results: Dict):
        self.current_context.update(results)

    def add_iteration_result(self, result: Dict):
        self.results_history.append(
            {
                "iteration": self.current_iteration,
                "result": result,
                "timestamp": datetime.now(),
            }
        )
        self.current_iteration += 1


class ExecutionManager:
    def __init__(self):
        self.execution_levels = []

    def get_execution_order(self) -> list:
        return self.execution_levels
