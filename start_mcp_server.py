"""MCP Server启动脚本 — v2.9.2新增

启动DeepRAG MCP Server，将项目工具暴露为标准MCP服务。

用法：
    # 直接启动
    python start_mcp_server.py

    # 在Claude Desktop配置中添加：
    {
      "mcpServers": {
        "deeprag": {
          "command": "python",
          "args": ["start_mcp_server.py"]
        }
      }
    }

面试要点：
- MCP Server支持stdio和Streamable HTTP两种传输方式
- 本脚本默认使用stdio模式
"""
import sys
import logging
from pathlib import Path

# 添加项目根目录到path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,  # MCP Server的stdout用于JSON-RPC通信，日志输出到stderr
)
log = logging.getLogger("deeprag")


def main():
    """启动MCP Server"""
    log.info("=" * 50)
    log.info("DeepRAG MCP Server v2.9.2")
    log.info("=" * 50)

    try:
        from src.tools.mcp_server import run_stdio, TOOLS, RESOURCES, PROMPTS

        log.info(f"已注册 {len(TOOLS)} 个Tools: {[t['name'] for t in TOOLS]}")
        log.info(f"已注册 {len(RESOURCES)} 个Resources: {[r['uri'] for r in RESOURCES]}")
        log.info(f"已注册 {len(PROMPTS)} 个Prompts: {[p['name'] for p in PROMPTS]}")
        log.info("启动stdio模式...")
        log.info("=" * 50)

        run_stdio()

    except KeyboardInterrupt:
        log.info("MCP Server已停止")
    except Exception as e:
        log.error(f"MCP Server启动失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
