#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from langchain.tools import tool

# Import your new message type alongside the standard ones
from llm_pack_interface.msg import TrajContext 
from llm_pack_interface.srv import String
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float32

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

import csv
import os
import time
#from dataLogger import export_string

_LOG_DIR = os.path.expanduser('~/comparison_logs')
os.makedirs(_LOG_DIR, exist_ok=True)
_LLM_LOG_PATH = os.path.join(_LOG_DIR, f'llm_prompts_{time.strftime("%Y%m%d")}.csv')

# ===================== LangChain Tool =====================
def make_publish_context_tool(node):
    @tool
    def publish_context(goal_x: float, goal_y: float, goal_z: float, v_const: float, a: float) -> str:
        """
        Publishes high-level trapezoidal trajectory parameters to the trajectory generation node.
        Use this tool to specify the TRAPEZOIDAL trajectory parameters, in order to command the robot.
        Assume the robot starts at (0,0,0) with 0 velocity, and points towards the goal.
        The robot front is positive x direction, left is positive y direction, and up is positive z direction.

        Args:
            goal_x: The X coordinate of the goal position (meters).
            goal_y: The Y coordinate of the goal position (meters).
            goal_z: The Z coordinate of the goal position (meters).
            v_const: The constant phase velocity of the trajectory (m/s).
            a: The acceleration/deceleration rate in the acceleration phase (m/s^2).

        The robot has the size of about 0.40m x 0.50m.
        The robot has the following kinematic constraints:
            Min velocity: 0.1 m/s
            Max velocity: 0.2 m/s
            Min acceleration: 0.02 m/s^2
            Max acceleration: 0.04 m/s^2
            Min distance to goal: 0.0 m
            Max distance to goal (per one trajectory plan): 5.0 m

        Returns: success or error message
        """
        try:
            msg = TrajContext()

            # 1. Set the goal vector
            msg.s_goal = Vector3(x=float(goal_x), y=float(goal_y), z=float(goal_z))

            # 2. Set the kinematics
            msg.v_const = float(v_const)
            msg.a = float(a)

            # 3. Set defaults for FM to overwrite/use
            msg.q_init = Vector3(x=0.0, y=0.0, z=0.0)
            msg.part = msg.CONSTANT

            # Publish the message to the topic
            node.context_publisher.publish(msg)

            return (f"Success: Published trajectory context -> "
                    f"Goal:({goal_x},{goal_y},{goal_z}), Vel:{v_const}, Accel:{a}. "
                    f"Task complete — do NOT call this tool again.")

        except Exception as e:
            return f"Error publishing context topic: {str(e)}"
            
    return publish_context


class LlmNode(Node):
    def __init__(self):
        super().__init__('llm_node')

        self.systemInstructions = [
            {"role": "system", "content": "You are the high-level planning brain of a mobile robot. Your ONLY job is to call the publish_context tool exactly once per user prompt."},
            {"role": "system", "content": "The robot starts at position (0,0,0) with zero velocity. Front is +x, left is +y. Interpret the user prompt and determine: goal_x, goal_y, goal_z (body-frame goal in meters), v_const (m/s), and a (m/s^2)."},
            {"role": "system", "content": "Always estimate parameters even for ambiguous or informal prompts (e.g. 'book-length'=0.3m, 'floor-tile'=0.3m, 'man step speed'=1.4m/s, 'dog speed'=2.0m/s, '7 o\'clock direction'=x=-0.5 y=-0.866, '2 o\'clock direction'=x=0.5 y=-0.866). Never refuse."},
            {"role": "system", "content": "Kinematic constraints: v_const in [0.1, 0.2] m/s, a in [0.02, 0.04] m/s^2, goal distance in [0.0, 5.0] m. Clamp values to these ranges."},
            {"role": "system", "content": "IMPORTANT: Call publish_context EXACTLY ONCE. After the tool returns success, output your final answer immediately — do NOT call the tool again."},
        ]

        # Create ROS2 Interfaces =======================================
        self.create_service(String, 'LlmPrompt', self.prompt_callback)
        
        # CHANGED: We now use a Publisher instead of a Service Client
        self.context_publisher = self.create_publisher(TrajContext, 'traj_context', 10)

        # Initialize LLM
        '''
        api_key = os.environ.get("OPENAI_API_KEY", None)
        if api_key is None:
            self.get_logger().warning("OPENAI_API_KEY not set!")
            
        self.llm = ChatOpenAI(model="gpt-5-nano", api_key=api_key)
        '''

        # initialize Google Gemini LLM instead
        # Initialize LLM
        api_key = os.environ.get("GOOGLE_API_KEY", None)
        if api_key is None:
            self.get_logger().warning("GOOGLE_API_KEY not set!")
            
        # Using Gemini Flash for fast, tool-calling compatible inference
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.0
        )


        # Register the new tool
        self.publish_context_tool = make_publish_context_tool(self)
        tools = [self.publish_context_tool]

        # Create Agent
        self.agent = create_react_agent(self.llm, tools)
        self.get_logger().info("LlmNode initialized. Listening on /LlmPrompt, publishing to /traj_context.")

    def prompt_callback(self, req, res):
        self.log_info(f"Received message: {req.prompt}")
        self.start_time = self.get_clock().now()
        res = String.Response()
        user_text = req.prompt.strip()

        # Fresh history every call — no contamination from previous runs.
        # Each prompt is independent; the system instructions are always prepended.
        messages = self.systemInstructions + [{"role": "user", "content": user_text}]

        try:
            # recursion_limit=4: think → call tool → observe → final answer.
            # This prevents the agent from calling publish_context more than once.
            result = self.agent.invoke(
                {"messages": messages},
                config={"recursion_limit": 4},
            )
            assistant_text = self._extract_user_facing_reply(result)

            self.log_info("\n================================================\n")
            self.log_info(f"Agent reply: {assistant_text}")

            thinking_ms = (self.get_clock().now() - self.start_time).nanoseconds / 1e6
            self.log_info(f"Thinking time: {thinking_ms:.1f} ms")
            self._log_llm_event(user_text, assistant_text, result, thinking_ms)
            res.response = str(assistant_text)

        except Exception as e:
            self.get_logger().error(f"Agent error: {e}")
            res.response = f"Error: {str(e)}"

        return res

    def _extract_user_facing_reply(self, result) -> str:
        """Pull a concise summary from the agent's result instead of dumping
        the entire conversation history. We look for, in priority order:

          1. A successful tool call (publish_context) → summarize the args.
          2. Plain text content in the latest AIMessage (e.g. rejection).
          3. Fall back to str(result).
        """
        if not isinstance(result, dict) or 'messages' not in result:
            return str(result)
        msgs = result['messages']
        if not msgs:
            return '(empty agent response)'

        last_tool_call = None
        last_tool_msg  = None
        last_ai_text   = None

        # Walk from the end backwards so we get the LATEST signals.
        for m in reversed(msgs):
            name = type(m).__name__
            if name in ('AIMessage', 'AIMessageChunk'):
                if last_tool_call is None:
                    tcs = getattr(m, 'tool_calls', None) or []
                    if tcs:
                        last_tool_call = tcs[0]
                if last_ai_text is None:
                    raw = getattr(m, 'content', '') or ''
                    if isinstance(raw, list) and raw:
                        # Gemini multimodal block: [{'type':'text', 'text':...}, ...]
                        first = raw[0]
                        raw = first.get('text', '') if isinstance(first, dict) else str(first)
                    if isinstance(raw, str) and raw.strip():
                        last_ai_text = raw.strip()
            elif name == 'ToolMessage' and last_tool_msg is None:
                last_tool_msg = getattr(m, 'content', '')

        if last_tool_call is not None:
            args = last_tool_call.get('args', {}) if isinstance(last_tool_call, dict) else getattr(last_tool_call, 'args', {})
            parts = []
            gx = args.get('goal_x'); gy = args.get('goal_y'); gz = args.get('goal_z')
            if gx is not None and gy is not None:
                parts.append(f'goal=({gx}, {gy}, {gz})')
            if 'v_const' in args:
                parts.append(f'v_const={args["v_const"]}')
            if 'a' in args:
                parts.append(f'a={args["a"]}')
            summary = 'publish_context(' + ', '.join(parts) + ')'
            if last_tool_msg:
                summary += f'  →  {last_tool_msg}'
            return summary

        if last_ai_text:
            return last_ai_text

        return '(no text and no tool call in agent response)'

    def _log_llm_event(self, prompt: str, response: str, result: dict, thinking_ms: float):
        """Append one row to the daily LLM prompt log CSV.

        Columns:
          timestamp       — wall-clock time of the response
          thinking_ms     — LLM inference latency in milliseconds
          prompt          — raw user text sent to the LLM
          tool_called     — 'publish_context' if tool was invoked, else 'none'
          goal_x_body     — LLM-output goal X in body frame (or nan)
          goal_y_body     — LLM-output goal Y in body frame (or nan)
          goal_z_body     — LLM-output goal Z in body frame (or nan)
          v_const         — commanded constant velocity (or nan)
          a               — commanded acceleration (or nan)
          tool_result     — success/error string from the tool
          llm_response    — full summary string
        """
        # Extract structured tool args from the raw agent result
        tool_called = 'none'
        goal_x = goal_y = goal_z = v_const = a = float('nan')
        tool_result_str = ''
        if isinstance(result, dict) and 'messages' in result:
            for m in reversed(result['messages']):
                name = type(m).__name__
                if name in ('AIMessage', 'AIMessageChunk') and tool_called == 'none':
                    tcs = getattr(m, 'tool_calls', None) or []
                    if tcs:
                        tc = tcs[0]
                        args = tc.get('args', {}) if isinstance(tc, dict) else getattr(tc, 'args', {})
                        tool_called = 'publish_context'
                        goal_x  = float(args.get('goal_x',  float('nan')))
                        goal_y  = float(args.get('goal_y',  float('nan')))
                        goal_z  = float(args.get('goal_z',  float('nan')))
                        v_const = float(args.get('v_const', float('nan')))
                        a       = float(args.get('a',       float('nan')))
                elif name == 'ToolMessage' and not tool_result_str:
                    tool_result_str = getattr(m, 'content', '')

        file_exists = os.path.exists(_LLM_LOG_PATH)
        with open(_LLM_LOG_PATH, 'a', newline='') as f:
            w = csv.writer(f)
            if not file_exists:
                w.writerow(['timestamp', 'thinking_ms',
                            'prompt', 'tool_called',
                            'goal_x_body', 'goal_y_body', 'goal_z_body',
                            'v_const', 'a',
                            'tool_result', 'llm_response'])
            w.writerow([
                time.strftime('%Y-%m-%dT%H:%M:%S'),
                f'{thinking_ms:.1f}',
                prompt,
                tool_called,
                f'{goal_x:.4f}', f'{goal_y:.4f}', f'{goal_z:.4f}',
                f'{v_const:.4f}', f'{a:.4f}',
                tool_result_str,
                response,
            ])
            f.flush()

    def log_info(self, text: str, file_name = "saved_log.txt"):
        #export_string(text, file_name)
        self.get_logger().info(text)

def main(args=None):
    rclpy.init(args=args)
    node = LlmNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()