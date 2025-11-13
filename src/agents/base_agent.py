"""
Base Agent Module

This module defines the base agent interface and common functionality
for all specialized agents in the multi-agent system.

Author: Umer Farooq, Hussain Waseem Syed, Muhammad Irtaza Khan
Email: umerfarooqcs0891@gmail.com

Purpose:
    - Define common agent interface and abstract methods
    - Provide shared functionality (logging, error handling)
    - Manage agent state and communication
    - Handle tool integration

Input:
    - Agent requests (messages, tasks)
    - Configuration parameters
    - Shared context

Output:
    - Agent responses (results, status)
    - Logs and metrics
    - Error messages (if any)

Dependencies:
    - RAG System: For context retrieval
    - Tools: For agent capabilities
    - Logging: For agent activity tracking

Links to other modules:
    - Base class for: DesignerAgent, OptimizerAgent, ValidatorAgent, EducationalAgent
    - Used by: Orchestrator, WorkflowManager
    - Uses: RAG system, Tools, Logging
"""

# This file will contain:
# - BaseAgent abstract base class
# - Common agent methods (execute, validate, log)
# - Agent communication protocol
# - Error handling and retry logic
# - Tool integration interface

