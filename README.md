# Self Improving AI Agent

A self-improving agent that uses the DigitalOcean Gradient SDK to iteratively refine its outputs. It employs a dual-LLM architecture: a **Task LLM** generates content, and a **Meta LLM** critiques and rewrites the prompt to improve quality based on a specific rubric.

## Features

*   **Task LLM**: Generates initial content (e.g., market research briefs).
*   **Quality Rubric**: Automatically checks outputs for required sections (e.g., Trends, Competitor Analysis).
*   **Meta LLM**: Rewrites prompts when quality checks fail, guiding the Task LLM to produce better results.
*   **Self-Correction Loop**: Iterates up to a configured maximum to achieve the desired quality.

## Prerequisites

*   Python 3.8+
*   A DigitalOcean account with access to Gradient.
*   `gradient` Python package.

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd "Self Improving AI Agent"
    ```

2.  Install dependencies:
    ```bash
    pip install gradient python-dotenv
    ```

## Configuration

1.  Create a `.env` file in the project root:
    ```bash
    touch .env
    ```

2.  Add your DigitalOcean Gradient credentials and model configurations to `.env`:
    ```env
    GRADIENT_MODEL_ACCESS_KEY="your_access_key_here"
    # Optional:
    # MODEL_TASK="llama3.3-70b-instruct"
    # MODEL_META="llama3.3-70b-instruct"
    ```

## Usage

Run the main script to see the agent in action:

```bash
python main.py
```

The script will demonstrate a market research task, showing the initial prompt, the Task LLM's response, quality checks, and any necessary prompt rewrites by the Meta LLM until the quality criteria are met or the maximum iterations are reached.

## How it Works

1.  **Initial Prompt**: The agent starts with a user-defined prompt.
2.  **Generation**: The Task LLM generates a response.
3.  **Evaluation**: The response is checked against a predefined quality rubric (e.g., must contain "Top 3 trends").
4.  **Refinement**: If checks fail, the Meta LLM analyzes the failure and rewrites the prompt to explicitly ask for the missing information.
5.  **Loop**: This process repeats until the response passes all checks or the iteration limit is hit.
