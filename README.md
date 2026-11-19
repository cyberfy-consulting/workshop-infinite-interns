# Infinite Interns - Get AI Agents to Do Your Busy Work

This is the repository for the Infinite Interns workshop, held by Cyberfy Consulting. 

![Slide1](./images/slides/Slide1.png)

In this workshop, the participants will learn how to:

* Make direct API calls to Azure AI Foundry (or any similar AI as a Service API)
* Build custom agents with LangGraph
* Integrate local and online tools via the Model Context Protocol (MCP)
* Orchestrate multi-agent workflows to automate complex tasks
* Implement Human-in-the-loop (HIL) systems to review and approve tool calls
* Handle security issues such as malicious implementations and prompt injection attacks

## Prerequisites

- Python 3.14 (the version Binder and the devcontainer use)
- pip
- A model deployed on Azure AI Foundry

## Setup

#### To run the workshop locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-org>/workshop-infinite-interns.git
   cd workshop-infinite-interns
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure your credentials when prompted in the notebook.

#### Run the workshop online with mybinder.org

Binder allows you to create custom computing environments that can be shared and used by many remote users. One such deployment lives at mybinder.org, and is free to use.

https://mybinder.org/v2/gh/cyberfy-consulting/workshop-infinite-interns/main?urlpath=%2Fdoc%2Ftree%2Fworkshop.ipynb


## Project Structure

The workshop is contained in a Jupyter notebook and includes several utility scripts:

- `workshop.ipynb`: The Jupyter Notebook used during the workshop.
- `utils/*`: Technical stuff that helps reduce the complexity within the workshop material.
- `images/`: Directory containing images and slides used in the notebook.