# 🧠 Maran: Autonomous AI Agent for Code Generation and Execution

Maran is an advanced autonomous AI agent designed to interpret natural language commands, generate corresponding Python code, execute it, and learn from the outcomes to improve over time. Inspired by the concept of a personal assistant, Maran aims to automate tasks by understanding user instructions and performing actions accordingly.

## 🚀 Features

* **Natural Language Understanding**: Interpret user commands written in plain English.
* **Dynamic Code Generation**: Generate Python scripts based on user instructions.
* **Autonomous Execution**: Execute generated code and handle outputs or errors.
* **Self-Improvement**: Learn from past executions to enhance future performance.
* **Modular Architecture**: Organized codebase with clear separation of concerns.
* **API Integration**: Communicate with external services through APIs.
* **Monitoring and Logging**: Track performance, errors, and system metrics.
* **Rate Limiting**: Control the frequency of operations to manage resources effectively.

## 🧩 Project Structure

```
Maran/
├── apiManagement/           # Handles API interactions
├── architectures/           # Contains model architectures
├── autonomous/              # Modules for autonomous decision-making
├── brain/                   # Core logic processing modules
├── configs/                 # Configuration files for various components
├── tokenizers/              # Tools for text tokenization
├── utils/                   # Utility functions and helpers
├── deepmodel.py             # Core model for interpreting and generating code
├── deepMain.py              # Main entry point of the application
├── deeepmain2.py            # Alternative entry point or testing script
├── reasoning.py             # Handles logical reasoning
├── self_improvement.py      # Implements self-learning mechanisms
├── reward_model.py          # Evaluates execution outcomes
├── monitoring.py            # Tracks system performance and logs
├── rate_limiter.py          # Manages operation frequency
├── model_versioning.py      # Handles model version control
├── singleFile.py            # Standalone demonstration script
├── singlefile2.py           # Another standalone demonstration script
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## ⚙️ Installation

### Prerequisites

* Python 3.8 or higher
* Git

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Gokul0616/Maran.git
   cd Maran
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## 🧪 Usage

Run the main application:

```bash
python deepMain.py
```

Follow the on-screen prompts to enter natural language commands. The AI agent will process your input, generate the corresponding Python code, execute it, and provide the results.

## 🛠️ Modules Overview

* **`deepmodel.py`**: Core module that interprets user commands and generates Python code.
* **`deepMain.py` / `deeepmain2.py`**: Entry points that handle user interaction and manage the execution flow.
* **`reasoning.py`**: Provides logical reasoning capabilities to understand complex instructions.
* **`self_improvement.py`**: Implements learning mechanisms to improve performance over time.
* **`reward_model.py`**: Evaluates the success of executed code to guide learning.
* **`monitoring.py`**: Tracks system performance, logs errors, and ensures reliability.
* **`rate_limiter.py`**: Controls the frequency of operations to manage resources.
* **`model_versioning.py`**: Manages different versions of the AI model for consistency and rollback capabilities.

## 📡 API Management

The `apiManagement/` directory contains modules that handle interactions with external APIs, enabling the AI agent to fetch data or perform actions beyond its local environment.

## 🧠 Architectures and Configurations

* **`architectures/`**: Contains definitions of various AI model architectures that can be used or experimented with.
* **`configs/`**: Holds configuration files that define parameters and settings for different components of the system.

## 🧰 Utilities

* **`tokenizers/`**: Includes tools for breaking down text input into tokens, essential for natural language processing.
* **`utils/`**: Provides utility functions and helper methods used across various parts of the project.

## 📄 Additional Scripts

* **`singleFile.py` & `singlefile2.py`**: Standalone scripts for testing or demonstrating the AI agent's capabilities.

## 📈 Monitoring and Logging

The `monitoring.py` module tracks the system's performance, logs errors, and ensures the AI operates reliably. Logs are stored in the `logs/` directory for review and analysis.

## 🔄 Self-Improvement Mechanism

The AI agent learns from its past actions using the `self_improvement.py` module. It evaluates the outcomes of executed code through the `reward_model.py` and adjusts its future behavior to enhance performance.

## 📖 License

This project is licensed under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## 📬 Contact

For questions or suggestions, please open an issue or contact [Gokul0616](https://github.com/Gokul0616).

