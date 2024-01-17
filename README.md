## Precision Prompts Utilizing RAG

# About

The need for this project is to revolutionize how businesses interact with LLMs, making the technology more accessible, efficient, and effective by addressing the challenges of prompt engineering. The following 3 components are necessary to achieve this goal.

1. Automatic Prompt Generation Component

   - This component streamlines the process of creating effective prompts, enabling businesses to efficiently utilize LLMs for generating high-quality, relevant content. It significantly reduces the time and expertise required in crafting prompts manually.

2. Automatic Evaluation Data Generation Service

   - This component automates the generation of diverse test cases, ensuring comprehensive coverage and identifying potential issues. This enhances the reliability and performance of LLM applications, saving significant time in the QA(Quality Assurance) process.

3. Prompt Testing and Ranking Service
   - This component evaluates and ranks different prompts based on effectiveness, helping Users to get the desired outcome from LLM. It ensures that chatbots and virtual assistants provide accurate, contextually relevant responses, thereby improving user engagement and satisfaction.

# Setup

STEP1: Create virtual environment

`python3 -m venv precision_rag_prompt`
`source precision_rag_prompt/bin/activate`

STEP2: Install the requirements inside the virtual environment

`pip3 install -r requirements.txt`

STEP3: To Run the project

`uvicorn main:app --reload`

STEP4: Swagger UI available in Below path after the app is started successfully

`http://127.0.0.1:8000/docs`
