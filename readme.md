# ITS (CODELLAMA-7B-INSTRUCT)
Core repository model of Intelligent Tutoring System using models from ollama: codellama-7b-instruct to automate assesment for student's python coding assignments

## TO RUN THE PROJECT
CLONE THE PROJECT
```git clone https://github.com/ITS-Research-dev/its-codellama-7b.git```

INSTALL DEPENDENCIES FIRST
```pip install openai python-dotenv --break-system-packages```

START YOUR AI MODEL SERVER (LM STUDIO or OLLAMA)
AND LOAD THE MODEL; USE "codellama-7b-instruct"
BECAUSE CURRENTLY IN THIS REPOSITORY WE ARE GOING
TO USE THAT MODEL

Update your .env and adjust with the AI model host url that you're using
(e.g if using LM STUDIO, usually its running on port 1234, OLLAMA runs at 11434)

IF ALL DONE ABOVE, TRY CHECKING THE MODEL BY RUNNING "model_check.py"

IF IT RETURNS ERROR, THERE'S SOMETHING WRONG WITH YOUR CONFIGURATION.
Either from models server, missing packages, or others.

-------------------------------------------------------------

# HOW TO RUN THE FINETUNING INPUT:

```pip install flask flask-cors openai python-dotenv ```
(if you dont have yet)

```python api-simple.py```
will run flask on: http://localhost:5050

open index.html on browser
start LM studio service or ollama, adjust the API host