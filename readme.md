# TO RUN THE PROJECT

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