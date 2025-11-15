# Examples:

### Example1 

What is the capital of France?

{'answer': "I don't know the answer.", 'source_text': ''}

CORRECT!!

### Example2

How many hours do cats sleep?

{'answer': '12–16 hours a day', 'source_text': 'They sleep 12–16 hours a day.'}

CORRECT!!

### Example3

How many hours do dogs sleep?

{'answer': '12–16 hours a day', 'source_text': 'They sleep 12–16 hours a day.'}

INCORRECT!!!!!
This is the result of splitting the text into too small of chunks. Right now the chunk is "They sleep 12-16 hours a day", but that's not informative to the model. It doesn't know who "They" is. If we were to make that into a bigger chunk, we would get less precision and more recall.