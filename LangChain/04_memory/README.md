# Memory

Demonstration of how to use memory with LLMs and langChain. Cap max capacity of LLM memory for token / rate control.

# Example Log

USER: 

hi

ASSISTANT: 

Hello! What's your name?

MEMORY: 

{'name': None, 'tasks': []}

USER: 

Johnny

ASSISTANT: 

I understand your message. Your name is Johnny.

MEMORY: 

{'name': 'Johnny', 'tasks': []}

USER: 

What's my name?

ASSISTANT: 

Your name is Johnny.

MEMORY: 

{'name': 'Johnny', 'tasks': []}

USER: 

I need to play soccer today.

ASSISTANT: 

I understand that you need to play soccer today.

MEMORY: 

{'name': 'Johnny', 'tasks': ['play soccer today']}

USER: 

I need to code a project today. 

ASSISTANT: 

I understand you need to code a project today.

MEMORY: 

{'name': 'Johnny', 'tasks': ['play soccer today', 'code a project today']}

USER: 

I need to go shopping today

ASSISTANT: 

I understand you need to go shopping today.

MEMORY: 

{'name': 'Johnny', 'tasks': ['play soccer today', 'code a project today', 'go shopping today']}

USER: 

What do I need to do today?

ASSISTANT: 

You need to play soccer today, code a project today, and go shopping today.

MEMORY: 

{'name': 'Johnny', 'tasks': ['play soccer today', 'code a project today', 'go shopping today']}

USER: 

I need to do laundry today.      

ASSISTANT: 

I understand that you need to do laundry today.

MEMORY: 

{'name': 'Johnny', 'tasks': ['code a project today', 'go shopping today', 'do laundry today']}

USER: 

What do I need to do today?

ASSISTANT: 

You have three tasks to complete today: code a project, go shopping, and do laundry.

MEMORY: 

{'name': 'Johnny', 'tasks': ['code a project today', 'go shopping today', 'do laundry today']}

USER: 

Thanks!

ASSISTANT: 

You're welcome, Johnny! If you need any help with your tasks, let me know.

MEMORY: 

{'name': 'Johnny', 'tasks': ['code a project today', 'go shopping today', 'do laundry today']}