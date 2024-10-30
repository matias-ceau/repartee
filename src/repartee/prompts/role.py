import openai

class Role:
    """
    A Role is a class that represents a specific role in a conversation.
    """
    SHORT_PROMPT = "Generate a prompt for a role that involves discussing topics in computer science, programming, philosophy, and psychology."

    LONG_PROMPT = (
        "Generate a detailed prompt for a role that requires in-depth knowledge and the ability to discuss various topics "
        "in computer science, programming, philosophy, and psychology. The role should be able to handle complex questions, "
        "provide insightful answers, and engage in meaningful conversations on these subjects."
    )

    def __init__(self, 
                 name, 
                 description : str = None, 
                 system_prompt : str = None,
                 helper : str = "gpt4o"):
        self._name = name
        self._description = description
        self._system_prompt = system_prompt
        self._helper_model_name = helper  

    @property
    def name(self):
        return self._name
    
    @property
    def description(self):
        return self._description
    
    @description.setter
    def description(self, value):
        self._description = value

    @property   
    def system_prompt(self):
        return self._system_prompt
    
    @system_prompt.setter
    def system_prompt(self, value):
        self._system_prompt

    def generate_prompt(self, model=None):
        model = model if model else self._helper_model_name
        helper = openai.Client(model)
        response = helper.chat.completion.create(
            model=self._helper_model_name,
            messages = [{"role" : "user", 
                         "content" : f"Generate a prompt for the role '{self._name}' with the following description: {self._description}",}]
            max_tokens=100
        )
        self.prompt = response.choices[0].text.strip()
        
    def interactive_prompt(self, model = None)
            ok = False
            while not ok:
                print(f"Generating a prompt for the role '{self._name}' with the following description: {self._description}")
                print("Please enter a prompt:")
                prompt = input()
                print("Is this prompt okay? (yes/no)")
                response = input()
                if response.lower() == "yes":
                    ok = True
