from pydantic import BaseModel
import instructor
from openai import OpenAI
import cv2
import numpy as np
import base64
from PIL import Image
from io import BytesIO



class LLMProxy:
    """
    A example node

    Class methods
    -------------
    INPUT_TYPES (dict): 
        Tell the main program input parameters of nodes.
    IS_CHANGED:
        optional method to control when the node is re executed.

    Attributes
    ----------
    RETURN_TYPES (`tuple`): 
        The type of each element in the output tuple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tuple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Second value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True, #True if you want the field to look like the one on the ClipTextEncode node
                    "default": "Description of the image",
                }),
                # "float_field": ("FLOAT", {
                #     "default": 1.0,
                #     "min": 0.0,
                #     "max": 10.0,
                #     "step": 0.01,
                #     "round": 0.001, #The value representing the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                #     "display": "number"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "system_prompt": ("STRING", {
                    "multiline": True, #True if you want the field to look like the one on the ClipTextEncode node
                    "default": "You are a helpful AI assistant",
                }),
                "temperature": ("FLOAT", {
                    "default": 0, 
                    "min": 0.0, #Minimum value
                    "max": 1.0, #Maximum value
                    "step": 0.01, #Slider's step
                    "display": "number" # Cosmetic only: display as "number" or "slider"
                }),
                "max_tokens": ("FLOAT", {
                    "default": 500, 
                    "min": 5, #Minimum value
                    "max": 10000, #Maximum value
                    "step": 10, #Slider's step
                    "display": "number" # Cosmetic only: display as "number" or "slider"
                }),
                "print_to_screen": (["enable", "disable"],),
                "model": ("STRING", {
                    "multiline": False, #True if you want the field to look like the one on the ClipTextEncode node
                    "default": "gpt-3.5-turbo",
                }),
                "api_key": ("STRING", {
                    "multiline": False, #True if you want the field to look like the one on the ClipTextEncode node
                    "default": "Your API key",
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "chat"

    #OUTPUT_NODE = False

    CATEGORY = "Example"

    def chat(self, prompt, image, system_prompt, temperature, max_tokens, print_to_screen, model, api_key):
        #do some processing on the image, in this example I just invert it
        # Define your desired output structure
        class ImageDescription(BaseModel):
            description: str

        # Patch the OpenAI client
        client = instructor.from_openai(OpenAI(
            api_key=api_key,
        ))

        for (batch_number, image) in enumerate(image):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            print(f"Image: {type(img)}")
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # Extract structured data from natural language
            msg = {
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            'url': f'data:image/jpeg;base64,{img_base64_str}'
                        }
                    }
                ]
            }
            print(f"Prompt Msg: {msg}")
            description = client.chat.completions.create(
                model=model,
                response_model=ImageDescription,
                messages=[ msg ],
                temperature=temperature,
                max_tokens=max_tokens,
            ) 
        return (f'{description.description}',)

    """
        The node will always be re executed if any of the inputs change but
        this method can be used to force the node to execute again even when the inputs don't change.
        You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
        executed, if it is different the node will be executed again.
        This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
        changes between executions the LoadImage node is executed again.
    """
    #@classmethod
    #def IS_CHANGED(s, image, string_field, int_field, float_field, print_to_screen):
    #    return ""

# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# WEB_DIRECTORY = "./somejs"



