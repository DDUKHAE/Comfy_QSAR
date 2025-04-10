class ShowText:
    @classmethod    
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"force_input": True}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("TEXT",)
    FUNCTION = "test"
    CATEGORY = "QSAR/TEST"
    OUTPUT_NODE = True

    @staticmethod
    def test(text):
        return {"ui": {"text": text},
                "result": (text)}

NODE_CLASS_MAPPINGS = {
    "ShowText": ShowText
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ShowText": "Show Text"
}