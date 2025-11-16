from typing import Optional, Dict, Any, List, Union


class TextGenerator:
    
    def __init__(
        self,
        model,
        max_length: Optional[int] = None,
        k_neighbors: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs
    ):
        self.model = model
        self.defaults = {
            "max_length": max_length or 128,
            "k_neighbors": k_neighbors or 3,
            "max_new_tokens": max_new_tokens or 20,
        }
        self.defaults.update(kwargs)
    
    def __call__(
        self,
        text: Union[str, List[str]],
        return_dict: bool = False,
        **kwargs
    ):
        if isinstance(text, list):
            return self.generate_batch(text, return_dict=return_dict, **kwargs)
        
        params = {**self.defaults, **kwargs}
        result = self.model.generate(
            text=text,
            enable_structural_edit=False,
            return_dict=True,
            **params
        )
        return result if return_dict else result["generated_text"]
    
    def generate_batch(
        self,
        texts: List[str],
        return_dict: bool = False,
        **kwargs
    ):
        return [self(text, return_dict=return_dict, **kwargs) for text in texts]


class TextEditor:
    
    def __init__(
        self,
        model,
        max_length: Optional[int] = None,
        k_neighbors: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        enable_structural_edit: bool = True,
        **kwargs
    ):
        self.model = model
        self.defaults = {
            "max_length": max_length or 128,
            "k_neighbors": k_neighbors or 3,
            "max_new_tokens": max_new_tokens or 20,
            "enable_structural_edit": enable_structural_edit,
        }
        self.defaults.update(kwargs)
    
    def __call__(
        self,
        text: Union[str, List[str]],
        return_topics: bool = True,
        **kwargs
    ):
        if isinstance(text, list):
            return self.edit_batch(text, return_topics=return_topics, **kwargs)
        
        params = {**self.defaults, **kwargs}
        result = self.model.generate(
            text=text,
            return_dict=True,
            **params
        )
        
        output = {
            "original": result["original_text"],
            "edited": result["generated_text"],
        }
        
        if return_topics:
            output["topics"] = result["topics"]
        
        return output
    
    def edit_batch(
        self,
        texts: List[str],
        **kwargs
    ):
        return [self(text, **kwargs) for text in texts]

