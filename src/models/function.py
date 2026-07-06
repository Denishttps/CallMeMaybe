from pydantic import BaseModel
from .parameter import Parameter


class Function(BaseModel):
    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: Parameter
