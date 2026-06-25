from pydantic import BaseModel
from models.parameter import Parameter


class Function(BaseModel):
    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: Parameter
