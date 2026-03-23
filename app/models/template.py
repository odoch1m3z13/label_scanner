from pydantic import BaseModel
from typing import Optional, Literal


class TemplateRegion(BaseModel):
    name: str
    type: Literal["text", "logo", "color", "barcode"]

    x: int
    y: int
    w: int
    h: int

    expected_text: Optional[str] = None
    strict: bool = False