from pydantic import Field, BaseModel


class Bug(BaseModel):
    pass


class Soda(Bug):
    """Drank soda"""

    liters: float = Field(
        ..., title="How much soda you drank (in liters)", gt=0
    )
    name: str = Field('diet coke', title="What was it")


class Mood(Bug):
    """Current mood & feel"""

    mood: int = Field(
        ..., title="How do you feel right now? (1=bad ... 5=great)", ge=1, le=5
    )


class Weight(Bug):
    """Weight today"""

    kg: float = Field(..., title="Your weight in kg", ge=0)


class Learned(Bug):
    """Learned stuff"""

    summary: str = Field(
        'nothing', title="Description of a thing you learned lately"
    )
