from pydantic import BaseModel
from pydantic import EmailStr
from pydantic import Field
from typing import Optional

class Person(BaseModel):
    name: str = "aftab"
    age: int
    phone: Optional[str] = 8237574142
    email: EmailStr
    cgpa: float = Field(gt = 0, le = 10, default = 8.0, description = "A decimal value representing the CGPA of a student")

new_person = {
    "age": "20", # Here age is provided as a string
    "email": "aftab@gmail.com",
    
    } 

# Pydantic performs type coercion here.
# It automatically converts "20" (str) into 20 (int)
# because the Person model specifies that age should be an int.
person = Person(**new_person)
print(person)
print(type(person))

person_dict = dict(person)
print(person_dict["name"])

person_json = person.model_dump_json()
print(person_json)