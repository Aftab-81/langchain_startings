class Student:

    @classmethod
    def show(cls):
        print(type(cls))

Student.show()
print(type(Student))
# Here show() is classmethod so that we can access it directly by using class name