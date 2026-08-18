class Student:

    @classmethod
    def show(cls):
        print(type(cls))

Student.show()
print(type(Student))