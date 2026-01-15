"""
Python Fundamentals - Basic Examples
This file contains examples of core Python concepts
"""

# ============================================================================
# 1. VARIABLES AND DATA TYPES
# ============================================================================

# Numbers
integer_var = 42
float_var = 3.14
complex_var = 3 + 4j

# Strings
string_var = "Hello, World!"
multiline_string = """This is a
multiline string"""

# Boolean
bool_var = True
bool_var2 = False

# None type
none_var = None

print("=== Variables and Data Types ===")
print(f"Integer: {integer_var}, Type: {type(integer_var)}")
print(f"Float: {float_var}, Type: {type(float_var)}")
print(f"String: {string_var}, Type: {type(string_var)}")
print(f"Boolean: {bool_var}, Type: {type(bool_var)}")
print()


# ============================================================================
# 2. OPERATORS
# ============================================================================

print("=== Operators ===")

# Arithmetic operators
a, b = 10, 3
print(f"Addition: {a} + {b} = {a + b}")
print(f"Subtraction: {a} - {b} = {a - b}")
print(f"Multiplication: {a} * {b} = {a * b}")
print(f"Division: {a} / {b} = {a / b}")
print(f"Floor Division: {a} // {b} = {a // b}")
print(f"Modulus: {a} % {b} = {a % b}")
print(f"Exponentiation: {a} ** {b} = {a ** b}")

# Comparison operators
print(f"\nComparison: {a} > {b} = {a > b}")
print(f"Comparison: {a} == {b} = {a == b}")
print(f"Comparison: {a} != {b} = {a != b}")

# Logical operators
x, y = True, False
print(f"\nLogical AND: {x} and {y} = {x and y}")
print(f"Logical OR: {x} or {y} = {x or y}")
print(f"Logical NOT: not {x} = {not x}")
print()


# ============================================================================
# 3. CONTROL FLOW - IF/ELSE STATEMENTS
# ============================================================================

print("=== Control Flow - If/Else ===")

age = 20
if age < 18:
    print("Minor")
elif age < 65:
    print("Adult")
else:
    print("Senior")

# Ternary operator
status = "Adult" if age >= 18 else "Minor"
print(f"Status: {status}")
print()


# ============================================================================
# 4. LOOPS
# ============================================================================

print("=== Loops ===")

# For loop with range
print("For loop (range):")
for i in range(5):
    print(f"  Iteration {i}")

# For loop with list
print("\nFor loop (list):")
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(f"  {fruit}")

# For loop with enumerate
print("\nFor loop (enumerate):")
for index, fruit in enumerate(fruits):
    print(f"  {index}: {fruit}")

# While loop
print("\nWhile loop:")
count = 0
while count < 3:
    print(f"  Count: {count}")
    count += 1

# Loop control: break and continue
print("\nLoop control (break/continue):")
for i in range(10):
    if i == 2:
        continue  # Skip this iteration
    if i == 7:
        break  # Exit loop
    print(f"  {i}")
print()


# ============================================================================
# 5. DATA STRUCTURES - LISTS
# ============================================================================

print("=== Lists ===")

# Creating lists
my_list = [1, 2, 3, 4, 5]
mixed_list = [1, "hello", 3.14, True]

# List operations
print(f"Original list: {my_list}")
print(f"First element: {my_list[0]}")
print(f"Last element: {my_list[-1]}")
print(f"Slicing [1:3]: {my_list[1:3]}")
print(f"Length: {len(my_list)}")

# List methods
my_list.append(6)
print(f"After append(6): {my_list}")

my_list.insert(0, 0)
print(f"After insert(0, 0): {my_list}")

my_list.remove(3)
print(f"After remove(3): {my_list}")

popped = my_list.pop()
print(f"After pop(): {my_list}, popped: {popped}")

# List comprehension
squares = [x**2 for x in range(5)]
print(f"List comprehension (squares): {squares}")

even_squares = [x**2 for x in range(10) if x % 2 == 0]
print(f"List comprehension (even squares): {even_squares}")
print()


# ============================================================================
# 6. DATA STRUCTURES - DICTIONARIES
# ============================================================================

print("=== Dictionaries ===")

# Creating dictionaries
my_dict = {
    "name": "John",
    "age": 30,
    "city": "New York"
}

# Accessing values
print(f"Dictionary: {my_dict}")
print(f"Name: {my_dict['name']}")
print(f"Age: {my_dict.get('age', 'N/A')}")

# Adding/updating values
my_dict["email"] = "john@example.com"
my_dict["age"] = 31
print(f"After updates: {my_dict}")

# Dictionary methods
print(f"Keys: {list(my_dict.keys())}")
print(f"Values: {list(my_dict.values())}")
print(f"Items: {list(my_dict.items())}")

# Dictionary comprehension
squared_dict = {x: x**2 for x in range(5)}
print(f"Dictionary comprehension: {squared_dict}")
print()


# ============================================================================
# 7. DATA STRUCTURES - TUPLES
# ============================================================================

print("=== Tuples ===")

# Creating tuples (immutable)
my_tuple = (1, 2, 3, "hello")
print(f"Tuple: {my_tuple}")
print(f"First element: {my_tuple[0]}")

# Tuple unpacking
a, b, c, d = my_tuple
print(f"Unpacked: a={a}, b={b}, c={c}, d={d}")

# Multiple return values (using tuples)
def get_name_age():
    return "Alice", 25

name, age = get_name_age()
print(f"Function returning tuple: name={name}, age={age}")
print()


# ============================================================================
# 8. DATA STRUCTURES - SETS
# ============================================================================

print("=== Sets ===")

# Creating sets (unique elements, unordered)
my_set = {1, 2, 3, 3, 4, 4, 5}
print(f"Set (duplicates removed): {my_set}")

# Set operations
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

print(f"Set1: {set1}")
print(f"Set2: {set2}")
print(f"Union: {set1 | set2}")
print(f"Intersection: {set1 & set2}")
print(f"Difference: {set1 - set2}")
print(f"Symmetric difference: {set1 ^ set2}")

# Set methods
set1.add(7)
print(f"After add(7): {set1}")
set1.remove(1)
print(f"After remove(1): {set1}")
print()


# ============================================================================
# 9. STRING OPERATIONS
# ============================================================================

print("=== String Operations ===")

text = "  Hello, World!  "
print(f"Original: '{text}'")
print(f"Lowercase: {text.lower()}")
print(f"Uppercase: {text.upper()}")
print(f"Strip whitespace: '{text.strip()}'")
print(f"Replace: {text.replace('World', 'Python')}")
print(f"Split: {text.split(',')}")
print(f"Join: {', '.join(['apple', 'banana', 'cherry'])}")

# String formatting
name = "Alice"
age = 30
print(f"f-string: My name is {name} and I'm {age} years old")
print("format(): My name is {} and I'm {} years old".format(name, age))
print("Old style: My name is %s and I'm %d years old" % (name, age))
print()


# ============================================================================
# 10. FUNCTIONS
# ============================================================================

print("=== Functions ===")

# Basic function
def greet(name):
    return f"Hello, {name}!"

print(greet("Bob"))

# Function with default arguments
def greet_with_title(name, title="Mr."):
    return f"Hello, {title} {name}!"

print(greet_with_title("Smith"))
print(greet_with_title("Smith", "Dr."))

# Function with *args (variable positional arguments)
def sum_numbers(*args):
    return sum(args)

print(f"Sum: {sum_numbers(1, 2, 3, 4, 5)}")

# Function with **kwargs (variable keyword arguments)
def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"  {key}: {value}")

print("Info:")
print_info(name="Charlie", age=28, city="Boston")

# Lambda functions (anonymous functions)
square = lambda x: x**2
print(f"Lambda square(5): {square(5)}")

# Using lambda with map and filter
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(f"Map (squared): {squared}")
print(f"Filter (evens): {evens}")
print()


# ============================================================================
# 11. ERROR HANDLING
# ============================================================================

print("=== Error Handling ===")

# Try-except block
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Error: Division by zero!")

# Multiple exceptions
try:
    value = int("not_a_number")
except ValueError as e:
    print(f"ValueError: {e}")
except Exception as e:
    print(f"General error: {e}")

# Try-except-else-finally
try:
    result = 10 / 2
except ZeroDivisionError:
    print("Division by zero!")
else:
    print(f"Success! Result: {result}")
finally:
    print("This always executes")

# Raising exceptions
def check_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative")
    return f"Age is {age}"

try:
    print(check_age(-5))
except ValueError as e:
    print(f"Caught error: {e}")
print()


# ============================================================================
# 12. CLASSES AND OBJECTS
# ============================================================================

print("=== Classes and Objects ===")

class Person:
    # Class variable
    species = "Homo sapiens"
    
    # Constructor
    def __init__(self, name, age):
        self.name = name  # Instance variable
        self.age = age
    
    # Instance method
    def introduce(self):
        return f"Hi, I'm {self.name} and I'm {self.age} years old"
    
    # Class method
    @classmethod
    def from_birth_year(cls, name, birth_year):
        age = 2024 - birth_year
        return cls(name, age)
    
    # Static method
    @staticmethod
    def is_adult(age):
        return age >= 18

# Creating objects
person1 = Person("David", 25)
print(person1.introduce())
print(f"Is adult: {Person.is_adult(person1.age)}")

person2 = Person.from_birth_year("Emma", 1995)
print(person2.introduce())

# Inheritance
class Student(Person):
    def __init__(self, name, age, student_id):
        super().__init__(name, age)
        self.student_id = student_id
    
    def introduce(self):
        return f"{super().introduce()} and my student ID is {self.student_id}"

student = Student("Frank", 20, "S12345")
print(student.introduce())
print()


# ============================================================================
# 13. FILE OPERATIONS
# ============================================================================

print("=== File Operations ===")

# Writing to a file
with open("example.txt", "w") as f:
    f.write("Hello, World!\n")
    f.write("This is a test file.\n")

# Reading from a file
with open("example.txt", "r") as f:
    content = f.read()
    print("File content:")
    print(content)

# Reading line by line
with open("example.txt", "r") as f:
    print("Lines:")
    for line in f:
        print(f"  {line.strip()}")

# Appending to a file
with open("example.txt", "a") as f:
    f.write("This line was appended.\n")
print()


# ============================================================================
# 14. LIST COMPREHENSIONS AND GENERATORS
# ============================================================================

print("=== List Comprehensions and Generators ===")

# List comprehension
numbers = [1, 2, 3, 4, 5]
squared = [x**2 for x in numbers]
print(f"Squared: {squared}")

# Nested list comprehension
matrix = [[i*j for j in range(1, 4)] for i in range(1, 4)]
print(f"Matrix: {matrix}")

# Generator expression (memory efficient)
squares_gen = (x**2 for x in range(5))
print(f"Generator: {list(squares_gen)}")

# Generator function
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

print(f"Fibonacci (first 10): {list(fibonacci(10))}")
print()


# ============================================================================
# 15. MODULES AND IMPORTS
# ============================================================================

print("=== Modules and Imports ===")

# Importing standard library modules
import math
import random
from datetime import datetime

print(f"Pi: {math.pi}")
print(f"Square root of 16: {math.sqrt(16)}")
print(f"Random number: {random.randint(1, 100)}")
print(f"Current time: {datetime.now()}")
print()


print("=== All Examples Complete ===")

