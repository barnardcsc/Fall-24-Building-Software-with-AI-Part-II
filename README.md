# Building Software with AI: Part II

This workshop continues where Part I left off, focusing on new ways to work with AI in software development. We're going to focus on two main themes:

- **Structured Outputs**: How to get AI to give us data in specific formats we can use more easily.
- **Dynamic AI Responses**: Using AI that can adapt and respond to our needs in real-time.

**Main Ideas**

1. Using AI to turn messy data into organized data. This opens up a lot of new possibilities.
2. Taking advantage of AI's ability to respond on the fly, which can make building and using software much smoother.

**What We'll Use This For**

- Better ways to handle and analyze data
- Creating software that adapts to users more easily
- Making the process of writing code faster and easier

**Useful Links**

- [Google Colab Notebook](https://platform.openai.com/docs/guides/gpt/function-calling](https://colab.research.google.com/drive/1-1LQEYCTMgzxJbzwJCSXQl6r1qZDgiSl?usp=sharing)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/gpt/function-calling)
- [public.work](https://public.work)

**Instructor**

- Marko Krkeljas (mkrkelja@barnard.edu)

---

## Intro

Programming is data manipulation. Every line of code we write deals with structured information, like this simple articles table. Without structure, we can't build functional software.

```python
articles = [
    {
        "title": "The Future of AI",
        "author": "Jane Smith",
        "body": "Artificial Intelligence is rapidly evolving..."
    },
    {
        "title": "The Benefits of Meditation",
        "author": "John Doe",
        "body": "Regular meditation practice can lead to numerous health benefits..."
    },
    {
        "title": "Advancements in Renewable Energy",
        "author": "Emma Wilson",
        "body": "Recent innovations in solar technology..."
    },
]
```

```python
df_articles = pd.DataFrame(articles)
df_articles
```

News websites like The New York Times rely on structured data. Each article is a set of organized fields: title, author, content, date. This structure allows enables them management of thousands of articles. A template then transforms this data into the familiar article format you see online. This approach - structured data plus templates - is the backbone of web publishing.

```python
from IPython.display import HTML, display

# Function to generate simple HTML for a single article
def article_to_html(row):
    return f"""
    <div style="margin-bottom: 20px;">
        <h2>{row['title']}</h2>
        <p><em>{row['author']}</em></p>
        <p>{row['body']}</p>
    </div>
    """

# Generate HTML for all articles
all_articles_html = "".join(df_articles.apply(article_to_html, axis=1))

print(all_articles_html)

# Very simple HTML structure
html_content = f"""
<h1>CSC Times</h1>
{all_articles_html}
"""

# Display the HTML
display(HTML(html_content))
```

Now, let's take this a step further and add tags to the articles. These could be used in a search feature to find similar articles. Instead of manually creating the tags, we can use AI to generate them dynamically.

```python
!pip install openai

import openai
from openai import OpenAI
import json

MODEL = "gpt-4o-2024-08-06"
API_KEY = ""

client = OpenAI(api_key=API_KEY)
```

```python
system_prompt = '''
Generate 3-5 tags for the provided articles.

Each article will have the following format:

{
"title": "[Article Title]",
"author": "[Author Name]",
"body": "[Article Body (may be truncated)]"
}

Guidelines:

- Analyze the title and body.
- Focus on relevant keywords and themes.
- Aim for discoverability in searches.
- Present tags as a JSON array with "tags" as the key.

Example output:

{
"title": "[Article Title]",
"author": "[Author Name]",
"body": "[Article Body (may be truncated)]",
"tags": "["Tag-1", "Tag-2", "Tag-3"]",
}

'''
```

```python
article_json = json.dumps(articles)

prompt = f"Generate tags for the following articles: \n\n{article_json}"

print(prompt)
```

```python
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt}
]

messages
```

```python
completion = client.chat.completions.create(
  model = MODEL,
  messages = messages
)

content = completion.choices[0].message.content

content
```

The response is JSON, which we need to convert to a Python dictionary. This line of code is intended to break, but occasionally it works!

```python
articles_with_tags = json.loads(content)
```

The line `articles_with_tags = json.loads(content)` won't work because:

1. The content isn't valid JSON. It's missing:
   - An enclosing array `[]`
   - Commas between objects

2. `json.loads()` expects a single, well-formatted JSON string.

3. Python's triple quotes (`'''`) don't automatically create valid JSON.

To fix this, you'd need to properly format the content as a JSON array, then use `json.loads()`.

To parse this content, you'd need to:

1. Split the string into individual JSON objects.
2. Parse each object separately.
3. Collect the results in a list.

A basic function might look like:

```python
import json

def parse_content(content):
    # Split the string into separate JSON objects
    json_strings = content.strip().split('},')

    articles = []
    for json_str in json_strings:
        # Add closing brace if it's missing
        if not json_str.strip().endswith('}'):
            json_str += '}'
        # Parse the individual JSON object
        article = json.loads(json_str)
        articles.append(article)

    return articles
```

## Structured Outputs

Instead of manually parsing the irregular JSON-like content, we're going to explore a more reliable approach using "Structured Outputs". This method allows us to guide the AI model to produce well-formatted, consistent results.

OpenAI's documentation explains:

> Structured Outputs is a feature that ensures the model will always generate responses that adhere to your supplied JSON Schema, so you don't need to worry about the model omitting a required key, or hallucinating an invalid enum value.

This powerful feature simplifies data handling and reduces errors. [You can learn more about Structured Outputs here](https://platform.openai.com/docs/guides/structured-outputs).

By leveraging Structured Outputs, we can ensure our data is always in the correct format.

```python
from pydantic import BaseModel
from typing import List

class Article(BaseModel):
    title: str
    author: str
    body: str
    tags: List[str]

class ArticleList(BaseModel):
    articles: List[Article]
```

You can reuse messages!

```python
messages
```

```python
completion = client.beta.chat.completions.parse(
    model = MODEL,
    messages = messages,
    response_format = ArticleList,
)

data = completion.choices[0].message.parsed

data
```

Convert to dictionary:

```python
articles_with_tags = [dict(article) for article in data.articles]
articles_with_tags
```

```python
pd.DataFrame(articles_with_tags)
```

Original:

```python
completion = client.chat.completions.create(
  model = MODEL,
  messages = messages
)
content = completion.choices[0].message.content
```

Updated:

```python
completion = client.beta.chat.completions.parse(
    model = MODEL,
    messages = messages,
    response_format = Colors,
)
data = completion.choices[0].message.parsed
```

Key differences:

1. API: Standard vs Beta (structured outputs)
2. Method: `create()` vs `parse()`
3. Parameters: Added `response_format`
4. Output: Raw text vs Parsed structured data

## Parsing Unstructured Data

In this next app, we're going to work with unstructured data from a web page. Specifically, we'll:

1. Make a request to Barnard College's course catalog for the Anthropology department.
2. Retrieve the raw HTML content from the page.
3. Extract the text content from the HTML, removing all markup.

This process will give us unstructured text data from the course catalog. Once we have this raw text, our next step will be to apply the "structured outputs" technique to organize and make sense of this data. This will allow us to transform the unstructured text into a more usable format.

```python
import requests
from bs4 import BeautifulSoup

def get_page_text(url):
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        page_text = soup.get_text(separator=' ', strip=True)
        return page_text
    else:
        return f"Failed to retrieve the page. Status code: {response.status_code}"

url = "https://catalog.barnard.edu/barnard-college/courses-instruction/anthropology/#coursestext"
text = get_page_text(url)
text
```

```python
len(text)
```

```python
text = text[0:2000] # Reduce the token count!
```

```python
from typing import Optional

class Course(BaseModel):
    code: str
    title: str
    points: float
    description: Optional[str] = None

class CourseList(BaseModel):
    courses: list[Course]
```

```python
completion = client.beta.chat.completions.parse(
    model = MODEL,
    messages = [
        {"role": "system", "content": "Extract the course information."},
        {"role": "user", "content": text},
    ],
    response_format = CourseList,
)

data = completion.choices[0].message.parsed

data
```

```python
for course in data.courses:
    print(course.title)
```

```python
courses_list = [dict(course) for course in data.courses]
courses_list
```

```python
df = pd.DataFrame(courses_list)

df
```

## AI Color Palette Generator

In our next app, we'll develop a proof of concept that takes any image, extracts its dominant colors, and then applies those colors in a new and creative way. This app will demonstrate the power of AI in analyzing visual data and generating structured information from it.

For image sources, you can find a great selection at [public.work](https://public.work).

Here's what we're going to do:

1. We'll define data structures to represent color information, including a `Color` class for individual colors and a `Colors` class to hold a collection of colors.

2. The user will be prompted to input a URL for any image they want to analyze.

3. We'll then use AI to extract color data from the image. The AI will analyze the image and return structured data about the colors present, including color names and their corresponding hex codes.

This approach allows us to transform unstructured visual data (the image) into structured, usable data (a list of colors with their names and hex codes). This structured data can then be used for various applications, such as generating color palettes, creating matching color schemes, or inspiring new designs based on the colors found in the original image.

```python
class Color(BaseModel):
    name: str
    hex_code: str

class Colors(BaseModel):
    colors: list[Color]
```

```python
# https://images.metmuseum.org/CRDImages/ad/original/DP119120.jpg
image_url = input("Please enter the image URL: ")
```

```python
system_prompt = '''
You are a color analysis AI expert.
When given an image, extract color information and return it in JSON format
with the following fields for each color: name, hex_code, opacity, and an optional description.
Provide accurate and comprehensive color data for the analyzed image.
'''

messages = [
    {"role": "system", "content": system_prompt},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Extract the color information from this image."},
            {
                "type": "image_url",
                "image_url": {"url": image_url},
            },
        ],
    },
]
```

```python
completion = client.beta.chat.completions.parse(
    model = MODEL,
    messages = messages,
    response_format = Colors,
)

data = completion.choices[0].message.parsed

data
```

```python
colors = [dict(color) for color in data.colors]
colors
```

---

**Credit Attribution**

The following code is based on the work of Eleanor Lutz

- **Author**: Eleanor Lutz
- **GitHub**: [https://github.com/eleanorlutz](https://github.com/eleanorlutz)
- **Source**: [https://github.com/eleanorlutz/AnimatedPythonPatterns/tree/main](https://github.com/eleanorlutz/AnimatedPythonPatterns/tree/main)

Please respect the original author's work and adhere to the license terms.

---

```python
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import patches

# Function to rotate points around a center point
def Rotate2D(pts, cnt, ang=np.pi/4):
    """Rotates points (nx2) around center cnt(2) by angle ang (in radians)."""
    return np.dot(pts - cnt, np.array([[np.cos(ang), np.sin(ang)],
                                       [-np.sin(ang), np.cos(ang)]])) + cnt

# Function to solve for a leg in a right triangle
def solveForLeg(h, leg1):
    """Uses the Pythagorean theorem to solve for the other leg (not hypotenuse)."""
    return math.sqrt(h * h - leg1 * leg1)

# Function to add a shape to the plot
def addShape(ax, points, degrees=0, alphaParam=1, ec='none', lw=0, joinstyle='round'):
    """Rotates and adds a shape to the plot."""
    origin = points[-2]
    color = points[-1]
    newPoints = points[:-2]
    pts = np.array(newPoints)
    radians = degrees * np.pi / 180
    rotated_pts = Rotate2D(pts, np.array([origin]), radians)
    ax.add_patch(patches.Polygon(rotated_pts, fc=color, ec=ec,
                                 alpha=alphaParam, joinstyle=joinstyle, lw=lw))

# Function to create a triangle
def side3(w, oX, oY, c):
    """Creates a triangle with side length w centered at (oX, oY)."""
    base = solveForLeg(w, w / 2)
    p1 = [oX + w / 2, oY - (base / 3)]
    p2 = [oX, oY + (2 * base / 3)]
    p3 = [oX - w / 2, oY - (base / 3)]
    return [p1, p2, p3, [oX, oY], c]

# Function to create a hexagon
def side6(w, oX, oY, c, e=0):
    """Creates a hexagon with side length w centered at (oX, oY)."""
    d = solveForLeg(w, w / 2)
    de = solveForLeg(w - e, (w - e) / 2)
    p1 = [oX, oY + w]
    p2 = [oX + de, oY + (w - e) / 2]
    p3 = [oX + d, oY - w / 2]
    p4 = [oX, oY - (w - e)]
    p5 = [oX - d, oY - w / 2]
    p6 = [oX - de, oY + (w - e) / 2]
    return [p1, p2, p3, p4, p5, p6, [oX, oY], c]

# Function to create a 12-sided polygon
def side12(w, oX, oY, c, e=0):
    """Creates a 12-sided polygon centered at (oX, oY)."""
    pts = side6(w, oX, oY, c)
    pts2 = side6(w - e, oX, oY, c)[:-2]
    rotated_pts = Rotate2D(np.array(pts2), np.array([oX, oY]), 30 * np.pi / 180).tolist()
    return [pts[0], rotated_pts[0], pts[5], rotated_pts[5], pts[4], rotated_pts[4],
            pts[3], rotated_pts[3], pts[2], rotated_pts[2], pts[1], rotated_pts[1], [oX, oY], c]

# Main function to generate the artwork
def starFlex(color_data):
    """Generates a static 'Star Flex' pattern using the provided color data."""
    # Extract hex codes
    hex_codes = [color['hex_code'] for color in color_data]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off')
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])

    # Background rectangle using the first color
    ax.add_patch(patches.Rectangle((0, 0), 100, 100, fc=hex_codes[0], ec='none'))

    # Coordinates and parameters
    ba = [[19.75, 50, -90], [80.25, 50, 90], [35, 24, 90],
          [65, 24, -90], [65, 76, -90], [35, 76, 90]]
    ori = [[50, 50], [4.5, 24], [4.5, 76],
           [95.5, 76], [95.5, 24], [50, 102],
           [50, -2]]
    tri = [[23, 55.65, -90], [77, 55.66, 90], [31.75, 29.65, 90],
           [68.25, 29.65, -90], [23, 44.45, -90], [77, 44.45, 90],
           [68.25, 70.45, -90], [31.75, 70.45, 90], [13.39, 50, -90],
           [86.71, 50, 90], [41.5, 24, 90], [58.45, 24, -90],
           [58.45, 76, -90], [41.5, 76, 90]]

    # Parameters for shapes
    x = 5  # Index to select parameters
    lhex = [-2, -1, 0, 2, 4, 7, 7, 7, 4, 2, 0, -1]
    lstar = [12, 11, 10, 8, 6, 3, 3, 3, 6, 8, 10, 11]
    op = [0.75, 0.7, 0.6, 0.5, 0.45, 0.4, 0.4, 0.4, 0.45, 0.5, 0.6, 0.7]
    linner = [-6, -7, -8, -6, -4, -3, -1, -3, -4, -6, -8, -7]
    linsize = [6.35, 6.6, 7, 9.5, 13.5, 18, 19.5, 18, 13.5, 9.5, 7, 6.6]
    linsize2 = [3, 3.5, 4, 5.5, 7, 9, 12, 9, 7, 5.5, 4, 3.5]
    linsize3 = [2, 2.5, 3, 4, 5, 6.5, 8, 6.5, 5, 4, 3, 2.5]
    lin2 = [-1, -2, -3, -5, -7, -9, -7, -9, -7, -5, -3, -2]
    op2 = [0.75, 0.8, 0.85, 0.95, 1, 1, 1, 1, 1, 0.95, 0.85, 0.8]

    # Function to get color cycling through the provided data
    def get_color(index):
        idx = index % len(color_data)
        return color_data[idx]['hex_code']

    # Base triangles and hexagons
    for i, item in enumerate(ba):
        color1 = get_color(i)
        color2 = get_color(i + 1)
        color3 = get_color(i + 2)
        pts = side3(11, item[0], item[1], color1)
        pts2 = side6(13, item[0], item[1], color2, lhex[x])
        pts3 = side3(22.5, item[0], item[1], color3)
        pts4 = side3(5.5, item[0], item[1], get_color(i + 3))
        addShape(ax, pts2, item[2] / 3)
        addShape(ax, pts3, item[2] / 3)
        addShape(ax, pts, item[2])
        addShape(ax, pts4, -item[2])

    # Mini triangles around the center
    for i, item in enumerate(tri):
        color = get_color(i + 4)
        pts = side3(5.5, item[0], item[1], color)
        addShape(ax, pts, item[2], alphaParam=op[x])

    # Hex stars and circles
    for i, item in enumerate(ori):
        color1 = get_color(i + 5)
        color2 = get_color(i + 6)
        circle_color = get_color(i + 7)
        circle = plt.Circle((item[0], item[1]), radius=3.5, color=circle_color)
        ax.add_artist(circle)
        pts = side12(24, item[0], item[1], color1, lstar[x])
        pts2 = side12(linsize[x], item[0], item[1], color2, linner[x])
        pts3 = side12(linsize2[x], item[0], item[1], get_color(i + 8), lin2[x])
        pts4 = side12(linsize3[x], item[0], item[1], color1, lin2[x] - 6)
        addShape(ax, pts)
        addShape(ax, pts2, alphaParam=min(1, op[x] + 0.25))
        addShape(ax, pts3, degrees=-30)
        addShape(ax, pts4, alphaParam=op2[x])

    plt.show()
```

Call the function with your color data:

```python
starFlex(colors)
```

## AI-Powered To-Do List

Our next app demonstrates the power of AI in parsing unstructured text. This technology is similar to what powers voice assistants like Siri, where spoken words are first converted to text and then parsed into a structured format compatible with specific APIs, such as the Reminders API.

Traditionally, parsing unstructured text accurately has been challenging without AI. Our app will showcase this difficulty and then demonstrate how AI can overcome it.

We'll start with a manual approach:

1. Prompt the user for input.
2. Attempt to create a table with task, priority, and category columns based on the input.
3. Demonstrate that entering plain text doesn't work, but properly formatted JSON does.

This manual method highlights the limitations of traditional parsing techniques. It requires users to input data in a specific format, which is neither user-friendly nor practical for real-world applications.

In the next step, we'll introduce an AI-powered solution that can interpret natural language input and convert it into structured data for our to-do list, showcasing the potential of AI in processing unstructured text.

```python
# "Manual" approach

from IPython.display import display, clear_output

sample = {
    "task": "Buy groceries",
    "priority": "High",
    "category": "Personal",
}

data_list = []

def add_data():
    user_input = input("Enter a JSON object or dictionary (or 'q' to quit): ")
    if user_input.lower() == 'q':
        return False
    try:
        data = eval(user_input)
        data_list.append(data)
        return True
    except:
        print("Invalid input. Please try again.")
        return True

def display_table():
    df = pd.DataFrame(data_list)
    clear_output(wait=True)
    display(df)

while add_data():
    display_table()

print("Final table:")
display_table()
```

Process:

1. Define a to-do item format.
2. Input text.
3. Use GPT-4 to interpret the text and extract information.
4. Organize the data into our defined format.
5. Display the structured result.

```python
from enum import Enum

class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class Category(str, Enum):
    WORK = "work"
    PERSONAL = "personal"
    SCHOOL = "school"

class ToDo(BaseModel):
    task: str
    priority: Priority
    category: Optional[Category] = None
```

```python
user_input = 'Finish bio homework tomorrow - really important!'

messages=[
    {"role": "system", "content": "Extract the to-do item information."},
    {"role": "user", "content": user_input},
]

completion = client.beta.chat.completions.parse(
    model = MODEL,
    messages = messages,
    response_format = ToDo,
)

message = completion.choices[0].message

message
```

## Discussion

The AI inferred "high" priority from "really important!" and "school" category from "bio homework".

These systems can derive structured information from casual inputs, making the software more adaptable and intuitive without needing explicit programming for every situation.

For users, this means they can interact naturally, without memorizing specific commands or formats.

For developers, it allows them to build more fluid, flexible, context-aware applications.

```python
# Updating the "manual" to-do method with the above.

data_list = []

def add_data():
    user_input = input("Enter a JSON object or dictionary (or 'q' to quit): ")
    if user_input.lower() == 'q':
        return False
    try:
        messages=[
            {"role": "system", "content": "Extract the to-do item information."},
            {"role": "user", "content": user_input},
        ]

        completion = client.beta.chat.completions.parse(
            model = MODEL,
            messages = messages,
            response_format = ToDo,
        )

        content = completion.choices[0].message.content

        data = eval(content)
        data_list.append(data)
        return True
    except:
        print("Invalid input. Please try again.")
        return True

def display_table():
    df = pd.DataFrame(data_list)
    clear_output(wait=True)
    display(df)

while add_data():
    display_table()

print("Final table:")
display_table()
```
