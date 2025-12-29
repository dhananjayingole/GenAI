from langchain_text_splitters import CharacterTextSplitter

text = """54,354 views  Apr 3, 2025  Generative AI using LangChain
Code - https://github.com/campusx-official/l...

My Notes: https://learnwith.campusx.in/products...

Did you like my teaching style?
Check my affordable mentorship program at : https://learnwith.campusx.in
DSMP FAQ: https://docs.google.com/document/d/1O...
============================

📱 Grow with us:
CampusX' LinkedIn:   / campusx-official  
Slide into our DMs:   / campusx.official  
My LinkedIn:   / nitish-singh-03412789  
Discord:   / discord  
E-mail us at support@campusx.in

⌚Time Stamps⌚

00:00 - Intro
00:37 - Text Splitting
09:55 - Length based Text Splitting
23:31 - Text-Structure based Text Splitting
39:05 - Document-Structure based Text Splitting
48:26 - Semantic Meaning Based"""

splitter = CharacterTextSplitter(
    chunk_size = 40,
    chunk_overlap = 0,
    separator=''
)

result = splitter.split_text(text)
 
print(result)