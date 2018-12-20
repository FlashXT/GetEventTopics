import re

text = '0.050*"wine" + 0.027*"tasting" + 0.020*"social" + 0.014*"cruise" + 0.010*"advance" + 0.010*"sell" + 0.009*"boat" + 0.009*"attend" + 0.009*"attending" + 0.009*"join"'

items = re.findall(r'(?<=\*").*?(?=")',text)
for item in items:
    print(item)