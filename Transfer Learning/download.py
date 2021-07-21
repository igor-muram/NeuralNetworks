import requests
import sys
from PIL import Image

urls = sys.argv[1]
path = sys.argv[2]
count = int(sys.argv[3])

file = open(urls, 'r', encoding="utf-8")
lines = file.readlines()

i = 1 

for line in lines:
   startpos = line.rfind(".")
   endpos = line.rfind("\\")
   extension = line[startpos + 1: endpos]

   try:
      response = requests.get(line, stream=True, timeout=1)
      
      if response.ok:
         try:
            img = Image.open(response.raw)
            img.save("{path}/{n}.{ex}".format(path = path, n = i, ex = extension))

            i += 1
            if i > count:
               break

         except:
            print("Parse image error")
      
   except:
      print("url request error")