# -*-coding:utf-8 -*-
import os
import codecs
import chardet

'''
Convert all files to UTF-8 code format
'''


def convert(filename, out_enc="utf-8"):
    content = codecs.open(filename, 'r').read()
    source_encoding = chardet.detect(content)['encoding']
    if source_encoding == None:
        print(filename, source_encoding)
    elif not str(source_encoding) == out_enc:
        print(filename, source_encoding)
        try:
            content = content.decode(str(source_encoding), 'ignore').encode(out_enc)
            codecs.open(filename, 'w').write(content)
        except UnicodeDecodeError or TypeError as error:
            print("error file:", filename)
            print(error)


def explore(dir):
    print("Decoding and encoding all files to utf-8 format...")
    for root,dirs,files in os.walk(dir):
        for file in files:
            if os.path.splitext(file)[1]=='.txt':
                path=os.path.join(root,file)
                convert(path)
    print("Encoding finished.")


if __name__ == "__main__":
    print("Decoding and encoding all files to utf-8 format...")
    explore("./Data")