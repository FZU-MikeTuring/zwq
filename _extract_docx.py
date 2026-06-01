import zipfile, xml.etree.ElementTree as ET, os

docx = os.path.join(os.path.dirname(os.path.abspath(__file__)), "152201226_张雯倩2 (1).docx")
zf = zipfile.ZipFile(docx)
xml_content = zf.read('word/document.xml')
tree = ET.fromstring(xml_content)
paragraphs = []
for p in tree.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
    texts = []
    for t in p.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'):
        if t.text:
            texts.append(t.text)
    if texts:
        paragraphs.append(''.join(texts))
full = '\n'.join(paragraphs)
print(full[:20000])
