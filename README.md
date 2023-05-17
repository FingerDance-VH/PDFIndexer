# PDFIndexer
---
### PDFIndexer基本描述
 PDFIndexer 编码 PDF 的文本和图像，同时提取标题并计算 PDF 的大小。
 
### PDFIndexer完成的工作
 首先提取PDF中的文本、图片、标题并计算PDF文件的大小，然后使用xlm-Roberta-base模型和clip-vit-large-patch14模型分别对pdf中的文本和图片进行编码，文本编码、图像编码、标题、大小和图像就是PDFIndexer最终输出的参数，最终输出是嵌套的，例如：doc.chunks[0]是一个文本文档，doc.chunks[0].chunks才是文本张量列表。
 
### 重要声明
 本代码仅用于学习交流。
  
 ---

### Basic description
PDFIndexer encodes TEXTs and IMAGEs of the PDFs, while extracting the title and caculating the size of the PDFs.

### Work done by PDFIndexer
Firstly, text, picture and title in PDF are extracted and the size of PDF file is calculated. Then, xlm-Roberta-base model and clip-vit-large-patch14 model are used to encode text and picture in pdf respectively. These are the final output parameters of PDFIndexer. The final output is nested, for example, doc.chunks[0] is a text document, doc.chunks[0].chunks is a list of text tensors.

### Important statement
This code is for learning communication only.
