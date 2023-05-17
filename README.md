# PDFIndexer

---

### PDFIndexer Basic description
PDFIndexer encodes the text and images of the PDF while extracting the title and calculating the size of the PDF.

### Work done by PDFIndexer
Firstly, text, picture, cover image and title in PDF are extracted and the size of PDF file is calculated. Then xlm-Roberta-base model and clip-vit-large-patch14 model are used to encode text and picture in pdf respectively. These are the final output parameters of PDFIndexer. The final output is nested, for example, doc.chunks[0] is a text document, doc.chunks[0].chunks is a list of text tensors. tags['title'] and tags['size'] are written directly into doc.tags['title'] and doc.tags['size'], perhaps slightly confusing [doge].

### Important statement
①sandbox cannot be used normally, because it always causes problems when introducing xlm-Roberta-base model and clip-vit-large-patch14 model, it is recommended to use Jina/Source;\n
②This code is only used for learning and communication.

---

### PDFIndexer基本描述
 PDFIndexer 编码 PDF 的文本和图像，同时提取标题并计算 PDF 的大小。
 
### PDFIndexer完成的工作
 首先提取PDF中的文本、图片、封面图、标题并计算PDF文件的大小，然后使用xlm-Roberta-base模型和clip-vit-large-patch14模型分别对pdf中的文本和图片进行编码，这些就是PDFIndexer最终输出的参数。最终输出是嵌套的，例如：doc.chunks[0]是一个文本文档，doc.chunks[0].chunks才是文本张量列表。而标题、文件大小则直接写进doc.tags['title']和doc.tags['size']，或许稍微有一点混乱[doge]。
 
### 重要声明
①sandbox无法正常使用，因为它总是会在引入xlm-Roberta-base模型和clip-vit-large-patch14模型时出问题，建议通过Jina/Source进行使用；
②本代码仅用于学习交流。
