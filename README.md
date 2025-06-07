# Image Processing Examples
This repository contains various examples of computer vision and image processing techniques used to solve real-world problems. The focus is on tasks such as document processing, optical answer sheet analysis, and object counting using computer vision algorithms.

---------------

## Project Overview
The project is divided into three main sections:

- **Document Processing and Masking**

 A document captured on an appropriate surface will be processed by masking the areas outside the document. The contrast of the document will be enhanced, and the background will be set to pure white if necessary.The corner points or boundary lines of the document (top-left, top-right, bottom-left, bottom-right) will be detected. If the document has a trapezoidal shape, it will be corrected.

-  **Optical Answer Sheet Analysis**

An optical answer sheet will be scanned, and at least 10 different answer sheets will be processed to extract the student number, number of correct answers, wrong answers, and blank responses.

-  **Granular Object Counting**

An image of granular objects (e.g., rice, corn) on a white surface will be analyzed to count the number of individual objects.

---------------

## Installation

Clone the repository:
~~~
git clone https://github.com/FezaiTech/image-processing-examples.git
~~~

Navigate to the project directory:
~~~
cd image-processing-examples
~~~

Install the required dependencies:

~~~
pip install -r requirements.txt
~~~

---------------
## Contributing

Feel free to fork this repository, submit pull requests, or open issues if you have any suggestions or improvements.
