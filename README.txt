Machine Learning HW 1 - Abhimanshu Mishra
Email: amishr11@binghamton.edu
B Number- B00817746

Academic Honesty Statement:
I have done this assignment completely on my own. I have not copied it, nor have
I given my solution to anyone else. I understand that if I am involved in
plagiarism or cheating I will have to sign an official form that I have cheated
and that this form will be stored in my official university record. I also
understand that I will receive a grade of 0 for the involved assignment for my
first offense and that I will receive a grade of “F” for the course for any
additional offense.
-Abhimanshu Mishra


External Libraries Used:

numpy - for matrix operations

Run Instructions:

1. Create a virtual environment with Python3.6
2. Install the required libraries using the command: pip install -r requirements.txt. Run this command in the same directory as the requirements file.
3. Continue to execution

Execution:

To use the command line: python ans.py <training-data-file-path> <dev-data-file-path> <test-data-file-path> <yes/no> <heuristic>
Example command: python ans.py dataset1/train_data.csv dataset1/dev_data.csv dataset1/test_data.csv yes entropy
This example command will build a decision tree using the entropy heuristic for the first dataset. If the 'to-print' flag is set to yes, the entire tree is printed along with its accuracy. If it is set to no, the accuracy for that dataset and metric will be appended to the end of the file accuracy.txt.

To run all heuristic possibilities on both datasets available in the homework, run the command: sh run_all.sh. This assumes that the dataset is in the same folder as the one where you're running the code from. It also assumes that the names of the dataset folders are the same as the ones that were provided to us.

Data Structures:

Node() is defined and used to represent one node of the decision tree being built.
