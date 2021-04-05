1. Download python 3.8
2. Create a virtual environment with the command in the application folder:
        python3.8 -m venv env
3. Activate the environment
        source env/bin/activate 
4. Install all the requirements from the requirement.txt file to install all the dependencies
	    pip install -r requirements.txt 

5. Run previous phase tasks to generate the similarity matrix(Apart from the directory path, rest of the values are defined)
    python phase2_task0a.py

    Enter the directory path(relative or absolute)
    Enter resolution(integer): 3
    Enter window size(integer): 3
    Enter split size(integer): 3

    python phase2_task0b.py

    Enter the directory path(relative or absolute)

    python phase2_task1.py

    Enter the directory path
    Choose one of vector recognition file: tf, tf-idf: tf-idf
    Choose one vector model: pca, svd, nmf, lda: pca
    Choose number of components: 5

    python phase2_task3.py
    
    Choose one of the model from the user option to create gesture gesture similarity matrix: PCA
    Enter the number of components: 5
    Choose one of the model from SVD or NMF to find the latent semantics: NMF

6. Run task 1
    python task1.py

    Enter the directory path(relative or absolute)
    Enter resolution(integer)
    Enter window size(integer)
    Enter split size(integer)

7. Run task 2
    python task2.py

    Enter the directory path(relative or absolute)

8. Run task 6 which will subsequently run task 3, 4 and 5 in the background on subsequent requests
    python task6.py

    Enter the Query Parameter
    Choose one of the feedback mechanism. Probabilistic Based Relevance Feedback(Task 4) and Classifier Based Relevance Feedback(Task 5)
    Enter the number of layers
    Enter the number of hashes per layer
    Enter the numebr of relevant gestures required for output