<h2>Notes of the course</h2>

<h4>First chapter - Overview of ML</h4>

1. Features and prediction

	- We can encode all features in 1 - True, 0 - False (binary features)
		I.e. length of the title > 10, sender domain "test.com"?
	- Then we have target variable: 1 if spam else 0
	- Model make prediciton: 0.7, 0.3
	- Rule: ex.  if > 0.5 => Spam else Good

	But: target can be normal figure, not binary: 2.5 thousand

2. Rule-based system vs ML apparoach
	- former has hard-coded rules with DATA and CODE as input and RESULT as output
	- later has DATA and OUTCOME as input and MODEL as an output

3. Supervised ML
	- we teach the algorithm by ourselves by feeding data and target (price) so that in 
		the future model can predict price for the car without the target
	- feature matrix is the thing where we make binary encoding. It's usually
		abbreviated as X. And we have corresponding Y matrix as a target variable
		which can be either binary result (Spam/not Spam) or price (1.5 million)
	- X is a two-dimensional array and Y is a one dimensional array.
	- y = g(X) where g() is a model, X is a matrix dataset; (X is a matrix where the
		columns of rows are `features` of observations); y is predicted value
		Ex: xi is a one row of observation with multiple features like make/model/year etc.
	- We want to be as close to y as possible. So, our goal is to train g().

	- Types of ML: regression (result is number prediction); classification (result is a category): subclass of classification is "multiclass" (result is ex: car, bus, plane) and binary (spam/not spam); ranking

4. Cross-Industry Standard Process for Data Mining (CRISP-DM)
	- Steps: 

			```bash
			business understanding <-> data understanding -> data preparation
							^
							|
			<-> modeling -> evaluation -> deployment
			```

	- At Business Understanding step we need to decide: do we need ML at all? Also
		we need to attach particular number: how much do we need to imporve our
		situation?
	- At data understanding: we need to be sure that we have all the required data or
		do we need to get some more? Also, is data reliable enough? (Users may mark
		message as spam when it's not)
	- Data preparation: here main task is transformation data into ML algorithm like
		exctracting features, cleaning noise (when message is marked as spam when it's not). We can `build pipeline` and finally convert into tabular form.
	- Modeling: linear regression/logistic regression/decision tree etc
	- Evaluation: measure how well the model performs and whether we can procede/cease
		the project
	- Deployemnt: in today's world we often test model on users (like on 10% of all
		users) and then decide whether it's good enough

	Then we have iterations over the ML project to check whether goal is satisfied.

5. Model Selection Process
	- we divide data into chunks to access our data: training, validation, testing.
		If result from those 3 is roughly the same -> can proceed, otherwise roll back.
		Again, predicitons can be either binary or numerical.
	- Such amount of division is required as we're to recheck everything.
	- Whole process: 1. split 2. train 3. validate 4. select the best model<br>
		5. test 6. check that all figures are roughly the same

	- Also, after doing so, we can unite train & validation dataset and retrain our
		model and then check again on test dataset. If everything is OK -> we proceed.

<ul>
	<p>Appendix:</p>
	<li>
		Ideally, we divide our dataset into 3 parts: training, validation, testing. At first we train various models, then we apply them on validation dataset. Pick the best one and after that apply it on testing dataset to eradicate “the multiple-comparisons problem or multiple-tests problem” (when our model gets trained so much on the same dataset that numbers we get after validation step are just lucky.)
	</li>
	<li>
		If we see that the difference in performance between validation and test is not big, we confirm that this model is indeed the best one
	</li>
</ul>

6. Linear algebra refresher
	- simple operations
	```bash
	
		2 	4     3		1	4	
	2 * 4 = 8	; 7 +	9 = 16
		6	12	  9		2   11
	```

	- Vector-vector  mulctiplication (dot product)
	```bash
		3	1	4
		7 *	9 = 63 and then sum all = 4 + 63 + 18
	 	9	2   18
	 ```
	 										3
	- if we want "row * col" mult: [2,4,5] * 6
	 										1
	 	=> at first transpose row (T) and then as usual

	- Matrix-vector multiplication

	```bash
		[0.65467089]    [0.90969502, 0.72492919, 0.06097546, 0.61336228]  
		[0.8640165 ]    [0.98125557, 0.05038929, 0.36204099, 0.78622776]
		[0.00112854]    [0.71849521, 0.24262338, 0.67413969, 0.5627456 ]
		[0.228264  ]    [0.79831637, 0.3811175 , 0.61071928, 0.72627451]
                		[0.09863846, 0.11702187, 0.59105956, 0.91169733]
    ```
	take row from matrix (right), then transpose and then multiply. After that sum.
	<h5>dimensions should match! i.e. cols in matrix must be equal to rows
		in vector
	</h5>

	- matrix-matrix multiplication
	  col1 == row2; row1 == col2
	  1. take full first matrix, transpose every row
	  2. multiply by vector of second (we take only corresponding)
	  3. sum

	- Identity matrix (I)
		- diagonal (left to right) is 1 and all other is 0
			i.e. U * I = U

	- Matrix Inverse
		- A(-1) * A = I
		- Only square matrix has inverse (rows == cols)












