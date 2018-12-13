# **Final_report_schema**


## Introduction

### Task

- Classify sentiment of corpus

### Research questions

- 📋 List

	- Obligatory “research questions: 4 ordered-list points

		- How important is word order for this task

			- Hypothesis

			- Importance

			- Literature

			- Findings Short summary

		- Does tree structure help for accruacy

			- Hypothesis

			- Importance

			- Literature

			- Findings Short summary

		- Performance on sentence length

			- Hypothesis

			- Importance

			- Literature

			- Findings Short summary

		- Performance on sentiment per node

			- Hypothesis

			- Importance

			- Literature

			- Findings Short summary

### Approach (high-level)

- Train different models and compare

## Background

### Explain

- Word embeddings

	- Quote paper

	- Embeddings

		- Explain GLOVE

			- ☑️ Steps

				- 

- Algorithms

	- Deep CBOW

		- First introduce CBOW

			- CBOW is like BOW, except the features aren’t counts, but vectors of arbitrary size

				- Dimension: R^D

				- These are semantic representations of a vector, not something we can interpret

			- We sum these like in BOW

			- By using the parameter W, we can learn the relationship between the summation of these semantic representations and the possible outputs

				- Dimension of W:  R^(5xD)

		- Extension on regular CBOW

			- Same idea, except now, the output now consists of a number of layers

				- Affine transformations from dimension E to 5, with Tanh activations in between

	- LSTM

		- Quote paper

	- Tree LSTM

		- Quote papers

## Models

### Explain implementation of models generically

- Figure

- “How do we do classification”

	- Logistic regression

###  ❓Optional/Potential

- Parameters

	- 📋 Example from Tree LSTM

### Table

- (for each model)

	- Model-name

	- Loss function

	- Nr of hidden layers

	- Hidden layer size

	- Nr of gates

## Experiments

### Data and resources

- DATA

	- Stanford data movie reviews short for 

		- Explain supervision signal (sentiment)

			- Per node

			- Per root

		- (We explain the task as): Infer sentiment for unseen data (testing)

		- 📰 About the paper

			- Ignoring word order is not plausible, cannot classify negation for instance or weakly sentimental pbhrases

			- Description of the dataset

				- Consists of 10,662 excerpt sentences from different movie reviews

					- Each sentence has as label the overall opinion of a writer’s review

					- In total, consists of 215,154 phrases

						- Each phrase has 

							- Which were annotated by MTurk

				- From length 20, full sentences

					- The longer the sentence, the less neutral it tends to be

			- Data is ‘fine-grained sentiment classified’

			- 👍

				- They are able to capture negation

### Three instances of model trained

- Training the models

	- Each model is initially trained using the train_model

		- Different parameters

			- Each model is trained for 30000 iterations

				- Except for vanilla LSTM, which gets trained for 25000

			- Minibatch prep (size of a mini batch is 25)

				- Mini-lstm is the minibatched variant, gets minibatched, and then prepped

				- Tree lasts get minibatched and then 

			- Learning rate

			- Parameters

	- Each model uses Adam for optimization

		- 💬 Adam

			- Variety of SGD with decaying learning rate

			- Adam maintains learning rates for each parameter

			- ❓ How does adam adapt LR

				- Uses an exponential moving average of the gradient (mean) and squared gradient  (variance)

				- Uses parameter beta1 and beta2 to control decay rate of these moving averages

					- Starts as 1

			- Adam has shown to work on sentiment analysis on IMDB sentiment analysis dataset

				- 🤔 No paper found so far

		- Has default parameters as decided by pytorch

			- lr={{variable}}, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False

### Evaluation

- Generic evaluation performance metric: Accuracy/variance

### ❗ Research-question specific

- Subsubsection (for each research question)

	- Explain per research question differences

		- Note: Each result is averaged over three different seed-trained model

			- RQ 1.  We set LSTM against CBOW , compare their accuracy/variance

			- RQ 2. We compare LSTM with Tree LSTM and their accuracy/variance

			- RQ 3.  We group our test set based on sentence length, in bins of range 5. We compare the performance of each bin for all of our models, averaged over three seeds

			- RQ 4. We compare a tree model which was supervised per node, and one which was supervised per root node

## 📊 Results and Analysis

### ❗ Research-question specific

- Subsection for q-sections

	- Question 1 and 2

		- Results in tabular form

			- Does tree structure help for accruacy

				- Accuracy/variance for LSTM vs Tree LSTM

			- How important is word order for this task

				- Accuracy/variance for Deep CBOW vs LSTM

	- Question 3

		- Plot #2

			- Performance on sentence length

				- CBOW/DEEP CBOW/ LSTM/ TREE LSTM accuracy/variance/ maybe smth else

	- Question 4

		- Results in tabular form

			- Performance on sentiment per node

## 🔚 Conclusion & Discussion

### Relation to prior literature

- Literature: Tree LSTM (3 different papers)

	- Question 2 results

	- Question 3 results

	- Question 4 results

- Literature: Original LSTM paper?

	- ☑️ Make sure we have a good paper otherwise to refer to

		- Maybe find a blog which refers to some paper

	- Question 1 results

### Answer our hypotheses

- Aka, do our results match, explain

### What next?

- Any improvements, future research, etc

## Appendix

