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

- Algorithms

	- Word embeddings

		- Quote paper

	- Deep CBOW

	- LSTM

		- Quote paper

	- Tree LSTM

		- Quote papers

## Models

### Explain implementation of models generically

- Figure

- “How do we do classification”

	- Logistic regression

### Table

- (for each model)

	- Model-name

	- Loss function

	- Nr of hidden layers

	- Hidden layer size

	- Nr of gates

## Experiments

### Data and resources

- Embeddings

	- Explain GLOVE

- DATA

	- Stanford data movie reviews short for 

		- Explain supervision signal (sentiment)

			- Per node

			- Per root

		- (We explain the task as): Infer sentiment for unseen data (testing)

### Training the models

- The `train_model` function

### Evaluation

- Generic evaluation performance metric

### ❗ Research-question specific

- Subsubsection (for each research question)

	- Explain per research question differences

## 📊 Results and Analysis

### ❗ Research-question specific

- Subsection for q-sections

	- Question 1 and 2

		- Results in tabular form

			- How important is word order for this task

				- Accuracy/variance for Deep CBOW vs LSTM

			- Does tree structure help for accruacy

				- Accuracy/variance for LSTM vs Tree LSTM

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

