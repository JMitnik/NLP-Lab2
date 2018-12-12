- How important is word order for this task?

  We can compare the performance of BOW-like model directly to the experiment of LSTM. But the thing is that it might not be fair to compare two very different models like LSTM and Deep CBOW. So in order to study the importance of the word order, we can train an LSTM with a corpus where the words in all sentence are randomly shuffled.

  To do this, we only need to write a new `prepare_minibatch_shuffled` function, referring the original `prepare_minibatch` function.

- Does the tree structure help to get a better accuracy?

  A direct comparison of test set accuracy with proper variance indicated and statistic test reported.

- How does performance depend on the sentence length? Compare the various models. Is there a model that does better on longer sentences? If so, why?

  Since all the training and evaluating stuff are done in a single `train_model` function, all we need to do is to hack this function. My method would be rewrite the `eval_fn` to take an extra `sent_scale_range` parameter, e.g.`[0,5]`, to filter those sentences that don't fall into this range. One can pass the partialed version(using `partial` from `functools` to feed the `scale_range` beforehand) to the `train_model` and get the test accuracy plot during the training, also the final best model test accuracy. But the shortcoming is if we want the accuracy plots for many ranges, probably can't be easily hacked. Fortunately we can always load the best model(saved during training) and use the `eval_fn` mentioned before to get the accuracy with different length sentences, which is all we need for this question.

- Do you get better performance if you supervise the sentiment **at each node in the tree**? You can extract more training examples by treating every node in each tree as a separate tree. You will need to write a function that extracts all subtrees given a treestring.

  A straightforward implementation would be rewrite a version of `prepare_treelstm_minibatch`, extract every subtree of each tree in the batch and merge them together to a mega-batch. But it is highly inefficient in that the subtree is usually much smaller in dimension, so most of the space is used as paddings. Also we need an elegant way to extract the subtrees given only tokens and transitions.



It is neater to write the scripts for those research question separately, but all of those stuff will inevitbly use something from the main notebook. One solution there is to mask out all the training code and store the notebook as a `.py` file, and download it at the beginning the new notebook in realtime,  then import all the symbols from it. This way we can work on new notebook without having to take care the original one.