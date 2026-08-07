# README

> Currently serves to showcase current state of project and improvements to make.

---

## 1. Overview

**Project name:** Autograd Engine v2

**One-line description:** C++ implementation of an autograd engine to understand backpropagation.

## DISCLAIMER: predict_sample.cpp and test_model.cpp were written by GPT 5.6 Terra High. 

## 2. Notes
**1. Batched Autograd**

Three different approaches considered for Batched autograd
 1. Loop over batches and epochs seperately in network.h and leave autograd.h 
    (slow but easy)

 2. Assume all tensors that are passed into autograd.h are 3 dimensional (for batches)
    and treat the 0th dimension as iteration count
    and operate on the other 2 dimensions as 2d matrices 
    (mid but nice generalization)

 3. Write specific code for each Node with specific assumptions 
    about the 2d matrices (parameters) and the 3d matrices (batched data) 
    
    (faster but no generalization) - currently using


**2. Graph Based vs Tape-based Autograd**

- With graph-based, every forward pass creates a new graph and every backward pass starts a new recursive topological sort. -> very inefficient

- Solution: 


  Tape based allows backward pass to be simple O(n) because operations are recorded as they are initialized into the tape.


**3. Graph build for every forward and backward pass**

- Right now, every forward and backward pass pair are building a graph.

- Solution: 
  
  Instead, we could build the graph once and rewrite the values in it for each pass pair



