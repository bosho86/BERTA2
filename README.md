In this paper, we present a pruning and fine-tuning
framework for compressing BERT for deployment on resource
constrained hardware. Our method identifies a three-step process
to achieve automatic pruning on BERTBASE, and deployment of
this compressed model onto the popular edge device Raspberry
Pi 5. We evaluate the approach on GLUE tasks (SST-2), demonstrating strong pre and post-fine-tuning accuracies compared
to BERTTickets. We provide a theoretical hardware analysis,
estimating FPGA speedups relative to CPU and GPU, and show
a 30% sparsity leads to a 1.43x theoretical latency improvement
on a Xilinx Alveo U200. Our edge deployment on a Raspberry
Pi 5 shows how resource constrained hardware compares with
larger CPUs and GPUs. These results highlight high viability on
pruning driven compression for edge deployments.
Transformer-based models have achieved remarkable suc-
cess across a wide range of natural language processing
(NLP) tasks, including sentiment classification [8], question
answering [9], text summarization [10], and language model-
ing [11]. Among them, BERT [7] has demonstrated substantial
improvements on the GLUE benchmark [12], a widely adopted
suite for evaluating language understanding systems. As deep
learning becomes increasingly part of industry [13], there is
growing interest in deploying BERT on resource-constrained
devices that require low-latency and high-accuracy inference.
Prior work has explored identifying sub-networks of the
BERTBASE model with desirable sparsity and accuracy [14]–
[16], but these approaches rely on expert-designed heuristics
that are often time-consuming and fail to yield globally
optimal solutions [17]. Automatic weight is computationally
expensive, requiring repeated sub-network sampling and multi-
epoch retraining. Moreover, pruning Transformer models re-
mains particularly challenging due to the sensitivity of syntax
and semantics in language tasks.
Motivated by these limitations, we introduce a simple prun-
ing framework for BERT that samples candidate sub-networks,
evaluates them without fine-tuning, and subsequently fine-
tunes the most promising candidates. Our approach aims to
identify high-quality sub-networks without requiring expert in-
tervention and eliminates the need for the thousands of training
epochs typical of existing automatic pruning methods
