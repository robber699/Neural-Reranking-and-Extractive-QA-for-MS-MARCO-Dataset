# Team information

**Student 1:** 11902050 Chen Anni

**Student 2:** 12302504 Carminati Roberto

**Student 3:** 11905145 Gülmez Gülsüm

# Report

## Part 1

This Part focuses on the aggregation of raw judgements. We first describe the hypothesis we used for the
aggregation. Then we dive deeper into the steps followed to implement the aggregation process. Finally, we present some results of our aggregation.

### Hypothesis

For the aggregation of raw judgements, we considered two hypotheses:

1. If a judgement was made significantly faster than the average time it takes to do this judgement, then this judgement
   should have less impact


2. If an annotator uses category c very often, then this might be a sign that the annotator is overusing c and we
   should give her votes for c less weight. However, if c is also a frequent choice of the population of annotators at large,
   then this might again temper that effect.

Hypothesis 2 is adapted from the paper "Empirical Analysis of Aggregation Methods for Collective Annotation" by
Qing et al. (2014).

### Implementation

We use pandas (v1.1.5)

The main idea is to group the raw judgements by ***queryId*** and ***documentId*** and implement a custom aggregation
function, which would choose the relevance grade based on the computed weights of the judgements. The weights consider
the previous hypotheses.


**Hypothesis 1:**


Within a group, we compute the mean time with the column ***durationUsedToJudgeMs***. Then, we iterate over each
judgement in the group, compare the individual time with the mean time. We also used a threshold of 0.3 to allow annotators
to be slightly faster than the average. Judgements which are beyond this threshold are penalized with a reduced weight (-20%).


**Hypothesis 2:**


We prepared an individual frequency table and a global frequency table.

**individual frequency table**

| userId | 0_NOT_RELEVANT | 1_TOPIC_RELEVANT_DOES_NOT_ANSWER | 2_GOOD_ANSWER | 3_PERFECT_ANSWER |
|--------|----------------|-----------------------------------|---------------|------------------|

This table contains the number of times each annotator used a category divided by the total number of his made judgements.


**global frequency table**

| 0_NOT_RELEVANT | 1_TOPIC_RELEVANT_DOES_NOT_ANSWER | 2_GOOD_ANSWER | 3_PERFECT_ANSWER | total_judgements |
|----------------|-----------------------------------|---------------|------------------|------------------|

This table contains the number of times each category was used by all annotators divided by the total number of judgements.


With the help of these tables, we initialize the weights for each judgement. We use the *difference-based* approach:
1 + global_freq(k) - inv_freq(k)

### Result Analysis

Beneath are some examples of the aggregated judgements. Judging from these results, we can say that the quality of
the aggregation is quite good. The relevance grades are consistent with the document text and the query text.

___
- **queryId**: `db_q_<dbpedia:Lu_Chuan>`
- **query text**: lu chuan
- **documentId**: `db_<dbpedia:City_of_Life_and_Death>`
- **document text**: city of life and death city of life and death is a 2009 chinese drama film written and directed by lu chuan ,
  marking his third feature film . the film deals with the battle of nanjing and its aftermath ( commonly referred to as the rape of nanking or the nanking massacre) during the second sino - japanese war .
- **relevance**: 2
___
- **queryId**: `db_q_<dbpedia:BlackBerry_Priv>`
- **query text**: whats the name of the blackberry Priv mobile phone that comes with android
- **documentId**: `db_<dbpedia:App_store_optimization>`
- **document text**: app store optimization app store optimization ( aso ) is the process of improving the visibility of a mobile app ( such as an iphone , ipad , android , blackberry or windows phone app ) in an app store ( such as itunes for ios , google play for android or blackberry world for blackberry ) . just like search engine optimization ( seo ) is for websites , app store optimization ( aso ) is for mobile apps .
- **relevance**: 0
___
- **queryId**: `rob_qq_FT943-14268`
- **query text**: Where is Apple Computer UK Ltd located?
- **documentId**: `rob_FT943-14268`
- **document text**: ft 15 jul 94 / technology ( worth watching ) : apple launches new operating system by vanessa houlder apple , the california - based computer company , is set to release a new operating system for its macintosh personal computers later this summer . the new system , known as macintosh system 7 . 5 , includes more than 50 new features and technologies to make the computer more productive and easier to use . these include an interactive guide to assist users , some features to streamline and speed up basic tasks and a simplified way to exchange information between macintosh and ms - dos or windows systems . the system includes powertalk , which allows users to send electronic mail , share files and forward documents . it also includes the quickdraw gx technology for high - quality printing and graphics . macintosh system 7 . 5 will run on computers with at least a 68020 processor . on a 68020 , 68030 or 68040 - based macintosh computer , the system requires at least four megabytes of ram to run the core elements and at least eight megabytes of ram to use powertalk and quickdraw gx . the new release will be compatible with most macintosh applications software currently available . apple computer uk ltd : uk , 081 730 2480 . countries : - usz united states of america . industries : - p3577 computer peripheral equipment , nec . types : - tech products & product use . the financial times london page 12
- **relevance**: 1
___
- **queryId**: `db_q_<dbpedia:.hn>`
- **query text**: what is the country code for honduras
- **documentId**: `db_<dbpedia:.hn>`
- **document text**: . hn . hn is the internet country code top - level domain ( cctld ) for honduras .
- **relevance**: 3
___

## Part 2

This part of the report presents the implementation and evaluation of two neural architectures based on the kernel-pooling paradigm - Kernel-Pooling Neural Re-ranking Model (KNRM) and Transformer-Kernel (TK) model - for performing re-ranking tasks.
The KNRM model applies kernel-pooling over the interaction matrix formed between query and document embeddings. Multiple Gaussian kernels are used to capture different levels of query-document similarity.
The TK model integrates transformer layers with kernel-pooling. The transformer layers encode the query and document into richer representations before computing the interaction matrix.

**Training process & Result evaluation**

The training process involves feeding the models with query-document pairs and optimizing the relevance scores using a suitable loss function. An early stopping mechanism based on the validation set performance is employed to prevent overfitting.
The early stopping implementation includes a parameter “patience” (set to 3 in our case) which indicates the number of epochs in a row tolerated to not improve the results before stopping the training process.
In training, Adam optimizer has been used, which adapts the learning rate for each parameter, leading to efficient and faster convergence. Margin Ranking Loss to encourage the model to rank relevant documents higher than irrelevant ones by maximizing the score margin between them. The learning rate was set to 0.015 for both models. Embedding dimension was set to 300 .

During the validation process, the model trained in each epoch was asked to perform predictions on the validation MS MARCO dataset (msmarco_tuples.validation.tsv), obtaining a ranked list of documents ordered based on relevance for each query. Comparing this list with the relevant labels of MS MARCO (msmarco_qrels.txt), metrics have been calculated.

The Kernel-Pooling Neural Re-ranking Model (KNRM), trained for 1 epochs with 11 kernels and a batch size of 1024, shows this metrics:

| Metric    | Value  |
|-----------|--------|
| MRR@10    | 0.0723 |
| MRR@20    | 0.0812 |
| Recall@10 | 0.5746 |
| MAP@1000  | 0.0885 |
| nDCG@10   | 0.0919 |
| nDCG@20   | 0.1245 |

**Table 1: Validation Metrics of KNRM trained for 1 epochs with 11 kernels and a batch size of 1024**

The Transformer-Kernel model (TK), trained for 1 epochs with 2 layers, 4 kernels, batch size of 128, and an attention matrix dimensions of 30, shows these metrics:

| Metric    | Value  |
|-----------|--------|
| MRR@10    | 0.1972 |
| MRR@20    | 0.2035 |
| Recall@10 | 0.3953 |
| MAP@1000  | 0.2029 |
| nDCG@10   | 0.2425 |
| nDCG@20   | 0.2656 |

**Table 2: Validation Metrics of Transformer-Kernel model (TK), trained for 1 epochs with 2 layers, 4 kernels, batch size of 128, and an attention matrix dimensions of 30**

The KNRM model shows moderate performance across most metrics. It achieves a reasonable Recall@10 of 0.5746, indicating that for over half of the queries, relevant documents are found within the top 10 ranked results. However, the MRR and nDCG scores suggest room for improvement, particularly in how well the model ranks relevant documents higher and optimizes the cumulative relevance across various cutoff points.

In contrast, the TK model demonstrates significantly improved performance across all metrics. It achieves higher MRR scores, indicating better ranking of relevant documents at the top positions (MRR@10: 0.1972, MRR@20: 0.2035). The MAP@1000 and nDCG@20 scores also indicate that the TK model consistently retrieves and ranks relevant documents more effectively across broader ranges. Additionally, the TK model's Recall@10 of 0.3953 indicates that approximately 39.53% of queries retrieve relevant documents within the top 10, which is notably lower compared to KNRM.

In conclusion, the model TK, even trained for just 1 epoch for computational reasons, shows in general a better performance along most metrics, with just a decrease in the Recall@10.

### Test Set Evaluation

#### MS-MARCO Sparse Labels
The model saved during the validation process is now used on new data for testing. Feeding both the KNRM and TK models with the test MS MARCO data (msmarco_tuples.test.tsv) and assessing the performances against the relevance judgments (msmarco_qrels.txt), the following metrics are obtained:

**KNRM**

| Metric    | Value  |
|-----------|--------|
| MRR@10    | 0.0801 |
| MRR@20    | 0.0892 |
| Recall@10 | 0.1747 |
| MAP@1000  | 0.0976 |
| nDCG@10   | 0.1014 |
| nDCG@20   | 0.1352 |

**Table 3: Test Metrics of KNRM on MSMARCO Sparse Labels**

**TK**

| Metric    | Value  |
|-----------|--------|
| MRR@10    | 0.2011 |
| MRR@20    | 0.2072 |
| Recall@10 | 0.4238 |
| MAP@1000  | 0.2070 |
| nDCG@10   | 0.2519 |
| nDCG@20   | 0.2740 |

**Table 4: Test Metrics of TK on MSMARCO Sparse Labels**

The KNRM model shows moderate performance, with an MRR@10 of 0.0801 and Recall@10 of 0.1747. The nDCG@10 score of 0.1014 suggests that relevant documents are often not ranked at the very top. The TK model demonstrates slightly better performance compared to KNRM. With an MRR@10 of 0.2011 and Recall@10 of 0.4238, it effectively ranks more relevant documents at the top positions. The nDCG@10 score of 0.2519 indicates a much better ranking of relevant documents, and the MAP@1000 of 0.2070 shows its superior performance across retrieved documents.

#### FiRA-2022 Fine-Grained Labels
Next, the models have been evaluated using the FiRA-2022 fine-grained tuples (fira-2022.tuples.tsv), which contain out-of-domain data, along with the query relevance judgments created in Part 1 of the project.

**KNRM - Implemented Qrels**

| Metric    | Value  |
|-----------|--------|
| MRR@10    | 0.8733 |
| MRR@20    | 0.8734 |
| Recall@10 | 0.9681 |
| MAP@1000  | 0.8601 |

**Table 5: Test Metrics of KNRM on FiRA-2022 Fine-Grained Tuples with Part 1 Implementation Query Relevance Judgments**

**TK - Implemented Qrels**

| Metric    | Value  |
|-----------|--------|
| MRR@10    | 0.922  |
| MRR@20    | 0.9221 |
| Recall@10 | 0.9083 |
| MAP@1000  | 0.9022 |

**Table 6: Test Metrics of TK on FiRA-2022 Fine-Grained Tuples with Part 1 Implementation Query Relevance Judgments**

The KNRM model, when tested on MS MARCO sparse labels, achieved an MRR@10 of 0.0801 and Recall@10 of 0.1747. In contrast, testing on FiRA-2022 fine-grained labels provides better results, with an MRR@10 of 0.8733 and Recall@10 of 0.9681. The MAP@1000 also showed a substantial improvement, rising from 0.0976 on MS MARCO to 0.8601 on FiRA-2022.

Similarly, the TK model demonstrated enhanced performance on FiRA-2022 fine-grained labels compared to MS MARCO sparse labels. On MS MARCO, the TK model had an MRR@10 of 0.2011 and Recall@10 of 0.4238, whereas on FiRA-2022, these metrics improved to an MRR@10 of 0.922 and Recall@10 of 0.9083. The MAP@1000 for the TK model increased from 0.2070 on MS MARCO to 0.9022 on FiRA-2022.

There is an important difference in performance between the MS MARCO sparse labels and FiRA-2022 fine-grained labels. One reason could be that FiRA-2022 fine-grained labels may provide more detailed and accurate relevance judgments, leading to better training and evaluation outcomes.

Next, to evaluate the models using the baseline label creation method, the FiRA-2022 tuples (fira-2022.tuples.tsv) were fed into the neural models, and the performance was assessed against the query relevance judgments provided (fira-2022.baseline-qrels.tsv).

**KNRM - Baseline Qrels**

| Metric    | Value  |
|-----------|--------|
| MRR@10    | 0.8848 |
| MRR@20    | 0.8848 |
| Recall@10 | 0.9125 |
| MAP@1000  | 0.8725 |

**Table 7: Test Metrics of KNRM on FiRA-2022 Fine-Grained Tuples with Baseline Query Relevance Judgments**

**TK - Baseline Qrels**

| Metric    | Value  |
|-----------|--------|
| MRR@10    | 0.9332 |
| MRR@20    | 0.9332 |
| Recall@10 | 0.9131 |
| MAP@1000  | 0.9156 |

**Table 8: Test Metrics of TK on FiRA-2022 Fine-Grained Tuples with Baseline Query Relevance Judgments**

Comparing the performance of both KNRM and TK models using the implemented query relevance judgments and the baseline query relevance judgments on FiRA-2022 fine-grained tuples, the results are quite similar across most metrics. Both sets of relevance judgments resulted in high MRR and Recall scores, indicating that the models effectively ranked relevant documents at the top positions and retrieved a high proportion of relevant documents within the top 10 results.

The slight variations in metrics suggest that our approach to generating query relevance judgments, although kept simple, aligns closely with the baseline method. The consistency in performance metrics between our implemented and baseline relevance judgments highlights the effectiveness and simplicity of our labeling approach.

## Part 3

This part of the project involves implementing an extractive question answering (QA) system using a pre-trained model from the HuggingFace model hub. The goal is to provide text spans that answer given query-passage pairs from the MSMARCO FIRA dataset.

### Implementation

#### Model Selection

The pre-trained model chosen for this task is `deepset/roberta-base-squad2`. This model is designed for extractive QA tasks, making it suitable for our needs.

#### Data Processing

Two datasets were used:
- **`msmarco-fira-21.qrels.qa-answers.tsv`**: Contains query-document pairs with relevance grades and text selections (answers).
- **`msmarco-fira-21.qrels.qa-tuples.tsv`**: Contains query-document pairs with relevance grades, query texts, document texts, and text selections.

The data was loaded and parsed to ensure it was in a suitable format for the model. The `qa-tuples` dataset was grouped by `queryid` to keep only the combination with the highest relevance grade for each query, reducing computational costs.

#### Model Inference

The model was used to tokenize the query-passage pairs and obtain answers. For each pair, the `query_text` and `document_text` were tokenized and fed into the model to generate an answer. The results were stored, including the `queryid`, `documentid`, `relevance-grade`, `answer`, and a `score`.

#### Result Analysis

Here are examples of the model's outputs. Where two of them where successful outputs but one failed.

___
- **queryId**: 1000000
- **documentId**: 7264308
- **query**: where does real insulin come from
- **text selection**: produced by beta cells of the pancreatic islets
- **document text**: Insulin (from the Latin, insula meaning island) is a peptide hormone produced by beta cells of the pancreatic islets, and it is considered to be the main anabolic hormone of the body.
- **computed answer**: where does real insulin come from Insulin (from the Latin
___
- **queryId**: 1000004
- **documentId**: 7264266
- **query**: where does name nora come from
- **text selection**: of English, Greek and Latin origin
- **document text**: What does Nora mean? N ora as a girls' name is pronounced NOR-ah. It is of English, Greek and Latin origin, and the meaning of Nora is light; woman of honor. Short form Eleanora (Greek) light, Honora (Latin) woman of honor, and Leonora. Also used as an independent name. In Scotland, Nora is often used as a feminine form of Norman.
- **computed answer**: English, Greek and Latin
___
- **queryId**: 1000006
- **documentId**: 2298308
- **query**: where does most of the iron ore come from
- **text selection**: Although iron ore resources occur in all the Australian States and Territories, almost 93% of identified resources (totalling 64 billion tonnes) occur in Western Australia, including almost 80% in the Hamersley Province, one of the world's major iron ore provinces
- **document text**: Most of the world's important iron ore resources occur in iron-rich sedimentary rocks known as banded iron formations (BIFs) which are almost exclusively of Precambrian age (i.e. greater than 600 million years old). BIFs occur on all continents.ustralian Resources. Although iron ore resources occur in all the Australian States and Territories, almost 93% of identified resources (totalling 64 billion tonnes) occur in Western Australia, including almost 80% in the Hamersley Province, one of the world's major iron ore provinces.
- **computed answer**: iron-rich sedimentary rocks
___

### Evaluation

#### Metrics

The evaluation metrics used were Exact Match (EM) and F1 Score. The results were as follows:

- **Exact Match**: 0.1265841013824885
- **F1 Score**: 0.47531292888185606

These metrics were computed by comparing the model's answers with the ground truth answers from `msmarco-fira-21.qrels.qa-answers.tsv`.

#### Problems & Solutions

- **Problem**: The model sometimes generated incomplete or incorrect answers.
    - **Solution**: Ensure that the tokenization and input preparation were correct and consider post-processing steps to refine the answers.

- **Problem**: High computational costs for processing large datasets.
    - **Solution**: Grouped data by `queryid` and kept only the combination with the highest relevance grade to reduce the dataset size.

### Conclusion

The extractive QA system was successfully implemented using the `deepset/roberta-base-squad2` model. Despite some challenges, the system was able to generate answers for query-passage pairs with reasonable accuracy, as indicated by the evaluation metrics.
